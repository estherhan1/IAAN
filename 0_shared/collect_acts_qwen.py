#!/usr/bin/env python3
"""Collect compact per-clip encoder activation statistics for Qwen2.5-Omni.

For each VoxParadox clip we run the audio_tower on:
    real, blank, and R noise seeds (energy-matched Gaussian, official recipe).
Per site (encoder layer fc2 input = post-activation FFN neuron), we store the
temporal mean and temporal variance as fp16 vectors:

    mean_real, var_real, mean_noise1, var_noise1, mean_noisebank, mean_blank

That is everything the A-tier per-clip scores need (SNR/effect-size, temporal
variance, reference-bank averaging, per-layer norm, presence/content baselines)
plus mean_real + labels for the B-tier global-mask methods (discriminative,
probe, OMP). Saved once, reused by characterize.py.

Run (omni env) from the VoxParadox repo dir:
  SHIM=.../voxparadox/VoxParadox/_steer_shim ; SCR=.../contra_decode/scripts
  PYTHONPATH=$SHIM:$SCR/steering:$SCR/ablation:$SCR/pipeline:$PWD \
  HF_HOME=/livingrooms/khlu/hf_cache \
  python methods_exp/collect_acts_qwen.py --data data/voxparadox_dev300.json \
      --audio_base data --out methods_exp/out/acts_dev300.pt --noise-seeds 4
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig)
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _omni_mod
_omni_mod.check_torch_load_is_safe = lambda *a, **k: None

from contra_decode.audio_conditions import load_audio_samples, make_blank
from contra_decode.hooks import ActivationStore, register_mlp_hooks
from contra_decode.modeling import processor_features, run_audio_encoder


def noise_like(wav, idx, seed):
    g = torch.Generator(device="cpu").manual_seed(1000 * seed + idx)
    n = torch.randn(wav.shape, generator=g, dtype=torch.float32)
    std = wav.std().clamp_min(1e-6)
    return n / n.std().clamp_min(1e-6) * std


@torch.no_grad()
def encode_stats(model, store, enc_name, samples, sr):
    """Return (mean, var) dict per site for one waveform."""
    cap = register_mlp_hooks(model.thinker.audio_tower, store, prefix=enc_name)
    try:
        store.clear()
        run_audio_encoder(model, processor_features(model, samples, sr))
        means = store.pooled_means()
        tvars = store.pooled_temporal_vars()
    finally:
        cap.remove()
    return means, tvars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--noise-seeds", type=int, default=4,
                    help="how many noise draws to average for the reference bank")
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(device)
    print(f"[collect] loading {args.model_path}", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    config = Qwen2_5OmniConfig.from_pretrained(args.model_path)
    config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    model.processor = processor
    enc_name = "thinker.audio_tower"
    store = ActivationStore()

    samples = json.load(open(args.data))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[collect] {len(samples)} clips, {args.noise_seeds} noise seeds", flush=True)

    records = []
    site_order = None
    for j, s in enumerate(tqdm(samples)):
        ap_path = os.path.join(args.audio_base, s["audio_path"])
        if not os.path.exists(ap_path):
            continue
        try:
            real = load_audio_samples(ap_path)
            wav, sr = real.samples, real.sample_rate
            m_real, v_real = encode_stats(model, store, enc_name, wav, sr)
            m_blank, _ = encode_stats(model, store, enc_name, make_blank(real).samples, sr)
            # noise bank
            m_noise_seeds, v_noise1 = [], None
            for k in range(args.noise_seeds):
                nz = noise_like(wav, j, seed=2 + k)
                mn, vn = encode_stats(model, store, enc_name, nz, sr)
                m_noise_seeds.append(mn)
                if k == 0:
                    v_noise1 = vn
            if site_order is None:
                site_order = list(m_real.keys())
            def stack(dlist_or_d, agg="mean"):
                if isinstance(dlist_or_d, list):
                    return {st: torch.stack([d[st] for d in dlist_or_d]).mean(0) for st in site_order}
                return dlist_or_d
            m_noisebank = stack(m_noise_seeds)
            m_noise1 = m_noise_seeds[0]
            rec = {
                "id": s["id"], "task": s["task_name"],
                "gt": s["answer_gt"], "adv": s.get("adversarial_labels", []),
                "mean_real": {st: m_real[st].half() for st in site_order},
                "var_real": {st: v_real[st].half() for st in site_order},
                "mean_noise1": {st: m_noise1[st].half() for st in site_order},
                "var_noise1": {st: v_noise1[st].half() for st in site_order},
                "mean_noisebank": {st: m_noisebank[st].half() for st in site_order},
                "mean_blank": {st: m_blank[st].half() for st in site_order},
            }
            records.append(rec)
        except Exception as e:
            print(f"[collect] ERR id={s.get('id')}: {type(e).__name__}: {str(e)[:120]}",
                  file=sys.stderr, flush=True)

    out = {"model": args.model_path, "site_order": site_order,
           "noise_seeds": args.noise_seeds, "records": records}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"[collect] saved {args.out}  ({len(records)} clips)", flush=True)


if __name__ == "__main__":
    main()
