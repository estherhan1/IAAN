#!/usr/bin/env python3
"""Qwen2.5-Omni VoxParadox per-clip steering with SWAPPABLE neuron-scoring methods.

Same official protocol / eval as omni_perq_official.py, but the per-clip neuron
score fed to top-K selection can be any of:

  content       real - noise        (the deployed baseline; signed mean diff)
  presence      real - blank
  refbank       real - mean(noise over R seeds)      (lower-variance mean diff)
  snr           (real - noise) / sqrt(var_real + var_noise)   (effect size)
  snr_refbank   (real - refbank)  / sqrt(var_real + var_noise)
  tvar          var_real - var_noise                 (temporal-dynamics contrast)
  meanplustvar  z(real-noise) + z(var_real-var_noise)  (mean + dynamics, z per clip)
  content_ln    per-layer-normalized content          (removes cross-layer scale bias)

Selection is always top-K by the score value and the intervention is z<-alpha*z on
the selected neurons (reuses build_steer/register_steer, exactly like the official
method). One encode per clip is shared across all methods x K x alpha.

Run (omni env), from the repo dir, with the shim on PYTHONPATH (see collect script).
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig)
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _omni_mod
from qwen_omni_utils import process_mm_info
_omni_mod.check_torch_load_is_safe = lambda *a, **k: None

from omni_eval_voxparadox import SYSTEM_PROMPT, build_text
from contra_decode.audio_conditions import load_audio_samples, make_blank
from contra_decode.hooks import ActivationStore, register_mlp_hooks
from contra_decode.modeling import processor_features, run_audio_encoder
from run_perquestion_steering import build_steer, register_steer

EPS = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_PY = os.path.join(HERE, "..", "data", "eval.py")


def noise_like(wav, idx, seed):
    g = torch.Generator(device="cpu").manual_seed(1000 * seed + idx)
    n = torch.randn(wav.shape, generator=g, dtype=torch.float32)
    return n / n.std().clamp_min(1e-6) * wav.std().clamp_min(1e-6)


@torch.no_grad()
def encode_stats(model, store, enc_name, samples, sr):
    cap = register_mlp_hooks(model.thinker.audio_tower, store, prefix=enc_name)
    try:
        store.clear()
        run_audio_encoder(model, processor_features(model, samples, sr))
        return store.pooled_means(), store.pooled_temporal_vars()
    finally:
        cap.remove()


def layer_norm_dict(dvec):
    """Z-score each site's vector within that site."""
    return {s: (v - v.mean()) / (v.std() + EPS) for s, v in dvec.items()}


def z_dict(dvec):
    """Z-score across ALL neurons of a clip (concatenated), preserving per-site shape."""
    allv = torch.cat([v for v in dvec.values()])
    mu, sd = allv.mean(), allv.std() + EPS
    return {s: (v - mu) / sd for s, v in dvec.items()}


def build_scores(real_m, real_v, noise_m, noise_v, refbank_m, blank_m, methods):
    sites = list(real_m.keys())
    out = {}
    content = {s: real_m[s] - noise_m[s] for s in sites}
    if "content" in methods:
        out["content"] = content
    if "presence" in methods:
        out["presence"] = {s: real_m[s] - blank_m[s] for s in sites}
    if "refbank" in methods:
        out["refbank"] = {s: real_m[s] - refbank_m[s] for s in sites}
    if "snr" in methods:
        out["snr"] = {s: (real_m[s] - noise_m[s]) / (real_v[s] + noise_v[s] + EPS).sqrt() for s in sites}
    if "snr_refbank" in methods:
        out["snr_refbank"] = {s: (real_m[s] - refbank_m[s]) / (real_v[s] + noise_v[s] + EPS).sqrt() for s in sites}
    if "tvar" in methods:
        out["tvar"] = {s: real_v[s] - noise_v[s] for s in sites}
    if "meanplustvar" in methods:
        za = z_dict({s: real_m[s] - noise_m[s] for s in sites})
        zb = z_dict({s: real_v[s] - noise_v[s] for s in sites})
        out["meanplustvar"] = {s: za[s] + zb[s] for s in sites}
    if "content_ln" in methods:
        out["content_ln"] = layer_norm_dict(content)
    if "tvar_real" in methods:                 # raw temporal variance of real (magnitude-ful "tracking" neurons)
        out["tvar_real"] = {s: real_v[s].clone() for s in sites}
    if "content_x_tvar" in methods:            # content magnitude gated UP by temporal dynamics (keeps sign+scale)
        ztv = z_dict({s: real_v[s] - noise_v[s] for s in sites})
        gate = {s: 1.0 + torch.relu(ztv[s]) for s in sites}
        out["content_x_tvar"] = {s: content[s] * gate[s] for s in sites}
    if "content_x_tvarreal" in methods:        # content gated by raw real dynamics
        ztv = z_dict({s: real_v[s] for s in sites})
        gate = {s: 1.0 + torch.relu(ztv[s]) for s in sites}
        out["content_x_tvarreal"] = {s: content[s] * gate[s] for s in sites}
    if "meanplus2tvar" in methods:             # weight dynamics more heavily
        za = z_dict({s: real_m[s] - noise_m[s] for s in sites})
        zb = z_dict({s: real_v[s] - noise_v[s] for s in sites})
        out["meanplus2tvar"] = {s: za[s] + 2.0 * zb[s] for s in sites}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--methods", default="content,refbank,snr,tvar,meanplustvar,content_ln")
    ap.add_argument("--ks", default="50,200")
    ap.add_argument("--alphas", default="8,16")
    ap.add_argument("--noise-seeds", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"; torch.cuda.set_device(device)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    Ks = [int(x) for x in args.ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"[methods] loading {args.model_path}", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    config = Qwen2_5OmniConfig.from_pretrained(args.model_path)
    config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)
    model.eval(); model.processor = processor
    encoder = model.thinker.audio_tower
    enc_name = "thinker.audio_tower"
    store = ActivationStore()

    samples = json.load(open(args.data_path))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[methods] {len(samples)} clips | methods={methods} Ks={Ks} alphas={alphas}", flush=True)

    @torch.no_grad()
    def generate(sample):
        audio_path = os.path.join(args.audio_base, sample["audio_path"])
        text_q = build_text(sample["question"], sample["choice_a"], sample["choice_b"],
                            sample["choice_c"], sample["choice_d"], letter_only_suffix=True)
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "audio", "audio": audio_path},
                                             {"type": "text", "text": text_q}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=False
                           ).to(model.device, model.dtype)
        plen = inputs.input_ids.shape[1]
        gen = model.generate(**inputs, use_audio_in_video=False, return_audio=False,
                             thinker_max_new_tokens=args.max_new_tokens, thinker_do_sample=False)
        seq = gen[0] if isinstance(gen, (tuple, list)) else gen
        if seq.dim() == 2:
            seq = seq[0]
        return processor.tokenizer.decode(seq[plen:], skip_special_tokens=True).strip()

    cfgs = [(m, K, a) for m in methods for K in Ks for a in alphas]
    tags = ["baseline"] + [f"{m}_K{K}_a{a:g}" for (m, K, a) in cfgs]
    fhs = {t: open(os.path.join(args.output_dir, f"predictions_{t}.jsonl"), "w", encoding="utf-8") for t in tags}

    try:
        for j, sample in enumerate(tqdm(samples)):
            sid = sample["id"]
            audio_path = os.path.join(args.audio_base, sample["audio_path"])
            scores = None
            if os.path.exists(audio_path):
                try:
                    real = load_audio_samples(audio_path)
                    wav, sr = real.samples, real.sample_rate
                    real_m, real_v = encode_stats(model, store, enc_name, wav, sr)
                    blank_m, _ = encode_stats(model, store, enc_name, make_blank(real).samples, sr)
                    noise_ms, noise_v = [], None
                    for k in range(args.noise_seeds):
                        mn, vn = encode_stats(model, store, enc_name, noise_like(wav, j, 2 + k), sr)
                        noise_ms.append(mn)
                        if k == 0:
                            noise_v = vn
                    noise_m = noise_ms[0]
                    refbank_m = {s: torch.stack([d[s] for d in noise_ms]).mean(0) for s in real_m}
                    scores = build_scores(real_m, real_v, noise_m, noise_v, refbank_m, blank_m, methods)
                except Exception as e:
                    print(f"[methods] encode ERR id={sid}: {type(e).__name__}: {str(e)[:120]}",
                          file=sys.stderr, flush=True)
                    scores = None

            try:
                resp = generate(sample) if scores is not None else ""
            except Exception:
                resp = ""
            fhs["baseline"].write(json.dumps({"id": sid, "response": resp}, ensure_ascii=False) + "\n")

            for (m, K, a) in cfgs:
                tag = f"{m}_K{K}_a{a:g}"
                resp = ""
                if scores is not None and m in scores:
                    gain, add = build_steer(scores[m], "pos", "scale", K, a)
                    handles = register_steer(encoder, gain, add, enc_name)
                    try:
                        resp = generate(sample)
                    except Exception:
                        resp = ""
                    finally:
                        handles.remove()
                fhs[tag].write(json.dumps({"id": sid, "response": resp}, ensure_ascii=False) + "\n")
            for t in tags:
                fhs[t].flush()
    finally:
        for t in tags:
            fhs[t].close()

    import subprocess
    summary = {}
    for t in tags:
        pred = os.path.join(args.output_dir, f"predictions_{t}.jsonl")
        rep = os.path.join(args.output_dir, f"report_{t}.json")
        subprocess.run([sys.executable, EVAL_PY, "--predictions", pred,
                        "--dataset", args.data_path, "--report", rep], check=True)
        r = json.load(open(rep))
        summary[t] = {"gt_acc": r["overall"]["gt_acc"], "ala": r["overall"]["ala"]}
        json.dump(summary, open(os.path.join(args.output_dir, "methods_summary.json"), "w"), indent=2)

    base = summary.get("baseline", {})
    print("\n=== QWEN OMNI method sweep (gt_acc / ala %, Δ vs baseline) ===", flush=True)
    for t in tags:
        r = summary[t]
        dd = "" if not base else f"  Δacc={r['gt_acc']-base['gt_acc']:+.2f}  Δala={r['ala']-base['ala']:+.2f}"
        print(f"  {t:<22} gt_acc={r['gt_acc']:.2f}  ala={r['ala']:.2f}{dd}", flush=True)


if __name__ == "__main__":
    main()
