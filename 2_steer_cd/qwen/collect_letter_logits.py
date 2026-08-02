#!/usr/bin/env python3
"""Collect answer-letter prefill logits for CD readout variants (Qwen2.5-Omni).

Variants (comma-sep --variants):
  real          baseline model, real audio
  blank         baseline model, silence of same length
  noise         baseline model, energy-matched Gaussian noise (seed=2 stream)
  <cfg>         steered on real audio; cfg grammar 'method:op:K:alpha[:beta]'
                as in omni_perq_interventions.py (e.g. meanplustvar:scale:50:16)

Output: letters.jsonl with {"id", "<variant>": {"letters": {A..D}, "top", "margin"}}
No generation — this is the data for offline contrastive-readout evaluation.
Steered variants trigger the per-clip scoring encodes; audio-only variants don't.
"""
from __future__ import annotations
import argparse, json, os, sys
import librosa
import numpy as np
import torch
from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig)
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _omni_mod
from qwen_omni_utils import process_mm_info
_omni_mod.check_torch_load_is_safe = lambda *a, **k: None

from omni_eval_voxparadox import SYSTEM_PROMPT, build_text
from contra_decode.audio_conditions import load_audio_samples, make_blank
from contra_decode.hooks import ActivationStore
from run_perquestion_steering import register_steer
from run_label_calibration import discover_letter_token_ids
from omni_perq_methods import encode_stats, noise_like, build_scores
from omni_perq_interventions import build_intervention

LETTERS = ["A", "B", "C", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--noise-seeds", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = "cuda:0"; torch.cuda.set_device(device)

    audio_variants, steer_cfgs, methods = [], [], set()
    for v in args.variants.split(","):
        v = v.strip()
        if not v:
            continue
        if v in ("real", "blank", "noise"):
            audio_variants.append(v)
        else:
            parts = v.split(":")
            m, op, K, alpha = parts[0], parts[1], int(parts[2]), float(parts[3])
            beta = float(parts[4]) if len(parts) > 4 else 1.0
            tag = f"{m}_{op}_K{K}_a{alpha:g}" + (f"_b{beta:g}" if len(parts) > 4 else "")
            steer_cfgs.append((tag, (m, op, K, alpha, beta)))
            methods.add(m)
    methods = sorted(methods)

    print(f"[cll] loading {args.model_path}", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    config = Qwen2_5OmniConfig.from_pretrained(args.model_path)
    config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)
    model.eval(); model.processor = processor
    encoder = model.thinker.audio_tower
    enc_name = "thinker.audio_tower"
    store = ActivationStore()
    lids = discover_letter_token_ids(processor)

    samples = json.load(open(args.data_path))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[cll] {len(samples)} clips | audio_variants={audio_variants} "
          f"steer={[t for t,_ in steer_cfgs]}", flush=True)

    def build_inputs(sample, audio_arg):
        text_q = build_text(sample["question"], sample["choice_a"], sample["choice_b"],
                            sample["choice_c"], sample["choice_d"], letter_only_suffix=True)
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "audio", "audio": audio_arg},
                                             {"type": "text", "text": text_q}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        return processor(text=text, audio=audios, images=images, videos=videos,
                         return_tensors="pt", padding=True, use_audio_in_video=False
                         ).to(model.device, model.dtype)

    @torch.no_grad()
    def letter_rec(inputs):
        logits = model.thinker(**inputs, return_dict=True, use_cache=False).logits[0, -1, :].float()
        ll = {L: float(logits[lids[L]]) for L in LETTERS if L in lids}
        order = sorted(ll.values(), reverse=True)
        return {"letters": ll, "margin": order[0] - order[1], "top": max(ll, key=ll.get)}

    fout = open(args.out, "w", encoding="utf-8")
    try:
        for j, sample in enumerate(tqdm(samples)):
            sid = sample["id"]
            audio_path = os.path.join(args.audio_base, sample["audio_path"])
            rec = {"id": sid}
            if not os.path.exists(audio_path):
                fout.write(json.dumps(rec) + "\n"); fout.flush(); continue
            try:
                real = load_audio_samples(audio_path)
                wav, sr = real.samples, real.sample_rate
                with torch.no_grad():
                    for v in audio_variants:
                        if v == "real":
                            arr = wav
                        elif v == "blank":
                            arr = make_blank(real).samples
                        else:
                            arr = noise_like(wav, j, 2)
                        arr = arr.numpy() if torch.is_tensor(arr) else np.asarray(arr)
                        arr = arr.astype(np.float32).reshape(-1)
                        if sr != 16000:      # ndarray path in qwen_omni_utils assumes 16 kHz
                            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                        rec[v] = letter_rec(build_inputs(sample, arr))
                if steer_cfgs:
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
                    refs = {"refbank": refbank_m, "blank": blank_m}
                    inputs = build_inputs(sample, audio_path)
                    for (tag, (m, op, K, alpha, beta)) in steer_cfgs:
                        gain, add = build_intervention(scores[m], refs, op, K, alpha, beta)
                        handles = register_steer(encoder, gain, add, enc_name)
                        try:
                            rec[tag] = letter_rec(inputs)
                        finally:
                            handles.remove()
            except Exception as e:
                print(f"[cll] ERR id={sid}: {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr, flush=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    finally:
        fout.close()
    print(f"[cll] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
