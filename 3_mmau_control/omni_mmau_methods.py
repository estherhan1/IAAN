#!/usr/bin/env python3
"""MMAU test-mini, Qwen2.5-Omni, per-clip encoder-neuron steering with swappable
scoring (content=IAAN baseline vs tvar/meanplustvar). SAME selection+intervention
machinery as omni_perq_methods.py; only the dataset, prompt, and eval change.

Reports overall AND per-task (sound/music/speech) accuracy so we can diagnose WHY
IAAN-style amplification helps on VoxParadox but not on general audio QA.
"""
from __future__ import annotations
import argparse, json, os, re, sys
import torch
from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig)
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _omni_mod
from qwen_omni_utils import process_mm_info
_omni_mod.check_torch_load_is_safe = lambda *a, **k: None

from omni_eval_voxparadox import SYSTEM_PROMPT
from contra_decode.audio_conditions import load_audio_samples, make_blank
from contra_decode.hooks import ActivationStore, register_mlp_hooks
from contra_decode.modeling import processor_features, run_audio_encoder
from run_perquestion_steering import build_steer, register_steer
from omni_perq_methods import encode_stats, noise_like, build_scores

LETTERS = ["A", "B", "C", "D", "E", "F"]


def build_mmau_text(question, choices):
    labeled = " ".join(f"({LETTERS[i]}): {c}." for i, c in enumerate(choices))
    return (f"{question} {labeled}\nReply with only the letter of the correct option: "
            + ", ".join(f"({LETTERS[i]})" for i in range(len(choices))) + ".")


def parse_pred(resp, choices):
    """Predicted letter -> choice text; fallback to choice-text appearing in resp."""
    tail = resp.strip().splitlines()[-1] if resp.strip() else ""
    m = re.search(r"(?i)(?<![A-Za-z])[\(\[]?\s*([A-F])\s*[\)\].:]?", tail) or \
        re.search(r"(?i)(?<![A-Za-z])[\(\[]?\s*([A-F])\s*[\)\].:]?", resp)
    if m:
        idx = LETTERS.index(m.group(1).upper())
        if 0 <= idx < len(choices):
            return choices[idx]
    low = resp.lower()
    hit = [c for c in choices if c.lower() in low]
    return hit[-1] if hit else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--methods", default="content,tvar,meanplustvar")
    ap.add_argument("--ks", default="50")
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

    print(f"[mmau] loading {args.model_path}", flush=True)
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
    print(f"[mmau] {len(samples)} clips | methods={methods} Ks={Ks} alphas={alphas}", flush=True)

    @torch.no_grad()
    def generate(sample):
        audio_path = os.path.join(args.audio_base, sample["audio_id"])
        text_q = build_mmau_text(sample["question"], sample["choices"])
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
    fhs = {t: open(os.path.join(args.output_dir, f"pred_{t}.jsonl"), "w", encoding="utf-8") for t in tags}

    try:
        for j, sample in enumerate(tqdm(samples)):
            sid = sample["id"]
            audio_path = os.path.join(args.audio_base, sample["audio_id"])
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
                    print(f"[mmau] encode ERR id={sid}: {type(e).__name__}: {str(e)[:100]}",
                          file=sys.stderr, flush=True)
                    scores = None
            rec = {"id": sid, "task": sample.get("task"), "answer": sample["answer"],
                   "choices": sample["choices"]}
            try:
                resp = generate(sample) if scores is not None else ""
            except Exception:
                resp = ""
            fhs["baseline"].write(json.dumps({**rec, "response": resp}, ensure_ascii=False) + "\n")
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
                fhs[tag].write(json.dumps({**rec, "response": resp}, ensure_ascii=False) + "\n")
            for t in tags:
                fhs[t].flush()
    finally:
        for t in tags:
            fhs[t].close()

    # score
    def score_file(path):
        rows = [json.loads(l) for l in open(path)]
        by_task = {}
        tot_c = tot_n = 0
        for r in rows:
            pred = parse_pred(r["response"], r["choices"])
            ok = int(pred.strip().lower() == str(r["answer"]).strip().lower())
            t = r.get("task", "?")
            c, n = by_task.get(t, (0, 0))
            by_task[t] = (c + ok, n + 1)
            tot_c += ok; tot_n += 1
        return {"overall": 100.0 * tot_c / max(1, tot_n),
                "per_task": {t: 100.0 * c / n for t, (c, n) in by_task.items()}}

    summary = {t: score_file(os.path.join(args.output_dir, f"pred_{t}.jsonl")) for t in tags}
    json.dump(summary, open(os.path.join(args.output_dir, "mmau_summary.json"), "w"), indent=2)
    base = summary["baseline"]["overall"]
    print("\n=== MMAU Qwen-Omni encoder-steering (acc %, Δ vs baseline) ===", flush=True)
    for t in tags:
        s = summary[t]
        pt = " ".join(f"{k}={v:.1f}" for k, v in sorted(s["per_task"].items()))
        print(f"  {t:<22} overall={s['overall']:.2f}  Δ={s['overall']-base:+.2f}   [{pt}]", flush=True)


if __name__ == "__main__":
    main()
