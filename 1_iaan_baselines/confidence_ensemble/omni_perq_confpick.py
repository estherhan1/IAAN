#!/usr/bin/env python3
"""Per-clip ADAPTIVE CONFIG SELECTION study (Qwen2.5-Omni, VoxParadox official).

Motivation: the oracle union over a handful of steering configs is ~11pp above the
best single config, but plain majority vote captures little of it. This script
tests a label-free selection signal: the model's own letter-logit confidence
margin under each config.

Per clip and per candidate config we do BOTH:
  * one prefill forward -> letter logits at the last position -> margin/probs
  * a full official generate -> predictions_<tag>.jsonl (same as other runners)

Offline we can then evaluate any selection rule exactly (argmax-margin, margin
thresholded fallback-to-default, per-task rules...) because we have every
config's generated answer AND its margin. Margins go to margins.jsonl.

Config grammar identical to omni_perq_interventions.py.
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
from contra_decode.hooks import ActivationStore
from run_perquestion_steering import register_steer
from run_label_calibration import discover_letter_token_ids
from omni_perq_methods import encode_stats, noise_like, build_scores, EVAL_PY
from omni_perq_interventions import build_intervention

LETTERS = ["A", "B", "C", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--noise-seeds", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"; torch.cuda.set_device(device)

    cfgs, methods = [], set()
    for c in args.configs.split(","):
        c = c.strip()
        if not c:
            continue
        if c == "baseline":
            cfgs.append(("baseline", None))
            continue
        parts = c.split(":")
        m, op, K, alpha = parts[0], parts[1], int(parts[2]), float(parts[3])
        beta = float(parts[4]) if len(parts) > 4 else 1.0
        tag = f"{m}_{op}_K{K}_a{alpha:g}" + (f"_b{beta:g}" if len(parts) > 4 else "")
        cfgs.append((tag, (m, op, K, alpha, beta)))
        methods.add(m)
    methods = sorted(methods)
    cfgs.sort(key=lambda x: 0 if x[0] == "baseline" else 1)
    tags = [t for t, _ in cfgs]

    print(f"[confpick] loading {args.model_path}", flush=True)
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
    print("[confpick] letter ids", lids, flush=True)

    samples = json.load(open(args.data_path))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[confpick] {len(samples)} clips | configs={tags}", flush=True)

    def build_inputs(sample):
        audio_path = os.path.join(args.audio_base, sample["audio_path"])
        text_q = build_text(sample["question"], sample["choice_a"], sample["choice_b"],
                            sample["choice_c"], sample["choice_d"], letter_only_suffix=True)
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "audio", "audio": audio_path},
                                             {"type": "text", "text": text_q}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        return processor(text=text, audio=audios, images=images, videos=videos,
                         return_tensors="pt", padding=True, use_audio_in_video=False
                         ).to(model.device, model.dtype)

    @torch.no_grad()
    def letter_margin(inputs):
        logits = model.thinker(**inputs, return_dict=True, use_cache=False).logits[0, -1, :].float()
        ll = {L: float(logits[lids[L]]) for L in LETTERS if L in lids}
        probs = torch.softmax(torch.tensor([ll[L] for L in LETTERS if L in ll]), dim=0).tolist()
        order = sorted(ll.values(), reverse=True)
        return {"letters": ll, "probs": probs, "margin": order[0] - order[1],
                "top": max(ll, key=ll.get)}

    @torch.no_grad()
    def generate(inputs):
        plen = inputs.input_ids.shape[1]
        gen = model.generate(**inputs, use_audio_in_video=False, return_audio=False,
                             thinker_max_new_tokens=args.max_new_tokens, thinker_do_sample=False)
        seq = gen[0] if isinstance(gen, (tuple, list)) else gen
        if seq.dim() == 2:
            seq = seq[0]
        return processor.tokenizer.decode(seq[plen:], skip_special_tokens=True).strip()

    fhs = {t: open(os.path.join(args.output_dir, f"predictions_{t}.jsonl"), "w", encoding="utf-8") for t in tags}
    fmg = open(os.path.join(args.output_dir, "margins.jsonl"), "w", encoding="utf-8")

    try:
        for j, sample in enumerate(tqdm(samples)):
            sid = sample["id"]
            audio_path = os.path.join(args.audio_base, sample["audio_path"])
            scores, refs, inputs = None, None, None
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
                    refs = {"refbank": refbank_m, "blank": blank_m}
                    inputs = build_inputs(sample)
                except Exception as e:
                    print(f"[confpick] encode ERR id={sid}: {type(e).__name__}: {str(e)[:120]}",
                          file=sys.stderr, flush=True)
                    scores = None

            mrec = {"id": sid}
            for (tag, spec) in cfgs:
                resp = ""
                if inputs is not None and (spec is None or (scores is not None and spec[0] in scores)):
                    handles = None
                    try:
                        if spec is not None:
                            m, op, K, alpha, beta = spec
                            gain, add = build_intervention(scores[m], refs, op, K, alpha, beta)
                            handles = register_steer(encoder, gain, add, enc_name)
                        mrec[tag] = letter_margin(inputs)
                        resp = generate(inputs)
                    except Exception:
                        resp = ""
                    finally:
                        if handles is not None:
                            handles.remove()
                fhs[tag].write(json.dumps({"id": sid, "response": resp}, ensure_ascii=False) + "\n")
            fmg.write(json.dumps(mrec, ensure_ascii=False) + "\n")
            for t in tags:
                fhs[t].flush()
            fmg.flush()
    finally:
        for t in tags:
            fhs[t].close()
        fmg.close()

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
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
