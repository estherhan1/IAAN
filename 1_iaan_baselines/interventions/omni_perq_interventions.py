#!/usr/bin/env python3
"""Qwen2.5-Omni VoxParadox per-clip steering with SWAPPABLE INTERVENTIONS.

The selector question is settled (meanplustvar > content, causal ceiling reached);
this sweeps the INTERVENTION applied to the selected neurons. The fc2 pre-hook is
affine (z <- z*gain + add), so every op below is expressible as (gain, add) vectors:

  scale         z <- alpha*z                      (deployed baseline intervention)
  contrast      z <- m_ref + alpha*(z - m_ref)    m_ref = refbank noise mean;
                                                  amplifies only the audio-specific
                                                  deviation, not the noise pedestal
  blankcontrast same, m_ref = blank (silence) mean
  bidir         top-K amplified by alpha AND bottom-K (most noise-leaning score)
                attenuated by beta (<1)
  suppress      ONLY bottom-K attenuated by alpha (<1); diagnostic
  graded        top-K with gain 1 + (alpha-1)*score_i/score_max (soft ranking,
                use larger K)
  contrastbidir contrast on top-K + attenuate bottom-K by beta

Config grammar (comma-separated):  method:op:K:alpha[:beta]   or 'baseline'
Same official protocol / eval as omni_perq_methods.py; scoring stats are computed
once per clip and shared across all configs.
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
from omni_perq_methods import encode_stats, noise_like, build_scores, EPS, EVAL_PY


def topk_indices(score, K, largest=True):
    """Global top-K across all sites; returns [(site, idx, value)]."""
    flat = []
    for s, v in score.items():
        v = v.float()
        for i, x in enumerate(v.tolist()):
            flat.append((x, s, i))
    flat.sort(reverse=largest)
    return [(s, i, x) for (x, s, i) in flat[:K]]


def build_intervention(score, refs, op, K, alpha, beta):
    """Return (gain, add) dicts covering all sites."""
    sites = {s: v.numel() for s, v in score.items()}
    gain = {s: torch.ones(D) for s, D in sites.items()}
    add = {s: torch.zeros(D) for s, D in sites.items()}

    if op in ("scale", "contrast", "blankcontrast", "graded", "bidir", "contrastbidir"):
        top = topk_indices(score, K, largest=True)
        if op == "graded":
            vmax = max((v for (_, _, v) in top if v > 0), default=1.0) or 1.0
        if op in ("contrast", "contrastbidir"):
            mref = refs["refbank"]
        elif op == "blankcontrast":
            mref = refs["blank"]
        for (s, i, v) in top:
            if op == "graded":
                a = 1.0 + (alpha - 1.0) * max(0.0, v) / vmax
                gain[s][i] = a
            else:
                gain[s][i] = alpha
                if op in ("contrast", "blankcontrast", "contrastbidir"):
                    add[s][i] = -(alpha - 1.0) * float(mref[s][i])

    if op in ("bidir", "contrastbidir", "suppress"):
        bot = topk_indices(score, K, largest=False)
        g = alpha if op == "suppress" else beta
        for (s, i, _) in bot:
            gain[s][i] = g
    return gain, add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--configs", required=True,
                    help="comma-sep 'method:op:K:alpha[:beta]' or 'baseline'")
    ap.add_argument("--noise-seeds", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"; torch.cuda.set_device(device)

    cfgs = []           # (tag, method, op, K, alpha, beta) ; baseline -> None
    methods = set()
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

    print(f"[intv] loading {args.model_path}", flush=True)
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
    print(f"[intv] {len(samples)} clips | configs={tags}", flush=True)

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

    fhs = {t: open(os.path.join(args.output_dir, f"predictions_{t}.jsonl"), "w", encoding="utf-8") for t in tags}

    try:
        for j, sample in enumerate(tqdm(samples)):
            sid = sample["id"]
            audio_path = os.path.join(args.audio_base, sample["audio_path"])
            scores, refs = None, None
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
                except Exception as e:
                    print(f"[intv] encode ERR id={sid}: {type(e).__name__}: {str(e)[:120]}",
                          file=sys.stderr, flush=True)
                    scores = None

            for (tag, spec) in cfgs:
                resp = ""
                if spec is None:
                    try:
                        resp = generate(sample) if scores is not None else ""
                    except Exception:
                        resp = ""
                elif scores is not None and spec[0] in scores:
                    m, op, K, alpha, beta = spec
                    gain, add = build_intervention(scores[m], refs, op, K, alpha, beta)
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
    print("\n=== QWEN OMNI intervention sweep (gt_acc / ala %, Δ vs baseline) ===", flush=True)
    for t in tags:
        r = summary[t]
        dd = "" if not base else f"  Δacc={r['gt_acc']-base['gt_acc']:+.2f}  Δala={r['ala']-base['ala']:+.2f}"
        print(f"  {t:<38} gt_acc={r['gt_acc']:.2f}  ala={r['ala']:.2f}{dd}", flush=True)


if __name__ == "__main__":
    main()
