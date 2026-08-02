#!/usr/bin/env python3
"""Apply a FIXED global neuron mask (from build_global_masks.py) to every clip.

For each (method, K, alpha): select top-K neurons of that method's global score,
scale by alpha, register the hooks ONCE, then run all clips (the mask does not
depend on the clip). Same official eval as omni_perq_official.py.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import torch
from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig)
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _omni_mod
from qwen_omni_utils import process_mm_info
_omni_mod.check_torch_load_is_safe = lambda *a, **k: None

from omni_eval_voxparadox import SYSTEM_PROMPT, build_text
from run_perquestion_steering import build_steer, register_steer

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_PY = os.path.join(HERE, "..", "data", "eval.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--masks", required=True, help="build_global_masks.py output .pt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--methods", default="content_global,discriminative,probe,probe_attr,omp")
    ap.add_argument("--ks", default="50,200")
    ap.add_argument("--alphas", default="8,16")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"; torch.cuda.set_device(device)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    Ks = [int(x) for x in args.ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    maskdata = torch.load(args.masks, map_location="cpu", weights_only=False)
    mask_scores = maskdata["scores"]

    print(f"[gmask] loading {args.model_path}", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    config = Qwen2_5OmniConfig.from_pretrained(args.model_path)
    config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)
    model.eval(); model.processor = processor
    encoder = model.thinker.audio_tower
    enc_name = "thinker.audio_tower"

    samples = json.load(open(args.data_path))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[gmask] {len(samples)} clips | methods={methods} Ks={Ks} alphas={alphas}", flush=True)

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

    # baseline first
    summary = {}
    def run_tag(tag, handles):
        fh = open(os.path.join(args.output_dir, f"predictions_{tag}.jsonl"), "w", encoding="utf-8")
        for sample in tqdm(samples, desc=tag):
            try:
                resp = generate(sample)
            except Exception:
                resp = ""
            fh.write(json.dumps({"id": sample["id"], "response": resp}, ensure_ascii=False) + "\n")
            fh.flush()
        fh.close()
        rep = os.path.join(args.output_dir, f"report_{tag}.json")
        subprocess.run([sys.executable, EVAL_PY, "--predictions",
                        os.path.join(args.output_dir, f"predictions_{tag}.jsonl"),
                        "--dataset", args.data_path, "--report", rep], check=True)
        r = json.load(open(rep))
        summary[tag] = {"gt_acc": r["overall"]["gt_acc"], "ala": r["overall"]["ala"]}
        json.dump(summary, open(os.path.join(args.output_dir, "gmask_summary.json"), "w"), indent=2)

    run_tag("baseline", None)

    for m in methods:
        if m not in mask_scores:
            print(f"[gmask] SKIP {m} (not in mask file)"); continue
        for K in Ks:
            for a in alphas:
                tag = f"{m}_K{K}_a{a:g}"
                gain, add = build_steer(mask_scores[m], "pos", "scale", K, a)
                handles = register_steer(encoder, gain, add, enc_name)
                try:
                    run_tag(tag, handles)
                finally:
                    handles.remove()

    base = summary.get("baseline", {})
    print("\n=== QWEN OMNI GLOBAL-MASK sweep (gt_acc / ala %, Δ vs baseline) ===", flush=True)
    for t, r in summary.items():
        dd = "" if not base else f"  Δacc={r['gt_acc']-base['gt_acc']:+.2f}  Δala={r['ala']-base['ala']:+.2f}"
        print(f"  {t:<22} gt_acc={r['gt_acc']:.2f}  ala={r['ala']:.2f}{dd}", flush=True)


if __name__ == "__main__":
    main()
