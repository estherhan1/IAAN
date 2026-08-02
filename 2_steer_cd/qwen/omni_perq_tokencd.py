#!/usr/bin/env python3
"""Per-token contrastive decoding with encoder steering (Qwen2.5-Omni, VoxParadox).

Two branches share the identical token sequence; they differ only in the audio
representation (steered vs unsteered encoder), i.e. only through their KV caches:
  * prefill once WITH steering hooks -> cache_s, once WITHOUT -> cache_b
  * each greedy step: next = argmax( logits_s - lam * logits_b ), append the SAME
    token to both branches (steering hooks are irrelevant after prefill: the
    audio encoder only runs during prefill).

lam=0 reproduces plain steered greedy decoding (used as a correctness check
against the hooked model.generate outputs). Outputs predictions_<tag>.jsonl and
runs the official eval, so this is the generate+parse-protocol version of the
Steer-CD letter readout.
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
from omni_perq_methods import encode_stats, noise_like, build_scores, EVAL_PY
from omni_perq_interventions import build_intervention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--expert", default="meanplustvar:scale:50:16")
    ap.add_argument("--lams", default="0,0.75")
    ap.add_argument("--plaus", type=float, default=0.0,
                    help="adaptive plausibility: contrast only among tokens with "
                         "p_expert >= plaus * max p_expert (0 = off)")
    ap.add_argument("--noise-seeds", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"; torch.cuda.set_device(device)
    parts = args.expert.split(":")
    exp_m, exp_op, exp_K, exp_a = parts[0], parts[1], int(parts[2]), float(parts[3])
    exp_b = float(parts[4]) if len(parts) > 4 else 1.0
    lams = [float(x) for x in args.lams.split(",")]
    psuf = f"_p{args.plaus:g}" if args.plaus > 0 else ""
    tags = [f"tokencd_l{l:g}{psuf}" for l in lams]

    print(f"[tokencd] loading {args.model_path}", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    config = Qwen2_5OmniConfig.from_pretrained(args.model_path)
    config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)
    model.eval(); model.processor = processor
    thinker = model.thinker
    encoder = thinker.audio_tower
    enc_name = "thinker.audio_tower"
    store = ActivationStore()

    eos = thinker.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.discard(None)
    if not eos_ids:
        eos_ids = {processor.tokenizer.eos_token_id}
    print(f"[tokencd] eos ids {sorted(eos_ids)}", flush=True)

    samples = json.load(open(args.data_path))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[tokencd] {len(samples)} clips | expert={args.expert} lams={lams}", flush=True)

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
    def prefill(inputs, steer=None):
        """Returns (past_key_values, last_logits). Fresh rope state per call."""
        thinker.rope_deltas = None
        handles = None
        if steer is not None:
            handles = register_steer(encoder, steer[0], steer[1], enc_name)
        try:
            out = thinker(**inputs, use_cache=True, return_dict=True)
        finally:
            if handles is not None:
                handles.remove()
        return out.past_key_values, out.logits[0, -1, :].float()

    @torch.no_grad()
    def step(tok, pkv, pos, mask_len):
        out = thinker(input_ids=torch.tensor([[tok]], device=device),
                      attention_mask=torch.ones((1, mask_len + 1), dtype=torch.long, device=device),
                      past_key_values=pkv, use_cache=True, return_dict=True,
                      cache_position=torch.tensor([pos], device=device))
        return out.past_key_values, out.logits[0, -1, :].float()

    @torch.no_grad()
    def cd_generate(inputs, steer, lam):
        plen = inputs.input_ids.shape[1]
        pkv_s, log_s = prefill(inputs, steer=steer)
        rope_deltas = thinker.rope_deltas   # identical token layout for both branches
        if lam != 0:
            pkv_b, log_b = prefill(inputs, steer=None)
            thinker.rope_deltas = rope_deltas
        toks = []
        for t in range(args.max_new_tokens):
            if lam == 0:
                logits = log_s
            else:
                logits = log_s - lam * log_b
                if args.plaus > 0:
                    p = torch.softmax(log_s, dim=-1)
                    logits = logits.masked_fill(p < args.plaus * p.max(), float("-inf"))
            tok = int(logits.argmax())
            if tok in eos_ids:
                break
            toks.append(tok)
            pos = plen + t
            pkv_s, log_s = step(tok, pkv_s, pos, pos)
            if lam != 0:
                pkv_b, log_b = step(tok, pkv_b, pos, pos)
        return processor.tokenizer.decode(toks, skip_special_tokens=True).strip()

    fhs = {t: open(os.path.join(args.output_dir, f"predictions_{t}.jsonl"), "w", encoding="utf-8") for t in tags}
    try:
        for j, sample in enumerate(tqdm(samples)):
            sid = sample["id"]
            audio_path = os.path.join(args.audio_base, sample["audio_path"])
            steer = None
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
                    scores = build_scores(real_m, real_v, noise_m, noise_v, refbank_m, blank_m, [exp_m])
                    refs = {"refbank": refbank_m, "blank": blank_m}
                    steer = build_intervention(scores[exp_m], refs, exp_op, exp_K, exp_a, exp_b)
                except Exception as e:
                    print(f"[tokencd] encode ERR id={sid}: {type(e).__name__}: {str(e)[:120]}",
                          file=sys.stderr, flush=True)
                    steer = None
            for lam, tag in zip(lams, tags):
                resp = ""
                if steer is not None:
                    try:
                        inputs = build_inputs(sample)
                        resp = cd_generate(inputs, steer, lam)
                    except Exception as e:
                        print(f"[tokencd] gen ERR id={sid} l={lam}: {type(e).__name__}: {str(e)[:120]}",
                              file=sys.stderr, flush=True)
                        resp = ""
                fhs[tag].write(json.dumps({"id": sid, "response": resp}, ensure_ascii=False) + "\n")
                fhs[tag].flush()
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
    for t in tags:
        print(f"  {t:<18} gt_acc={summary[t]['gt_acc']:.2f}  ala={summary[t]['ala']:.2f}", flush=True)


if __name__ == "__main__":
    main()
