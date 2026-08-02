#!/usr/bin/env python3
"""B3: causal / attribution-based global neuron mask for Qwen2.5-Omni.

For each labeled clip we do ONE forward+backward through the full thinker with the
real audio and measure, for each encoder FFN neuron z_i (fc2 input), the
first-order effect of amplifying it on the margin

    target = logit(gt letter) - logit(best adversarial letter)

via input x gradient:  attr_i = z_i * d(target)/d(z_i)  (summed over time, mean over
clips). A large positive attr_i means "scaling this neuron up increases the model's
preference for the correct answer over the transcript-implied one" -- exactly the
objective the method optimizes, selected causally rather than by a real-vs-noise gap.

This uses the gt label, so it is an ORACLE / upper-bound selector: it bounds how much
a better (label-aware) neuron finder could gain over the label-free contrast.

Output .pt matches build_global_masks.py: {"scores": {"causal": {site: vec},
"causal_abs": {...}}, "site_order": [...]}.
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
from contra_decode.hooks import _is_linear_pair_block, _is_swiglu_block
from run_label_calibration import discover_letter_token_ids


def letter_of(item, value):
    for L in "ABCD":
        if str(item.get(f"choice_{L.lower()}", "")).strip() == str(value).strip():
            return L
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()

    device = "cuda:0"; torch.cuda.set_device(device)
    print(f"[causal] loading {args.model_path}", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    config = Qwen2_5OmniConfig.from_pretrained(args.model_path)
    config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)
    model.eval(); model.processor = processor
    # Freeze ALL parameters: we only need gradients w.r.t. the captured encoder
    # activations, not the 7B params. Otherwise .grad buffers for every param would
    # be allocated (~17GB) and OOM. We use torch.autograd.grad on the activations.
    for p in model.parameters():
        p.requires_grad_(False)
    enc = model.thinker.audio_tower
    enc_name = "thinker.audio_tower"
    lids = discover_letter_token_ids(processor)
    print("[causal] letter ids", lids, flush=True)

    # locate the FFN neuron sites and register capture-with-grad hooks
    site_modules = []
    for name, module in enc.named_modules():
        full = f"{enc_name}.{name}" if name else enc_name
        if _is_linear_pair_block(module):
            site_modules.append((full, module.fc2))
        elif _is_swiglu_block(module):
            site_modules.append((full, module.down_proj))
    site_order = [s for s, _ in site_modules]
    print(f"[causal] {len(site_order)} sites", flush=True)

    captured = {}
    def mk_hook(site):
        def hook(_m, inputs):
            z = inputs[0]
            z.requires_grad_(True)   # make it a differentiable input; params are frozen
            captured[site] = z
            return None
        return hook

    handles = [mod.register_forward_pre_hook(mk_hook(s)) for s, mod in site_modules]

    samples = json.load(open(args.data_path))
    if args.limit:
        samples = samples[:args.limit]

    attr_sum = {s: None for s in site_order}
    attr_abs = {s: None for s in site_order}
    n_used = 0

    for sample in tqdm(samples):
        audio_path = os.path.join(args.audio_base, sample["audio_path"])
        if not os.path.exists(audio_path):
            continue
        gt_L = letter_of(sample, sample["answer_gt"])
        adv_vals = sample.get("adversarial_labels", []) or []
        adv_L = letter_of(sample, adv_vals[0]) if adv_vals else None
        if gt_L is None or gt_L not in lids:
            continue
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
        captured.clear()
        try:
            with torch.enable_grad():
                out = model.thinker(**inputs, return_dict=True, use_cache=False)
                logits = out.logits[0, -1, :].float()
                target = logits[lids[gt_L]]
                if adv_L is not None and adv_L in lids:
                    target = target - logits[lids[adv_L]]
                sites_here = [s for s in site_order if s in captured]
                zs = [captured[s] for s in sites_here]
                grads = torch.autograd.grad(target, zs, retain_graph=False, allow_unused=True)
            with torch.no_grad():
                for s, z, g in zip(sites_here, zs, grads):
                    if g is None:
                        continue
                    contrib = (z.detach().float() * g.detach().float())        # (B,T,D)
                    v = contrib.reshape(-1, contrib.shape[-1]).sum(0).cpu()    # (D,)
                    attr_sum[s] = v if attr_sum[s] is None else attr_sum[s] + v
                    attr_abs[s] = v.abs() if attr_abs[s] is None else attr_abs[s] + v.abs()
            n_used += 1
        except torch.cuda.OutOfMemoryError:
            print(f"[causal] OOM on id={sample.get('id')}, skipping", file=sys.stderr, flush=True)
        finally:
            captured.clear()
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    print(f"[causal] used {n_used} clips", flush=True)
    scores = {"causal": {s: (attr_sum[s] / max(1, n_used)) for s in site_order if attr_sum[s] is not None},
              "causal_abs": {s: (attr_abs[s] / max(1, n_used)) for s in site_order if attr_abs[s] is not None}}
    torch.save({"scores": scores, "site_order": [s for s in site_order if attr_sum[s] is not None]}, args.out)
    print(f"[causal] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
