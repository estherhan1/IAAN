#!/usr/bin/env python3
"""AF3 letter-logit collection for Steer-CD (cross-model validation).

Adapted from af3_perq_official.py (official prompt, HF AudioFlamingo3). AF3
answers as "(A) ..." so the letter position is AFTER a forced "(" token: we
append the "(" token id to the prompt and read the A-D letter logits at the
last position. Variants: 'real' (unsteered) plus steered configs
'<method>:<K>:<alpha>' with method in {content, meanplustvar} (selection on
audio_tower fc2, per-clip real-vs-noise contrast, z <- alpha*z on top-K).

Output: letters.jsonl, one record per clip: {"id", "<variant>": {"letters",
"margin", "top"}} — same schema as collect_letter_logits.py, so the same
offline CD evaluation applies.

Run in the af3 conda env from the VoxParadox repo dir.
"""
from __future__ import annotations
import argparse, json, os, sys

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

CHOICE_LETTERS = "ABCD"
EPS = 1e-6


def build_question_text(item):
    choices = " ".join(
        f"({L}): {item[f'choice_{L.lower()}']}."
        for L in CHOICE_LETTERS if item.get(f"choice_{L.lower()}"))
    return f"\n{item['question']} {choices}"


def read_wav(path):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    return wav.astype("float32")


def make_noise(wav, idx):
    rng = np.random.default_rng(1000 + idx)
    std = float(wav.std()) or 1.0
    return rng.standard_normal(len(wav)).astype("float32") * std


def z_all(d):
    allv = torch.cat([v for v in d.values()])
    mu, sd = allv.mean(), allv.std() + EPS
    return {i: (v - mu) / sd for i, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/audio-flamingo-3-hf")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--audio-base", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="real,content:50:8,meanplustvar:50:8,meanplustvar:50:16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    audio_base = args.audio_base or os.path.dirname(os.path.abspath(args.data_path))
    samples = json.load(open(args.data_path))
    if args.limit is not None:
        samples = samples[:args.limit]

    steer_cfgs, want_real = [], False
    for v in args.variants.split(","):
        v = v.strip()
        if v == "real":
            want_real = True
        elif v:
            m, K, a = v.split(":")
            steer_cfgs.append((f"{m}_K{K}_a{a}", m, int(K), float(a)))
    need_var = any(m == "meanplustvar" for _, m, _, _ in steer_cfgs)

    dtype = getattr(torch, args.dtype)
    print(f"[af3-cd] loading {args.model}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, device_map="cuda").eval()
    enc = model.audio_tower
    n_layers = len(enc.layers)
    n_feat = enc.layers[0].fc2.in_features
    dev, dt = model.device, model.dtype
    print(f"[af3-cd] audio_tower {n_layers} layers x {n_feat}", flush=True)

    tok = proc.tokenizer
    def one_id(s):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"{s!r} -> {ids}"
        return ids[0]
    lids = {L: one_id(L) for L in CHOICE_LETTERS}
    paren_id = one_id("(")
    print(f"[af3-cd] letter ids {lids} paren {paren_id}", flush=True)

    caps_m, caps_v = {}, {}
    def cap(i):
        def h(_m, inp):
            x = inp[0].detach().float()
            caps_m[i] = x.mean(dim=(0, 1)).cpu()
            if need_var:
                caps_v[i] = x.var(dim=1, unbiased=False).mean(0).cpu()
        return h

    @torch.no_grad()
    def encode(wav):
        caps_m.clear(); caps_v.clear()
        f = proc.feature_extractor([wav], sampling_rate=16000, return_tensors="pt")
        inf = f["input_features"].to(dev, dt)
        mask = f["attention_mask"].to(dev)
        hs = [enc.layers[i].fc2.register_forward_pre_hook(cap(i)) for i in range(n_layers)]
        try:
            enc(inf, input_features_mask=mask)
        finally:
            for h in hs:
                h.remove()
        return {i: caps_m[i].clone() for i in caps_m}, \
               ({i: caps_v[i].clone() for i in caps_v} if need_var else None)

    def steer_hook(gvec):
        gv = gvec.to(dev, dt)
        def h(_m, inp):
            return (inp[0] * gv,) + tuple(inp[1:])
        return h

    def build_inputs(it, wav):
        conversation = [[{
            "role": "user",
            "content": [
                {"type": "text", "text": build_question_text(it)},
                {"type": "audio", "audio": wav},
            ],
        }]]
        inputs = proc.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_dict=True, sampling_rate=16000).to(model.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dt)
        # force "(" so the next position is the letter
        pt = torch.tensor([[paren_id]], device=model.device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], pt], dim=1)
        if "attention_mask" in inputs:
            am = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1)
        return inputs

    @torch.no_grad()
    def letter_rec(inputs):
        logits = model(**inputs, return_dict=True).logits[0, -1, :].float()
        ll = {L: float(logits[lids[L]]) for L in CHOICE_LETTERS}
        order = sorted(ll.values(), reverse=True)
        return {"letters": ll, "margin": order[0] - order[1], "top": max(ll, key=ll.get)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    try:
        for j, it in enumerate(tqdm(samples)):
            sid = it["id"]
            ap_ = os.path.join(audio_base, it["audio_path"])
            rec = {"id": sid}
            if not os.path.exists(ap_):
                fout.write(json.dumps(rec) + "\n"); fout.flush(); continue
            try:
                wav = read_wav(ap_)
                inputs = build_inputs(it, wav)
                if want_real:
                    rec["real"] = letter_rec(inputs)
                if steer_cfgs:
                    rm, rv = encode(wav)
                    nm, nv = encode(make_noise(wav, j))
                    scores = {}
                    content = {i: rm[i] - nm[i] for i in rm}
                    scores["content"] = content
                    if need_var:
                        za = z_all(content)
                        zb = z_all({i: rv[i] - nv[i] for i in rv})
                        scores["meanplustvar"] = {i: za[i] + zb[i] for i in za}
                    for (tag, m, K, a) in steer_cfgs:
                        d = scores[m]
                        flat = [(d[i][k].item(), i, k) for i in d for k in range(d[i].numel())]
                        flat.sort(reverse=True)
                        g = {i: torch.ones(n_feat) for i in range(n_layers)}
                        for _, i, k in flat[:K]:
                            g[i][k] = a
                        hs = [enc.layers[i].fc2.register_forward_pre_hook(steer_hook(g[i]))
                              for i in range(n_layers)]
                        try:
                            rec[tag] = letter_rec(inputs)
                        finally:
                            for h in hs:
                                h.remove()
            except Exception as e:
                print(f"[af3-cd] ERR id={sid}: {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr, flush=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    finally:
        fout.close()
    print(f"[af3-cd] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
