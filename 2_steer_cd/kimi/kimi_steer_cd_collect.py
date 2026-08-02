#!/usr/bin/env python3
"""Kimi-Audio letter-logit collection for Steer-CD (cross-model validation).

Kimi answers VoxParadox MCQs with a bare letter, so the first text token IS the
answer: one alm.forward over the prompt gives the letter distribution at the
last position. Variants: 'real' (unsteered) + steered configs
'<method>:<K>:<alpha>' (method in {content, meanplustvar}), selection on the
whisper speech_encoder fc2 neurons per clip (real vs energy-matched noise),
z <- alpha*z on the top-K — identical machinery to kimi_perq_official.py.

Steering hooks are active during prompt building (get_prompt ->
extract_whisper_feat is where the encoder runs), mirroring the official script.

Output: letters.jsonl (same schema as collect_letter_logits.py).
Run in the kimi conda env from the VoxParadox repo dir:
  HF_HOME=/livingrooms/khlu/hf_cache HF_HUB_OFFLINE=1 \
  /home/khlu/miniconda3/envs/kimi/bin/python methods_exp/kimi_steer_cd_collect.py \
      --data_path data/voxparadox_dev300.json --audio_base data \
      --out methods_exp/out/cll_kimi_dev300.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/livingrooms/khlu/kimi_code")
from kimia_infer.api.kimia import KimiAudio

CHOICE_LETTERS = "ABCD"
EPS = 1e-6
DEFAULT_MODEL = ("/livingrooms/khlu/hf_cache/hub/models--moonshotai--Kimi-Audio-7B-Instruct"
                 "/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b")


def build_text(item):
    choices = " ".join(
        f"({L}): {item[f'choice_{L.lower()}']}."
        for L in CHOICE_LETTERS if item.get(f"choice_{L.lower()}"))
    return (f"{item['question']} {choices}\n"
            "Reply with only the letter of the correct option: (A), (B), (C), or (D).")


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
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--audio_base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="real,content:200:12,meanplustvar:200:12")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

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

    print(f"[kimi-cd] loading {args.model_path}", flush=True)
    model = KimiAudio(model_path=args.model_path, load_detokenizer=False)
    enc = model.prompt_manager.whisper_model.speech_encoder
    n_layers = len(enc.layers)
    n_feat = enc.layers[0].fc2.in_features
    dev = torch.cuda.current_device()
    dt = next(enc.parameters()).dtype
    print(f"[kimi-cd] whisper speech_encoder {n_layers} x {n_feat} ({dt})", flush=True)

    tok = model.prompt_manager.text_tokenizer
    def one_id(s):
        ids = tok.encode(s, bos=False, eos=False)   # TikTokenTokenizer signature
        assert len(ids) == 1, f"{s!r} -> {ids}"
        return ids[0]
    lids = {L: one_id(L) for L in CHOICE_LETTERS}
    print(f"[kimi-cd] letter ids {lids}", flush=True)

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
        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        hs = [enc.layers[i].fc2.register_forward_pre_hook(cap(i)) for i in range(n_layers)]
        try:
            model.prompt_manager.extract_whisper_feat(wav_t)
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

    def build_msgs(text_q, audio_path):
        return [
            {"role": "user", "message_type": "text", "content": text_q},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]

    @torch.no_grad()
    def letter_rec(text_q, audio_path):
        """Prompt build (encoder runs here; hooks may be active) + one alm.forward."""
        history = model.prompt_manager.get_prompt(build_msgs(text_q, audio_path),
                                                  output_type="text")
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature
        audio_input_ids = audio_input_ids.to(dev)
        text_input_ids = text_input_ids.to(dev)
        is_continuous_mask = is_continuous_mask.to(dev)
        audio_features = [f.to(dev) for f in audio_features]
        position_ids = (torch.arange(0, audio_input_ids.shape[1], device=dev)
                        .unsqueeze(0).long())
        _, text_logits, _ = model.alm.forward(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=audio_features,
            is_continuous_mask=is_continuous_mask,
            position_ids=position_ids,
            past_key_values=None,
            return_dict=False,
        )
        if text_logits.dim() == 3:                 # [B, T, V] -> last position
            text_logits = text_logits[:, -1, :]
        logits = text_logits.float().reshape(-1)
        ll = {L: float(logits[lids[L]]) for L in CHOICE_LETTERS}
        order = sorted(ll.values(), reverse=True)
        return {"letters": ll, "margin": order[0] - order[1], "top": max(ll, key=ll.get)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    try:
        for j, it in enumerate(tqdm(samples)):
            sid = it["id"]
            apath = os.path.join(args.audio_base, it["audio_path"])
            rec = {"id": sid}
            if not os.path.exists(apath):
                fout.write(json.dumps(rec) + "\n"); fout.flush(); continue
            text_q = build_text(it)
            try:
                if want_real:
                    rec["real"] = letter_rec(text_q, apath)
                if steer_cfgs:
                    wav = read_wav(apath)
                    rm, rv = encode(wav)
                    nm, nv = encode(make_noise(wav, j))
                    scores = {"content": {i: rm[i] - nm[i] for i in rm}}
                    if need_var:
                        za = z_all(scores["content"])
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
                            rec[tag] = letter_rec(text_q, apath)
                        finally:
                            for h in hs:
                                h.remove()
            except Exception as e:
                print(f"[kimi-cd] ERR id={sid}: {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr, flush=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    finally:
        fout.close()
    print(f"[kimi-cd] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
