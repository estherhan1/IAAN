#!/usr/bin/env python3
"""Offline: answer-letter-level contrastive decoding + prob ensembles from a
confpick margins.jsonl.

Readouts evaluated (letter chosen from the 4 letter logits, prefill position):
  * prefill-argmax per config (consistency check vs generate+parse accuracy)
  * ensemble: mean prob across steered configs
  * contrastive: argmax_L [ logit_cfg(L) - lam * logit_baseline(L) ]  per config
  * contrastive ensemble: mean over steered configs of the contrasted logits

All label-free. GT/ALA scored as in data/eval.py.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp_eval", os.path.join(HERE, "..", "data", "eval.py"))
vp_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp_eval)
LETTERS = ["A", "B", "C", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lams", default="0,0.25,0.5,0.75,1,1.5,2")
    args = ap.parse_args()

    data = {it["id"]: it for it in json.load(open(args.dataset))}
    margins = {}
    for line in open(os.path.join(args.dir, "margins.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        margins[r.pop("id")] = r

    ids = [i for i in data if i in margins and margins[i]]
    n = len(ids)
    tags = sorted({t for i in ids for t in margins[i]})
    steered = [t for t in tags if t != "baseline"]
    print(f"{n} items | configs: {tags}\n")

    def score(pick_fn, name):
        gt_ok = ala_ok = used = 0
        for i in ids:
            L = pick_fn(i)
            if L is None:
                continue
            used += 1
            item = data[i]
            gt_ok += int(L == vp_eval.gt_letter(item))
            ala_ok += int(L in vp_eval.adv_letters(item))
        print(f"  {name:<52} gt_acc={100.0*gt_ok/n:.2f}  ala={100.0*ala_ok/n:.2f}  (n={used})")

    def letters_of(i, t):
        e = margins[i].get(t)
        return e["letters"] if e else None

    print("  prefill-argmax per config:")
    for t in tags:
        score(lambda i, t=t: (max(l, key=l.get) if (l := letters_of(i, t)) else None), t)

    print("\n  prob-ensemble over steered:")
    def prob_ens(i):
        acc = {L: 0.0 for L in LETTERS}
        m = 0
        for t in steered:
            l = letters_of(i, t)
            if not l:
                continue
            mx = max(l.values())
            Z = sum(math.exp(v - mx) for v in l.values())
            for L in l:
                acc[L] += math.exp(l[L] - mx) / Z
            m += 1
        return max(acc, key=acc.get) if m else None
    score(prob_ens, "mean-prob ensemble (steered)")

    for lam in [float(x) for x in args.lams.split(",")]:
        print(f"\n  contrastive lam={lam:g}:")
        for t in steered:
            def cd(i, t=t, lam=lam):
                ls, lb = letters_of(i, t), letters_of(i, "baseline")
                if not ls or not lb:
                    return None
                return max(LETTERS, key=lambda L: ls.get(L, -1e9) - lam * lb.get(L, 0.0))
            score(cd, f"{t} - {lam:g}*baseline")
        def cd_ens(i, lam=lam):
            lb = letters_of(i, "baseline")
            if not lb:
                return None
            acc = {L: 0.0 for L in LETTERS}
            m = 0
            for t in steered:
                ls = letters_of(i, t)
                if not ls:
                    continue
                for L in LETTERS:
                    acc[L] += ls.get(L, 0.0)
                m += 1
            if not m:
                return None
            return max(LETTERS, key=lambda L: acc[L] / m - lam * lb.get(L, 0.0))
        score(cd_ens, f"ENSEMBLE(steered) - {lam:g}*baseline")


if __name__ == "__main__":
    main()
