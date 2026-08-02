#!/usr/bin/env python3
"""Offline Steer-CD evaluation for any letters.jsonl (VoxParadox format datasets).

letters.jsonl records: {"id", "<variant>": {"letters": {A..D: logit}, ...}, ...}
For each expert variant and each lambda: answer = argmax_L [expert(L) − λ·amateur(L)].
Scores gt_acc and ALA with data/eval.py semantics; optional per-task table.

Usage:
  python methods_exp/eval_cd_letters.py --letters out/cll_test700.jsonl \
      --dataset data/voxparadox_test700.json --amateur real \
      --experts meanplustvar_scale_K50_a16,meanplustvar_scale_K100_a16 \
      --lams 0,0.25,0.5,0.75,1,1.5 [--per-task-for meanplustvar_scale_K100_a16:0.75]
"""
from __future__ import annotations
import argparse, importlib.util, json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp_eval", os.path.join(HERE, "..", "data", "eval.py"))
vp_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp_eval)
LET = ["A", "B", "C", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--amateur", default="real")
    ap.add_argument("--experts", default=None, help="comma-sep variant tags; default = all non-amateur")
    ap.add_argument("--lams", default="0,0.25,0.5,0.75,1,1.5")
    ap.add_argument("--per-task-for", default=None, help="'<expert>:<lam>' to print a per-task table")
    args = ap.parse_args()

    data = {it["id"]: it for it in json.load(open(args.dataset))}
    rec = {}
    for line in open(args.letters, encoding="utf-8"):
        r = json.loads(line)
        rec[r.pop("id")] = r
    ids = [i for i in data if i in rec and args.amateur in rec[i]]
    n = len(ids)
    all_tags = sorted({t for i in ids for t in rec[i]})
    experts = ([e.strip() for e in args.experts.split(",")] if args.experts
               else [t for t in all_tags if t != args.amateur])
    lams = [float(x) for x in args.lams.split(",")]
    print(f"{n} items | amateur={args.amateur} | experts={experts}")

    def pick(i, ex, lam):
        e, b = rec[i].get(ex), rec[i].get(args.amateur)
        if not e or not b:
            return None
        return max(LET, key=lambda L: e["letters"].get(L, -1e9) - lam * b["letters"].get(L, 0.0))

    def score(fn, name):
        g = a = 0
        for i in ids:
            L = fn(i)
            if L is None:
                continue
            item = data[i]
            g += int(L == vp_eval.gt_letter(item))
            a += int(L in vp_eval.adv_letters(item))
        print(f"  {name:<52} gt={100*g/n:.2f}  ala={100*a/n:.2f}")

    score(lambda i: (rec[i][args.amateur]["top"] if args.amateur in rec[i] else None),
          f"{args.amateur} (argmax)")
    for ex in experts:
        for lam in lams:
            score(lambda i, ex=ex, lam=lam: pick(i, ex, lam), f"{ex} - {lam:g}*{args.amateur}")

    if args.per_task_for:
        ex, lam = args.per_task_for.rsplit(":", 1)
        lam = float(lam)
        per = defaultdict(lambda: [0, 0, 0])
        for i in ids:
            item = data[i]
            gt = vp_eval.gt_letter(item)
            t = item["task_name"]
            e = rec[i].get(ex)
            per[t][0] += int(bool(e) and e["top"] == gt)
            per[t][1] += int(pick(i, ex, lam) == gt)
            per[t][2] += 1
        print(f"\n  per-task ({ex}, lam={lam:g}): steer-only -> +CD")
        for t in sorted(per):
            s, c, m = per[t]
            print(f"    {t:<32} {100*s/m:5.1f} -> {100*c/m:5.1f}")


if __name__ == "__main__":
    main()
