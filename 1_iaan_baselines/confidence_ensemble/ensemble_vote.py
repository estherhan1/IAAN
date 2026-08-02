#!/usr/bin/env python3
"""Offline ensemble (majority vote over steering configs) on existing predictions.

Label-free per-clip combination: each config votes with its parsed letter; the
majority letter wins, ties broken by the config priority order given in --prio.
Evaluated with the same parsing/GT semantics as data/eval.py.

Usage:
  python methods_exp/ensemble_vote.py --dir methods_exp/out/test700_variants \
      --dataset data/voxparadox_test700.json \
      --members meanplustvar_K50_a16,content_K50_a16,tvar_real_K50_a16 \
      --prio meanplustvar_K50_a16
"""
from __future__ import annotations
import argparse, importlib.util, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp_eval", os.path.join(HERE, "..", "data", "eval.py"))
vp_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp_eval)


def load_letters(path, data):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        item = data.get(r["id"])
        if item is not None:
            out[r["id"]] = vp_eval.parse_response(r.get("response", ""), item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--members", required=True, help="comma-sep tags")
    ap.add_argument("--prio", required=True, help="comma-sep tie-break priority (first wins)")
    args = ap.parse_args()

    data = {it["id"]: it for it in json.load(open(args.dataset))}
    members = [m.strip() for m in args.members.split(",")]
    prio = [m.strip() for m in args.prio.split(",")]
    letters = {m: load_letters(os.path.join(args.dir, f"predictions_{m}.jsonl"), data) for m in members}

    n = len(data)
    acc = {m: 0 for m in members}
    vote_gt = vote_ala = 0
    by_task = defaultdict(lambda: [0, 0])
    for sid, item in data.items():
        gt = vp_eval.gt_letter(item)
        adv = vp_eval.adv_letters(item)
        votes = [letters[m].get(sid) for m in members]
        for m, v in zip(members, votes):
            acc[m] += int(v is not None and v == gt)
        cnt = Counter(v for v in votes if v is not None)
        if cnt:
            top = max(cnt.values())
            cands = {L for L, c in cnt.items() if c == top}
            pick = None
            for m in prio:
                v = letters[m].get(sid)
                if v in cands:
                    pick = v
                    break
            if pick is None:
                pick = sorted(cands)[0]
        else:
            pick = None
        ok = int(pick is not None and pick == gt)
        vote_gt += ok
        vote_ala += int(pick is not None and pick in adv)
        task = item["task_name"]
        by_task[task][0] += ok
        by_task[task][1] += 1

    print(f"members: {members}")
    for m in members:
        print(f"  {m:<34} acc={100.0*acc[m]/n:.2f}")
    print(f"\n  MAJORITY VOTE  gt_acc={100.0*vote_gt/n:.2f}  ala={100.0*vote_ala/n:.2f}")
    print("\n  vote per-task:")
    for task in sorted(by_task):
        c, m_ = by_task[task]
        print(f"    {task:<32} {100.0*c/m_:5.1f}")


if __name__ == "__main__":
    main()
