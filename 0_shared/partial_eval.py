#!/usr/bin/env python3
"""Early/partial read of an in-progress sweep. eval.py divides by the FULL dataset
count, so to score only the clips done so far we truncate the dataset to the first
N items (the runner writes clips in dataset order) and truncate every predictions
file to N lines, then eval each and print gt_acc/ala/Δ vs baseline."""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_PY = os.path.join(HERE, "..", "data", "eval.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.dir, "predictions_*.jsonl")))
    if not preds:
        print("no predictions yet"); return
    N = min(sum(1 for _ in open(p)) for p in preds)
    if N == 0:
        print("no rows yet"); return
    full = json.load(open(args.dataset))
    tmp = tempfile.mkdtemp()
    ds = os.path.join(tmp, "ds.json")
    json.dump(full[:N], open(ds, "w"))

    rows = {}
    for p in preds:
        tag = os.path.basename(p)[len("predictions_"):-len(".jsonl")]
        tp = os.path.join(tmp, f"{tag}.jsonl")
        with open(p) as f, open(tp, "w") as o:
            for i, ln in enumerate(f):
                if i >= N:
                    break
                o.write(ln)
        rep = os.path.join(tmp, f"{tag}.json")
        subprocess.run([sys.executable, EVAL_PY, "--predictions", tp, "--dataset", ds,
                        "--report", rep], check=True, stdout=subprocess.DEVNULL)
        r = json.load(open(rep))["overall"]
        rows[tag] = (r["gt_acc"], r["ala"])

    base = rows.get("baseline")
    print(f"\n=== PARTIAL eval on first N={N} clips ===")
    print(f"{'config':<24}{'acc':>7}{'ala':>7}{'Δacc':>8}{'Δala':>8}")
    print("-" * 54)
    for tag, (acc, ala) in sorted(rows.items(), key=lambda x: -x[1][0]):
        if base:
            da = f"{acc-base[0]:+.2f}"; dl = f"{ala-base[1]:+.2f}"
        else:
            da = dl = "-"
        print(f"{tag:<24}{acc:>7.2f}{ala:>7.2f}{da:>8}{dl:>8}")


if __name__ == "__main__":
    main()
