#!/usr/bin/env python3
"""Per-sample overlap / oracle analysis across steering configs.

For a directory of predictions_<tag>.jsonl files (same dataset), compute:
  * per-config accuracy (sanity, should match reports)
  * ORACLE UNION: fraction of items answered correctly by AT LEAST ONE config
    -> headroom for per-clip adaptive config selection
  * pairwise complementarity: for the top two configs, items A-right/B-wrong etc.
  * per-task deltas of each config vs a chosen control

Usage:
  python methods_exp/overlap_analysis.py --dir methods_exp/out/test700_variants \
      --dataset data/voxparadox_test700.json --control content_K50_a16
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp_eval", os.path.join(HERE, "..", "data", "eval.py"))
vp_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp_eval)


def load_preds(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = r.get("response", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--control", default=None)
    args = ap.parse_args()

    data = {it["id"]: it for it in json.load(open(args.dataset))}
    tags = sorted(f[len("predictions_"):-len(".jsonl")]
                  for f in os.listdir(args.dir)
                  if f.startswith("predictions_") and f.endswith(".jsonl"))

    correct = {}     # tag -> {id: bool}
    for t in tags:
        preds = load_preds(os.path.join(args.dir, f"predictions_{t}.jsonl"))
        cmap = {}
        for sid, item in data.items():
            resp = preds.get(sid, "")
            pred_L = vp_eval.parse_response(resp, item)
            cmap[sid] = pred_L is not None and pred_L == vp_eval.gt_letter(item)
        correct[t] = cmap

    ids = list(data.keys())
    n = len(ids)
    print(f"{n} items | configs: {tags}\n")
    accs = {t: 100.0 * sum(correct[t][i] for i in ids) / n for t in tags}
    for t in sorted(tags, key=lambda x: -accs[x]):
        print(f"  {t:<34} acc={accs[t]:.2f}")

    steered = [t for t in tags if t != "baseline"]
    for group, label in [(tags, "ALL (incl baseline)"), (steered, "steered only")]:
        oracle = 100.0 * sum(any(correct[t][i] for t in group) for i in ids) / n
        print(f"\n  ORACLE UNION over {label}: {oracle:.2f}")

    # per-task oracle vs best single
    best = max(steered, key=lambda t: accs[t])
    by_task_o, by_task_b = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for i in ids:
        task = data[i].get("task_name") or i.split("__")[0]
        by_task_o[task][1] += 1; by_task_b[task][1] += 1
        by_task_o[task][0] += any(correct[t][i] for t in steered)
        by_task_b[task][0] += correct[best][i]
    print(f"\n  per-task: best single ({best}) vs steered-oracle")
    for task in sorted(by_task_o):
        c_o, m = by_task_o[task]; c_b, _ = by_task_b[task]
        print(f"    {task:<32} best={100.0*c_b/m:5.1f}  oracle={100.0*c_o/m:5.1f}")

    if args.control and args.control in correct:
        ctrl = args.control
        print(f"\n  pairwise vs control={ctrl}:")
        for t in steered:
            if t == ctrl:
                continue
            a_only = sum(correct[t][i] and not correct[ctrl][i] for i in ids)
            b_only = sum(correct[ctrl][i] and not correct[t][i] for i in ids)
            both = sum(correct[t][i] and correct[ctrl][i] for i in ids)
            print(f"    {t:<34} both={both:4d}  only-{t.split('_')[0]}={a_only:4d}  only-ctrl={b_only:4d}")


if __name__ == "__main__":
    main()
