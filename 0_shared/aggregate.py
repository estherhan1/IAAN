#!/usr/bin/env python3
"""Aggregate all method summaries into one comparison table vs the baseline.

Reads any of methods_summary.json / gmask_summary.json / *_summary.json under the
given dirs, aligns every config against its run's 'baseline' tag, and prints a
sorted table of gt_acc / ala / Δacc / Δala. The baseline is the reproduced
deployed method (per-clip real-vs-noise top-K, i.e. the current paper method), so
Δacc>0 means the new neuron-finder beats the current one.
"""
from __future__ import annotations
import argparse, glob, json, os


def load_summaries(dirs):
    runs = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*summary.json")):
            try:
                runs[f] = json.load(open(f))
            except Exception:
                pass
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sort", default="dacc", choices=["dacc", "acc", "dala"])
    args = ap.parse_args()

    rows = []
    for f, summ in load_summaries(args.dirs).items():
        base = summ.get("baseline")
        tag_run = os.path.basename(os.path.dirname(f))
        for tag, r in summ.items():
            if tag == "baseline":
                continue
            dacc = r["gt_acc"] - base["gt_acc"] if base else None
            dala = r["ala"] - base["ala"] if base else None
            rows.append({"run": tag_run, "config": tag, "acc": r["gt_acc"], "ala": r["ala"],
                         "dacc": dacc, "dala": dala,
                         "base_acc": base["gt_acc"] if base else None,
                         "base_ala": base["ala"] if base else None})
    if not rows:
        print("no summaries found"); return
    key = {"dacc": lambda x: -(x["dacc"] or -1e9), "acc": lambda x: -x["acc"],
           "dala": lambda x: (x["dala"] if x["dala"] is not None else 1e9)}[args.sort]
    rows.sort(key=key)

    print(f"{'config':<26}{'acc':>7}{'ala':>7}{'Δacc':>8}{'Δala':>8}{'(base acc/ala)':>18}")
    print("-" * 74)
    for r in rows:
        b = f"{r['base_acc']:.1f}/{r['base_ala']:.1f}" if r['base_acc'] is not None else "-"
        da = f"{r['dacc']:+.2f}" if r['dacc'] is not None else "-"
        dl = f"{r['dala']:+.2f}" if r['dala'] is not None else "-"
        print(f"{r['config']:<26}{r['acc']:>7.2f}{r['ala']:>7.2f}{da:>8}{dl:>8}{b:>18}")


if __name__ == "__main__":
    main()
