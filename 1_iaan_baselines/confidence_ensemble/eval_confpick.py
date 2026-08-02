#!/usr/bin/env python3
"""Offline evaluation of per-clip config-selection rules from a confpick run.

Inputs: an output dir of omni_perq_confpick.py (predictions_<tag>.jsonl for every
config + margins.jsonl). The selected config's GENERATED answer is what gets
scored, so every rule below is protocol-faithful (official generate+parse).

Rules evaluated:
  * each single config (sanity)
  * oracle union (headroom)
  * argmax-margin over all configs / over steered only
  * default+switch: keep DEFAULT unless some config's margin beats it by tau
  * margin-vs-correctness calibration per config
"""
from __future__ import annotations
import argparse, importlib.util, json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp_eval", os.path.join(HERE, "..", "data", "eval.py"))
vp_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp_eval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--default", required=True, help="default config tag for default+switch")
    ap.add_argument("--taus", default="0,1,2,4,8")
    args = ap.parse_args()

    data = {it["id"]: it for it in json.load(open(args.dataset))}
    tags = sorted(f[len("predictions_"):-len(".jsonl")]
                  for f in os.listdir(args.dir)
                  if f.startswith("predictions_") and f.endswith(".jsonl"))
    letters, correct = {}, {}
    for t in tags:
        lm, cm = {}, {}
        for line in open(os.path.join(args.dir, f"predictions_{t}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            item = data.get(r["id"])
            if item is None:
                continue
            L = vp_eval.parse_response(r.get("response", ""), item)
            lm[r["id"]] = L
            cm[r["id"]] = L is not None and L == vp_eval.gt_letter(item)
        letters[t], correct[t] = lm, cm

    margins = {}
    for line in open(os.path.join(args.dir, "margins.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        margins[r.pop("id")] = r

    ids = [i for i in data if i in margins]
    n = len(ids)
    steered = [t for t in tags if t != "baseline"]
    print(f"{n} items | configs: {tags}\n")
    for t in sorted(tags, key=lambda x: -sum(correct[x].values())):
        print(f"  {t:<38} acc={100.0*sum(correct[t][i] for i in ids)/n:.2f}")
    print(f"\n  oracle union (all):     {100.0*sum(any(correct[t][i] for t in tags) for i in ids)/n:.2f}")
    print(f"  oracle union (steered): {100.0*sum(any(correct[t][i] for t in steered) for i in ids)/n:.2f}")

    def margin_of(i, t):
        e = margins.get(i, {}).get(t)
        return e["margin"] if e else None

    def run_rule(pool, pick_fn, name):
        gt_ok = ala_ok = 0
        for i in ids:
            t = pick_fn(i, pool)
            L = letters[t].get(i)
            item = data[i]
            gt_ok += int(L is not None and L == vp_eval.gt_letter(item))
            ala_ok += int(L is not None and L in vp_eval.adv_letters(item))
        print(f"  {name:<44} gt_acc={100.0*gt_ok/n:.2f}  ala={100.0*ala_ok/n:.2f}")

    def argmax_margin(i, pool):
        best, bm = pool[0], -1e9
        for t in pool:
            m = margin_of(i, t)
            if m is not None and m > bm:
                best, bm = t, m
        return best

    print("\n  selection rules:")
    run_rule(tags, argmax_margin, "argmax-margin (all)")
    run_rule(steered, argmax_margin, "argmax-margin (steered)")
    for tau in [float(x) for x in args.taus.split(",")]:
        def default_switch(i, pool, tau=tau):
            dm = margin_of(i, args.default)
            dm = -1e9 if dm is None else dm
            best, bm = args.default, dm
            for t in pool:
                m = margin_of(i, t)
                if m is not None and m > bm + tau:
                    best, bm = t, m - tau   # require each challenger to clear tau over current
            return best
        run_rule(steered, default_switch, f"default={args.default} switch-if-margin>+{tau:g}")

    print("\n  margin calibration (correct vs wrong mean margin):")
    for t in tags:
        mc = [margin_of(i, t) for i in ids if correct[t][i] and margin_of(i, t) is not None]
        mw = [margin_of(i, t) for i in ids if not correct[t][i] and margin_of(i, t) is not None]
        if mc and mw:
            print(f"    {t:<38} correct={sum(mc)/len(mc):6.2f}  wrong={sum(mw)/len(mw):6.2f}")


if __name__ == "__main__":
    main()
