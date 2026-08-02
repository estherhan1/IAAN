#!/usr/bin/env python3
"""Offline characterization of neuron-selection methods (no GPU generation).

Loads acts_*.pt (from collect_acts_qwen.py) and, for each scoring method, forms
the selected top-K neuron set the way the deployed per-clip method would, then
reports how different that selection is from the current baseline (real-vs-noise
signed mean difference, "content_pos"):

  - Jaccard overlap of top-K vs baseline  (per clip, averaged)
  - layer distribution of selected neurons (fraction in the final layer)
  - concentration: z-score of the K-th selected neuron in the score distribution

A method that picks nearly the same neurons as the baseline cannot move the
metric; low overlap is a necessary (not sufficient) condition for a gain. This
tells us which methods are worth an expensive end-to-end run.

It also builds the B-tier GLOBAL masks (discriminative, whitened-probe, OMP) from
the labeled dev split and reports their overlap with the aggregated baseline mask.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import torch

EPS = 1e-6


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    sites = d["site_order"]
    L = len(sites)
    recs = d["records"]
    n = len(recs)
    Dsite = recs[0]["mean_real"][sites[0]].numel()
    N = L * Dsite
    layer_of = torch.arange(L).repeat_interleave(Dsite)  # (N,) layer index per flat neuron

    def flat(rec, key):
        return torch.cat([rec[key][s].float() for s in sites])  # (N,)

    keys = ["mean_real", "var_real", "mean_noise1", "var_noise1", "mean_noisebank", "mean_blank"]
    stacks = {k: torch.stack([flat(r, k) for r in recs]) for k in keys}  # each (n, N)
    meta = [{"id": r["id"], "task": r["task"], "gt": r["gt"], "adv": r["adv"]} for r in recs]
    return stacks, meta, layer_of, L, Dsite, N, sites


def clip_scores(stacks, method):
    """Return (n, N) per-clip signed score; higher = more selected (top-K by value)."""
    mr, vr = stacks["mean_real"], stacks["var_real"]
    mn1, vn1 = stacks["mean_noise1"], stacks["var_noise1"]
    mnb, mbl = stacks["mean_noisebank"], stacks["mean_blank"]
    if method == "content":            # deployed Qwen baseline (real - noise, signed pos)
        return mr - mn1
    if method == "presence":           # real - blank
        return mr - mbl
    if method == "snr":                # effect size / Welch t (per-clip temporal var)
        return (mr - mn1) / (vr + vn1 + EPS).sqrt()
    if method == "snr_refbank":
        return (mr - mnb) / (vr + vn1 + EPS).sqrt()
    if method == "tvar":               # temporal-dynamics contrast
        return vr - vn1
    if method == "tvar_snr":
        return (vr - vn1) / (vn1 + EPS)
    if method == "refbank":            # averaged noise ref -> lower-variance mean diff
        return mr - mnb
    if method == "meanplustvar":       # z(mean diff) + z(tvar diff), z within clip
        a = mr - mn1
        b = vr - vn1
        za = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + EPS)
        zb = (b - b.mean(1, keepdim=True)) / (b.std(1, keepdim=True) + EPS)
        return za + zb
    raise ValueError(method)


def per_layer_norm(score, layer_of, L):
    """Z-score each layer's neurons within that layer (across the (n,N) or (N,) tensor)."""
    out = score.clone()
    for l in range(L):
        m = (layer_of == l)
        seg = out[..., m]
        out[..., m] = (seg - seg.mean(-1, keepdim=True)) / (seg.std(-1, keepdim=True) + EPS)
    return out


def topk_sets(score, K):
    """score (n,N) -> list of frozenset(topk idx) per row."""
    idx = torch.topk(score, K, dim=1).indices
    return [frozenset(row.tolist()) for row in idx], idx


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def _stats(sc, base_sets, layer_of, L, K):
    sets, idx = topk_sets(sc, K)
    jac = sum(jaccard(s, b) for s, b in zip(sets, base_sets)) / len(sets)
    sel_layers = layer_of[idx]                      # (n,K)
    frac_last = float((sel_layers == (L - 1)).float().mean())
    n_layers_used = float((torch.stack([torch.bincount(sel_layers[i], minlength=L) > 0
                                        for i in range(sel_layers.shape[0])]).float().sum(1)).mean())
    kth = torch.topk(sc, K, dim=1).values[:, -1]
    conc = ((kth - sc.mean(1)) / (sc.std(1) + EPS)).mean().item()
    return {"jaccard_vs_content": round(jac, 3),
            "frac_last_layer": round(frac_last, 3),
            "mean_layers_used": round(n_layers_used, 2),
            "kth_zscore": round(conc, 2)}


def characterize_perclip(stacks, layer_of, L, methods, K):
    """All methods (and a per-layer-normalized 'content_ln') are compared against the
    RAW content baseline top-K set."""
    base_score = clip_scores(stacks, "content")
    base_sets, _ = topk_sets(base_score, K)
    rows = {}
    for m in methods:
        rows[m] = _stats(clip_scores(stacks, m), base_sets, layer_of, L, K)
    rows["content_layernorm"] = _stats(per_layer_norm(base_score, layer_of, L),
                                       base_sets, layer_of, L, K)
    return rows


# ---------- B-tier global masks (need labels) ----------
def global_baseline_mask(stacks, K):
    d = (stacks["mean_real"] - stacks["mean_noise1"]).mean(0)   # aggregate content
    return frozenset(torch.topk(d, K).indices.tolist())


def discriminative_mask(stacks, meta, K):
    """Across-attribute-VALUE variance of per-group mean real activation, within each
    task, summed over tasks. Picks neurons that separate the acoustic attribute VALUES
    (male/female, happy/sad, ...) rather than audio-vs-noise."""
    mr = stacks["mean_real"]                          # (n, N)
    by = defaultdict(list)
    for i, m in enumerate(meta):
        by[(m["task"], m["gt"])].append(i)
    task_groups = defaultdict(list)
    for (task, gt), idxs in by.items():
        task_groups[task].append(mr[idxs].mean(0))    # group-mean vector
    score = torch.zeros(mr.shape[1])
    for task, gmeans in task_groups.items():
        if len(gmeans) >= 2:
            score = score + torch.stack(gmeans).var(0, unbiased=False)
    return score, frozenset(torch.topk(score, K).indices.tolist())


def probe_mask(stacks, meta, K, mode="real_vs_noise"):
    """Whitened difference-of-means (ITI/mass-mean style). Direction = Sigma^-1 (mu_pos-mu_neg);
    neuron score = |direction| (importance under a linear probe). Diagonal whitening for stability."""
    if mode == "real_vs_noise":
        pos = stacks["mean_real"]; neg = stacks["mean_noise1"]
        X = torch.cat([pos, neg], 0)
        mu = pos.mean(0) - neg.mean(0)
    else:
        raise ValueError(mode)
    var = X.var(0, unbiased=False) + 1e-3
    direction = mu / var                              # diagonal-whitened mean difference
    score = direction.abs()
    return score, frozenset(torch.topk(score, K).indices.tolist())


def omp_mask(base_score_vec, stacks, K, corr_penalty=0.5):
    """Greedy, redundancy-aware selection. Start from a univariate score; iteratively add the
    highest-scoring neuron after down-weighting neurons correlated (across clips) with those
    already picked. Cheap surrogate for OMP/conditional selection."""
    mr = stacks["mean_real"]                          # (n, N) feature matrix
    Xc = (mr - mr.mean(0)) / (mr.std(0) + EPS)        # standardized features
    score = base_score_vec.clone()
    chosen = []
    avail = torch.ones_like(score, dtype=torch.bool)
    penalty = torch.zeros_like(score)
    # cap candidate pool for tractable correlation (top 4000 by base score)
    pool = torch.topk(score, min(4000, score.numel())).indices
    poolmask = torch.zeros_like(avail); poolmask[pool] = True
    for _ in range(K):
        s = score - corr_penalty * penalty
        s[~(avail & poolmask)] = -1e9
        j = int(torch.argmax(s))
        chosen.append(j)
        avail[j] = False
        # update penalty = max correlation to any chosen neuron
        cj = Xc[:, j]
        corr = (Xc[:, pool] * cj.unsqueeze(1)).mean(0).abs()   # (|pool|,)
        penalty[pool] = torch.maximum(penalty[pool], corr)
    return frozenset(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--Ks", default="50,200")
    args = ap.parse_args()

    stacks, meta, layer_of, L, Dsite, N, sites = load(args.acts)
    print(f"[char] {len(meta)} clips, L={L} sites, D={Dsite}, N={N}")
    Ks = [int(x) for x in args.Ks.split(",")]

    perclip_methods = ["content", "presence", "refbank", "snr", "snr_refbank",
                       "tvar", "tvar_snr", "meanplustvar"]
    report = {"n_clips": len(meta), "L": L, "N": N, "Ks": Ks, "perclip": {}, "global": {}}

    for K in Ks:
        print(f"\n===== PER-CLIP selection, K={K} (baseline=content real-vs-noise) =====")
        rows = characterize_perclip(stacks, layer_of, L, perclip_methods, K)
        report["perclip"][K] = rows
        hdr = f"{'method':<18}{'jac_vs_base':>12}{'frac_last':>11}{'layers_used':>13}{'kth_z':>8}"
        print(hdr); print("-" * len(hdr))
        for m, r in rows.items():
            print(f"{m:<18}{r['jaccard_vs_content']:>12}{r['frac_last_layer']:>11}"
                  f"{r['mean_layers_used']:>13}{r['kth_zscore']:>8}")

    print("\n===== GLOBAL masks (B-tier), overlap vs aggregated content mask =====")
    for K in Ks:
        base = global_baseline_mask(stacks, K)
        disc_s, disc = discriminative_mask(stacks, meta, K)
        probe_s, probe = probe_mask(stacks, meta, K)
        base_vec = (stacks["mean_real"] - stacks["mean_noise1"]).mean(0)
        omp = omp_mask(base_vec, stacks, K)
        g = {
            "discriminative": round(jaccard(disc, base), 3),
            "probe_whitened": round(jaccard(probe, base), 3),
            "omp": round(jaccard(omp, base), 3),
        }
        # layer fraction in last layer for each
        def fl(s):
            return round(float((layer_of[torch.tensor(sorted(s))] == (L - 1)).float().mean()), 3)
        report["global"][K] = {"jaccard_vs_base": g,
                               "frac_last": {"base": fl(base), "discriminative": fl(disc),
                                             "probe": fl(probe), "omp": fl(omp)}}
        print(f"K={K}: jaccard vs base -> {g}")
        print(f"      frac_last_layer  -> base={fl(base)} disc={fl(disc)} probe={fl(probe)} omp={fl(omp)}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\n[char] saved {args.out}")


if __name__ == "__main__":
    main()
