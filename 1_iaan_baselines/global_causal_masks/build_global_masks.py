#!/usr/bin/env python3
"""Build B-tier GLOBAL neuron masks from labeled activations (no GPU generation).

Unlike the per-clip A-tier scores, these produce ONE score vector per site for the
whole model, computed from a labeled dev split, then applied as a fixed mask to
every clip at inference. Methods:

  discriminative  across-attribute-VALUE variance of per-group mean activation, per
                  task, summed. Targets neurons separating attribute VALUES
                  (male/female, happy/sad) rather than audio-vs-noise.
  probe           diagonal-whitened difference-of-means, real vs noise (ITI/mass-mean
                  style): direction = (mu_real-mu_noise)/var; score = |direction|.
  probe_attr      whitened multi-class discriminant: per task, sum over value-pairs of
                  |(mu_a-mu_b)/var|; multivariate attribute separability.
  omp             greedy redundancy-aware selection seeded by the content score.

Output .pt: {"scores": {method: {site: vec}}, "site_order": [...]}. Consumed by
omni_global_mask_steer.py (top-K by score, z<-alpha*z).
"""
from __future__ import annotations
import argparse, itertools, json
from collections import defaultdict
import torch

EPS = 1e-6


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    sites = d["site_order"]
    recs = d["records"]
    def flat(rec, key):
        return torch.cat([rec[key][s].float() for s in sites])
    keys = ["mean_real", "var_real", "mean_noise1", "mean_noisebank", "mean_blank"]
    stacks = {k: torch.stack([flat(r, k) for r in recs]) for k in keys}
    meta = [{"task": r["task"], "gt": r["gt"]} for r in recs]
    Dsite = recs[0]["mean_real"][sites[0]].numel()
    return stacks, meta, sites, Dsite


def to_site_dict(vec, sites, Dsite):
    return {s: vec[i * Dsite:(i + 1) * Dsite].clone() for i, s in enumerate(sites)}


def discriminative(stacks, meta):
    mr = stacks["mean_real"]
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["task"], m["gt"])].append(i)
    task_gms = defaultdict(list)
    for (task, gt), idxs in groups.items():
        task_gms[task].append(mr[idxs].mean(0))
    score = torch.zeros(mr.shape[1])
    for task, gms in task_gms.items():
        if len(gms) >= 2:
            score = score + torch.stack(gms).var(0, unbiased=False)
    return score


def probe(stacks):
    pos, neg = stacks["mean_real"], stacks["mean_noise1"]
    X = torch.cat([pos, neg], 0)
    var = X.var(0, unbiased=False) + 1e-3
    return ((pos.mean(0) - neg.mean(0)) / var).abs()


def probe_attr(stacks, meta):
    mr = stacks["mean_real"]
    var = mr.var(0, unbiased=False) + 1e-3
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["task"], m["gt"])].append(i)
    by_task = defaultdict(dict)
    for (task, gt), idxs in groups.items():
        by_task[task][gt] = mr[idxs].mean(0)
    score = torch.zeros(mr.shape[1])
    for task, gmeans in by_task.items():
        vals = list(gmeans.values())
        for a, b in itertools.combinations(range(len(vals)), 2):
            score = score + ((vals[a] - vals[b]) / var).abs()
    return score


def omp(base_vec, stacks, K, corr_penalty=0.5, pool_size=4000):
    mr = stacks["mean_real"]
    Xc = (mr - mr.mean(0)) / (mr.std(0) + EPS)
    score = base_vec.clone()
    pool = torch.topk(score, min(pool_size, score.numel())).indices
    poolmask = torch.zeros_like(score, dtype=torch.bool); poolmask[pool] = True
    penalty = torch.zeros_like(score)
    chosen, avail = [], torch.ones_like(score, dtype=torch.bool)
    for _ in range(K):
        s = score - corr_penalty * penalty
        s[~(avail & poolmask)] = -1e9
        j = int(torch.argmax(s)); chosen.append(j); avail[j] = False
        cj = Xc[:, j]
        corr = (Xc[:, pool] * cj.unsqueeze(1)).mean(0).abs()
        penalty[pool] = torch.maximum(penalty[pool], corr)
    # return a score vec that ranks the chosen set on top (so top-K reproduces it)
    out = torch.full_like(score, -1e9)
    for rank, j in enumerate(chosen):
        out[j] = K - rank
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--omp-K", type=int, default=200, help="build OMP to cover up to this K")
    args = ap.parse_args()

    stacks, meta, sites, Dsite = load(args.acts)
    print(f"[masks] {stacks['mean_real'].shape[0]} clips, {len(sites)} sites, D={Dsite}")

    content = stacks["mean_real"].mean(0) - stacks["mean_noise1"].mean(0)
    scores_vec = {
        "content_global": content,
        "discriminative": discriminative(stacks, meta),
        "probe": probe(stacks),
        "probe_attr": probe_attr(stacks, meta),
        "omp": omp(content, stacks, args.omp_K),
    }
    scores = {m: to_site_dict(v, sites, Dsite) for m, v in scores_vec.items()}
    torch.save({"scores": scores, "site_order": sites}, args.out)

    # quick report: overlap of top-200 vs content_global
    def top(v, K=200):
        return set(torch.topk(v, K).indices.tolist())
    base = top(content)
    print("[masks] top-200 Jaccard vs content_global:")
    for m, v in scores_vec.items():
        j = len(top(v) & base) / len(top(v) | base)
        print(f"   {m:<16} {j:.3f}")
    print(f"[masks] saved {args.out}")


if __name__ == "__main__":
    main()
