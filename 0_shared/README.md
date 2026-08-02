# 0_shared — activation caching, selection diagnostics, aggregation

Utilities used across the baseline and Steer-CD experiments. All Qwen2.5-Omni, all in
the `omni` conda env.

## Scripts

| Script | Purpose |
|---|---|
| `collect_acts_qwen.py` | Run the audio_tower on each VoxParadox clip for real / blank / R noise seeds and cache, per encoder FFN neuron (fc2 input), the temporal mean and temporal variance as fp16 vectors (`mean_real`, `var_real`, `mean_noise1`, `var_noise1`, `mean_noisebank`, `mean_blank`). This is everything the per-clip selectors need — including `var_noise1` for the snr / effect-size scores — cached once. Output: `results/acts_dev300.pt`. |
| `characterize.py` | Load `acts_*.pt` and, for each candidate selector, build the top-K neuron set it would pick and measure how far it differs from the deployed content selector (Jaccard overlap, layer distribution, K-th-neuron concentration). Low overlap is a necessary condition for a gain, so this triages which selectors are worth an expensive end-to-end run. Also builds the B-tier global masks. Output: `results/characterization.json`. |
| `aggregate.py` | Read any `*_summary.json` under given dirs, align every config against its run's `baseline`, and print a sorted `gt_acc / ALA / Δacc / Δala` table. |
| `partial_eval.py` | Score an in-progress run by truncating every `predictions_*.jsonl` to the shortest length and evaluating that prefix. |
| `overlap_analysis.py` | Oracle-union accuracy (headroom for per-clip config selection) and pairwise complementarity across a directory of prediction files. |

## Results

- `results/acts_dev300.pt` — symlink to the 578 MB activation cache (dev300, 300 clips × 32 layers).
- `results/acts_smoke.pt` — symlink to the small smoke cache.
- `results/characterization.json` — selector overlap / layer / concentration stats. Key finding: `snr`-family selectors pick almost disjoint neurons (Jaccard 0.002) but do not help; `tvar` / `meanplustvar` overlap the content selector ~0.35-0.51 and concentrate in the final layers.
- `results/collect.log` — activation-collection log.

## Re-run

```bash
$PY collect_acts_qwen.py --data data/voxparadox_dev300.json --audio_base data \
    --out methods_exp/out/acts_dev300.pt --noise-seeds 4
# characterize.py needs --acts (required) and --out to write the json:
$PY characterize.py --acts methods_exp/out/acts_dev300.pt \
    --out methods_exp/out/characterization.json
```
