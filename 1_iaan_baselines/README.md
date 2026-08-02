# 1_iaan_baselines — the IAAN steering baselines and their saturation controls

Everything here is Qwen2.5-Omni on VoxParadox, official generate+parse protocol
(`data/eval.py`), scored as `gt_acc / ALA`. `baseline` in every run is the reproduced
deployed method (per-clip real-vs-noise top-K, multiplicative scale). The purpose of
this whole tier is to answer: **can a better neuron finder or a better intervention
beat the deployed ~34?** Answer: no. The four sub-folders are the four ways we tried.

The intervention machinery is shared: a forward-pre-hook on each encoder-layer `fc2`
applies an affine map `z ← z·gain + add` to the FFN-neuron input; every method below
is a choice of which neurons to touch (selection) and what `(gain, add)` to use
(intervention).

---

## selectors/ — swap the per-clip neuron score

`omni_perq_methods.py`. One encode per clip is shared across all methods × K × α.
Selectors: `content` (deployed, signed mean diff real−noise), `presence` (real−blank),
`refbank` (lower-variance mean diff), `snr` (effect size), `tvar` (temporal-variance
contrast), `meanplustvar` (z(mean) + z(tvar)), `content_ln` (per-layer normalized),
and hybrids (`content_x_tvar`, `meanplus2tvar`, `tvar_real`).

Dev300 (`results/atier`):

| config | gt_acc | ALA |
|---|---|---|
| baseline | 8.67 | 76.0 |
| content `K50 a16` | 31.0 | 45.3 |
| tvar `K50 a16` | 34.67 | 45.0 |
| **meanplustvar `K50 a16`** | **34.67** | 44.0 |
| snr `K50 a16` | 9.33 | 75.0 (fails) |
| content_ln `K50 a16` | 19.33 | 24.0 |

Held-out test700 (`results/test700`, steer-only): baseline 6.86, content `K50 a16`
29.43, tvar `K50 a24` 32.71, **meanplustvar `K50 a16` 33.57**. `results/refine` is a
finer K∈{30,50} × α∈{16,24} sweep; `results/test700_variants` checks the hybrid
selectors on test700 (all land 28-34). Conclusion: `meanplustvar` is the best selector
and it tops out at ~34.

Re-run: `$PY omni_perq_methods.py --data_path data/voxparadox_dev300.json --audio_base data --output_dir methods_exp/out/atier --methods content,snr,tvar,meanplustvar,content_ln --ks 50,200 --alphas 8,16 --noise-seeds 4`

---

## interventions/ — swap the intervention on the selected neurons

`omni_perq_interventions.py`. Selector fixed to `meanplustvar`; sweep the op:
`scale` (z←αz, deployed), `contrast` (amplify only the audio-specific deviation off the
noise mean), `blankcontrast` (off silence), `bidir` (amplify top-K, attenuate bottom-K),
`suppress` (attenuate bottom-K only), `graded` (soft rank-weighted gain).

Dev300 (`results/intv_dev300`):

| config | gt_acc | ALA |
|---|---|---|
| baseline | 8.67 | 76.0 |
| **meanplustvar scale `K50 a16`** | **34.67** | 44.0 |
| meanplustvar bidir `K50 a16 b0.25` | 34.67 | 45.3 |
| meanplustvar blankcontrast `K50 a16` | 34.0 | 43.0 |
| meanplustvar contrast `K50 a16` | 32.67 | 46.3 |
| meanplustvar graded `K400 a16` | 20.0 | 61.3 |
| meanplustvar suppress `K50 a0.25` | 8.67 | 77.0 (no-op) |

Conclusion: the affine intervention family is saturated. Nothing beats plain `scale`.

Re-run: `$PY omni_perq_interventions.py --data_path data/voxparadox_dev300.json --audio_base data --output_dir methods_exp/out/intv_dev300 --configs baseline,meanplustvar:scale:50:16,meanplustvar:contrast:50:16,meanplustvar:bidir:50:16:0.25,meanplustvar:graded:400:16,meanplustvar:suppress:50:0.25`

---

## global_causal_masks/ — fixed masks instead of per-clip selection

`build_global_masks.py` (label-free-ish global scores: `discriminative`, `probe`,
`probe_attr`, `omp`), `build_causal_mask.py` (the label-aware **causal oracle**: input×
gradient of the gt-vs-adversarial margin, an upper bound on what a better selector could
do), and `omni_global_mask_steer.py` (apply one fixed mask to every clip).

Dev300:

| config | gt_acc | ALA |
|---|---|---|
| baseline | 8.67 | 76.0 |
| discriminative `K200 a12` | 26.0 | 53.0 |
| omp `K50 a12` | 30.33 | 49.0 |
| probe / probe_attr | ~8.7 | ~76 (fail) |
| causal oracle `K200 a16` | 29.67 | 45.0 |
| **causal_abs oracle `K200 a16`** | **33.33** | 44.0 |

Conclusion: even the label-aware causal oracle only reaches ~33, i.e. the same ceiling
as the label-free per-clip selectors. The selector is not the bottleneck.

Re-run: see `run_chains/run_btier_chain.sh` (builds masks then evaluates).

---

## confidence_ensemble/ — combine configs without labels

`omni_perq_confpick.py` (per config: one prefill for a letter-logit confidence margin
plus a full generate), `eval_confpick.py` (single-config sanity, oracle union,
argmax-margin over all/steered configs, default+switch threshold fallback, and
margin-vs-correctness calibration — all over the generated answers), `ensemble_vote.py`
(majority vote).

Dev300 (`results/confpick_dev300`): the single best config is `meanplustvar scale
K50 a16` at 34.67. The oracle union over configs is ~11 pp higher, but no label-free
combination captures it: the baseline is anti-calibrated (its margin is *larger* when it
is wrong), so confidence-based per-clip selection fails and majority vote adds ≈ +1.
This is what motivated changing the readout itself, which is Steer-CD (folder `2_`).
`results/confpick_dev300/margins.jsonl` holds every config's per-clip letter logits and
margins.
