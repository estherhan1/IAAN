# IAAN — Identifying and Amplifying Acoustic Neurons

**Enhancing How Large Audio-Language Models Hear: Identifying and Amplifying Acoustic Neurons in the Audio Encoder**  
Yu-Han Huang, Chih-Kai Yang, Ke-Han Lu, An-Yu Cheng, Hung-yi Lee · IEEE SLT 2026

[Paper (OpenReview)](https://openreview.net/forum?id=aZaFtJs6dr) · [arXiv](https://arxiv.org/abs/2607.11801) · [PDF](https://arxiv.org/pdf/2607.11801)

## Code status

**The SLT 2026 reproduction code is not yet complete.** The `paper_reproduction/` directory contains the model-agnostic IAAN algorithm, a configurable command-line launcher, and a result-table generator. The original model-specific greedy-generation/evaluation runners and benchmark data are **not included**, so running `bash reproduce_main_table.sh` alone **will not reproduce Table I** yet.

See [paper reproduction instructions](paper_reproduction/README.md) for supported CLI flags, paper-selected parameters and missing dependencies.

The directories `0_shared/`, `1_iaan_baselines/`, `2_steer_cd/`, `3_mmau_control/`, and `run_chains/` are an archive of additional research and follow-up Steer-CD experiments. The archived README below refers to that research, **not to a complete IAAN paper release**.

---

## Archived Steer-CD experimental notes

# Steer-CD release — IAAN steering baselines + Steer-CD across three models

This folder is an organized, self-contained archive of every experiment behind the
Steer-CD result on VoxParadox: the earlier IAAN per-clip steering baselines, the
control experiments that showed those baselines were saturated, and the Steer-CD
breakthrough reproduced on three audio-language models (Qwen2.5-Omni, Audio-Flamingo-3,
Kimi-Audio).

Everything here is a **copy** of the working files in `../methods_exp` (code + text
results, 7.8 MB). The large activation caches (`*.pt`, ~580 MB) are **symlinks** to
the originals, not copies, so this archive stays small. The originals in
`../methods_exp` are untouched and remain the runnable versions.

The narrative and the deep per-task tables live in `STEER_CD_REPORT.md` (top level of
this folder). This README is the map.

---

## The one-paragraph story

The deployed method (IAAN / SPAN) amplifies per-clip audio-selective encoder FFN
neurons and reads the answer by generating text. On VoxParadox it lifts ground-truth
accuracy from a floored, language-prior-dominated baseline (~7-10) to ~34, but no
matter how the neurons are **selected** (mean-diff, effect size, temporal variance,
combined, global masks, or even a label-aware causal oracle) or how they are
**intervened on** (scale, contrast, bidirectional, graded), the accuracy saturates
around 34. The gain is capped at the readout, not the selector. **Steer-CD** breaks
that ceiling: it keeps the steering but changes the readout to a contrast between the
steered and unsteered model, `answer = argmax_L [ logit_steered(L) - λ·logit_unsteered(L) ]`.
This subtracts the residual language prior and lifts held-out test accuracy to ~49 on
Qwen (+15 over steer-only, +20 over the deployed method), and it transfers to AF3
(48.0) and Kimi (32.7). A no-conflict control (MMAU) confirms the mechanism: where the
audio does not conflict with the prior, the same contrast strictly hurts.

---

## Folder map

| Folder | What it holds |
|---|---|
| `0_shared/` | Activation caching + selection-diagnostics + result-aggregation utilities shared by everything. |
| `1_iaan_baselines/` | The earlier IAAN steering baselines and their saturation controls: per-clip **selectors**, affine **interventions**, **global/causal masks**, and **confidence/ensemble** combination. All Qwen2.5-Omni, official generate+parse protocol. |
| `2_steer_cd/` | The Steer-CD breakthrough on **three models**: `qwen/` (letter-logit readout + per-token generation), `af3/`, `kimi/`. |
| `3_mmau_control/` | No-conflict mechanism check on MMAU: steering and CD both hurt where the baseline is already audio-grounded. |
| `run_chains/` | The serial-GPU orchestration shell scripts that produced the dev/test runs, plus their driver logs. |
| `STEER_CD_REPORT.md` | Full write-up with per-task breakdowns and ablations. |

Each folder has its own `README.md` with the exact scripts, result files, headline
numbers, and re-run commands.

---

## Headline results

All numbers are `gt_acc` (ground-truth accuracy, %) with `ALA` (adversarial /
language-prior answer rate, %) as the pathology gauge. Higher gt is better; lower ALA
means less language-prior leakage.

### 1. IAAN steering is saturated (~34), regardless of selector or intervention

Qwen2.5-Omni, VoxParadox dev300, generate+parse protocol:

| Method (best config) | gt_acc | ALA | note |
|---|---|---|---|
| baseline (deployed content readout) | 8.67 | 76.0 | language prior dominates |
| content selector `K50 a16` | 31.0 | 45.3 | the deployed IAAN selector |
| tvar selector `K50 a16` | 34.67 | 45.0 | temporal-dynamics selector |
| meanplustvar selector `K50 a16` | **34.67** | 44.0 | best selector |
| meanplustvar + contrast/bidir/graded intervention | ≤ 34.67 | 43-61 | affine family, none beats plain scale |
| global discriminative / omp mask | 26-30 | 33-53 | fixed masks, worse |
| **causal oracle mask** (label-aware) `K200 a16` | 33.33 | 44.0 | even the label-aware ceiling ≈ 34 |

Held-out test700 (steer-only, meanplustvar `K50 a16`): **33.57**. The selector question
is closed: nothing moves the ceiling.

### 2. Steer-CD breaks the ceiling (held-out test700, letter-logit readout)

λ chosen on dev300; each model steered on its own audio-encoder FFN:

| Model | baseline | steer-only (λ=0) | **Steer-CD** | Δ vs steer | ALA base → CD | operating point |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 9.86 | 34.43 | **49.43** | +15.0 | 77 → 13 | meanplustvar `K100 a16`, λ0.75 |
| Audio-Flamingo-3-7B | 15.00 | 32.00 | **48.00** | +16.0 | 67 → 10 | meanplustvar `K50 a8`, λ1.0 |
| Kimi-Audio-7B | 20.29 | 30.43 | **32.71** | +2.3 | 66 → 27 | content `K200 a12`, λ0.5 |

The gain tracks pathology severity: Qwen and AF3 have floored, prior-dominated
baselines and gain +15-16; Kimi is the least pathological (highest baseline, steering
already recovers most of its gap) so CD mostly trades ALA for spread, +2.3. The λ
plateau centre rises with pathology (Kimi 0.5 < Qwen 0.75 < AF3 1.0). Both components
are necessary: CD without steering fails, steering without CD saturates at 34.

### 3. Steer-CD survives the real generation protocol (per-token CD)

Qwen2.5-Omni, dual KV-cache branches, adaptive-plausibility mask `p≥0.1·max`:

| Split | config | gt_acc | ALA |
|---|---|---|---|
| dev300 | steer-only (λ0) | 34.67 | 44.0 |
| dev300 | tokencd λ0.75 `p0.1` | 52.67 | 17.0 |
| **test700** | **tokencd λ0.75 `p0.1`** | **48.71** | 15.57 |

Within 0.7 pp of the letter-logit readout, so the win is not a readout artifact.

### 4. Mechanism check — no-conflict control (MMAU, overall acc)

| config | overall acc |
|---|---|
| baseline | 68.0 |
| content `K50 a16` steer | 49.67 |
| meanplustvar `K50 a16` steer | 54.33 |

Where the audio does not conflict with the language prior, the baseline distribution
**is** the signal; steering and CD monotonically destroy it. This is the mirror image
of the VoxParadox result and confirms Steer-CD is a conflict-resolution mechanism.

---

## How to re-run

The scripts here are archived copies. The runnable originals are in `../methods_exp`;
run them from the repo root `contra_decode/voxparadox/VoxParadox` with the shim on
`PYTHONPATH` (Qwen scripts import the `contra_decode` steering package and the repo's
`omni_eval_voxparadox`):

```bash
cd .../contra_decode/voxparadox/VoxParadox
SHIM=$PWD/_steer_shim
SCR=.../contra_decode/scripts
export PYTHONPATH="$SHIM:$SCR/steering:$SCR/ablation:$SCR/pipeline:$PWD"
export HF_HOME=/livingrooms/khlu/hf_cache
PY=/home/khlu/miniconda3/envs/omni/bin/python     # Qwen + MMAU + all offline evals
```

- **Qwen / MMAU** scripts: `omni` conda env, command above.
- **AF3** collector (`2_steer_cd/af3/`): `af3` conda env, run from the repo root.
- **Kimi** collector (`2_steer_cd/kimi/`): `kimi` conda env, add `HF_HUB_OFFLINE=1`.
- **Offline Steer-CD evaluation** (no GPU): `eval_cd_letters.py` on any `cll_*.jsonl`.

Per-folder READMEs give the exact command line that produced each result directory.

---

## Provenance (result dir → producing script)

| Result dir / file | Produced by | Split |
|---|---|---|
| `1_iaan_baselines/selectors/results/atier` | `omni_perq_methods.py` | dev300 |
| `.../selectors/results/refine` | `omni_perq_methods.py` (finer K/α) | dev300 |
| `.../selectors/results/test700`, `test700_variants` | `omni_perq_methods.py` | test700 |
| `1_iaan_baselines/interventions/results/intv_dev300` | `omni_perq_interventions.py` | dev300 |
| `.../global_causal_masks/results/btier_global` | `omni_global_mask_steer.py` + `build_global_masks.py` | dev300 |
| `.../global_causal_masks/results/btier_causal` | `omni_global_mask_steer.py` + `build_causal_mask.py` | dev300 |
| `.../confidence_ensemble/results/confpick_dev300` | `omni_perq_confpick.py` | dev300 |
| `2_steer_cd/qwen/results/cll_test700.jsonl` | `collect_letter_logits.py` | test700 |
| `2_steer_cd/qwen/results/cll_mmsu_dev600.jsonl` | `collect_letter_logits.py` (dev-transfer) | MMSU_para |
| `2_steer_cd/qwen/results/tokencd_*` | `omni_perq_tokencd.py` | dev/test |
| `2_steer_cd/af3/results/cll_af3_*` | `af3_steer_cd_collect.py` | dev/test |
| `2_steer_cd/kimi/results/cll_kimi_*` | `kimi_steer_cd_collect.py` | dev/test |
| `3_mmau_control/results/mmau300` | `omni_mmau_methods.py` | MMAU-300 |
| `3_mmau_control/results/cll_mmau300.jsonl` | `collect_letter_logits_mmau.py` | MMAU-300 |
| `0_shared/results/acts_dev300.pt` | `collect_acts_qwen.py` | dev300 |
| `0_shared/results/characterization.json` | `characterize.py` | dev300 |
