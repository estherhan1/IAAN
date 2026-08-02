# Steering + Contrastive Readout (Steer-CD): method and full results

Date: 2026-07-10/11. Model: Qwen2.5-Omni-7B. Benchmark: VoxParadox (official
protocol / `data/eval.py` semantics). All experiments in `methods_exp/`.

## TL;DR

Answer each MCQ by contrasting the steered model against the unsteered model at
the answer-letter position:

```
answer = argmax_L [ logit_steered(L) − λ · logit_unsteered(L) ]
```

- expert: encoder FFN neuron steering, per-clip meanplustvar selection, top-K=50
  (or 100), gain α=16 — identical to the existing SPAN/IAAN intervention.
- amateur: the SAME model on the SAME real audio, no steering.
- λ = 0.75 (chosen on dev300; broad plateau λ ∈ [0.5, 1.0]).
- Two prefill forwards per item, no generation needed.

**Held-out test700: 49.43 gt_acc** vs 33.57 steering-only (generate) /
34.43 (prefill) vs 29.43 deployed content method vs 6.86–9.86 baseline.
ALA drops 43 → 13. Dev300: 57.67. Survives official generate+parse (48.71,
per-token CD). Transfers to two more architectures: AF3 48.00 (steer 32.00),
Kimi-Audio 32.71 (steer 30.43) — see "Cross-model validation".

## Why it works (each claim has an ablation)

1. The unsteered model's letter distribution is almost pure language prior
   (baseline: 7–10 gt, 77 ALA). Subtracting it removes the residual prior that
   steering leaves behind.
2. CD **without** steering fails: real − λ·blank and real − λ·noise give
   4.7–8.3 (dev) / 5.3–7.6 (test), i.e. baseline level. The unsteered model's
   audio contribution to the letter logits is too weak to extract by contrast —
   this weakness IS the pathology the paper describes.
3. Steering **without** CD plateaus at 33.6–34.7 (selector and intervention
   family both saturated — see "Negative results").
4. Together: steering amplifies the audio evidence, CD strips the prior.

## Main numbers

### test700 (held out; operating point chosen on dev300 before looking at test)

| readout | gt_acc | ALA |
|---|---|---|
| baseline (prefill-argmax) | 9.86 | 77.43 |
| baseline (generate+parse, earlier run) | 6.86 | 78.57 |
| deployed content K50 α16 (generate) | 29.43 | 44.43 |
| meanplustvar K50 α16 (generate) | 33.57 | 45.00 |
| meanplustvar K50 α16 (prefill-argmax) | 34.43 | 43.43 |
| meanplustvar K100 α16 (prefill-argmax) | 31.29 | 45.29 |
| **Steer-CD K100 λ0.75 (PRIMARY, dev-chosen)** | **49.43** | **13.43** |
| Steer-CD K50 λ1.0 (best post-hoc) | 50.43 | 10.71 |

Robustness (test700): K ∈ {50,100} × λ ∈ {0.5, 0.75, 1.0} all land in
48.0–50.4. λ=0.25: 44–45. λ=1.5: 45.7–46.4. Broad plateau, no knife-edge.

No-steer CD ablation (test700): real − λ·blank: 5.29–5.57; real − λ·noise:
6.14–7.57 (λ ∈ {0.5, 0.75, 1.0}). Fails.

### Per-task, test700 (steer-only K100 prefill → Steer-CD K100 λ0.75)

| task | steer | +CD |
|---|---|---|
| age_prediction | 25.7 | 42.9 |
| emotion_recognition | 30.0 | 64.3 |
| gender_prediction | 50.0 | 97.1 |
| intonation_perception | 32.9 | 51.4 |
| pitch_comparison | 30.0 | 37.1 |
| speaker_identity_recognition | 35.7 | 34.3 |
| speed_comparison | 21.4 | 31.4 |
| total_speaker_counting | 42.9 | 54.3 |
| vocal_range_comparison | 22.9 | 32.9 |
| volume_comparison | 21.4 | 48.6 |

9/10 tasks improve; the one exception is flat (−1.4).

### dev300 (development set; where the operating point was chosen)

| readout | gt_acc | ALA |
|---|---|---|
| baseline (prefill) | 7.00 | 77.33 |
| meanplustvar K50 α16 (prefill = generate) | 34.67 | 43.00 |
| Steer-CD K100 λ0.75 | **57.67** | 15.00 |
| Steer-CD K50 λ0.75 | 56.00 | 15.33 |
| no-steer CD (all λ, blank or noise amateur) | 4.67–8.33 | 68–84 |
| steered expert with blank/noise amateur | 29.3–34.3 | 32–40 |

The amateur must be the unsteered model on the same real audio; blank/noise
amateurs give nothing beyond steering alone.

Per-task dev300 (steerK100 → +CD λ0.75): age 23.3→56.7, emotion 36.7→63.3,
gender 50→100, intonation 36.7→60.0, pitch 26.7→46.7, speaker_id 36.7→43.3,
speed 26.7→43.3, counting 40.0→63.3, vocal_range 20.0→56.7, volume 23.3→43.3.

### MMAU mirror image (mechanism check, mmau_sub300)

Prediction: on a no-conflict benchmark the baseline distribution IS signal, so
subtracting it should hurt monotonically in λ. Confirmed:

| readout | overall (music/sound/speech) |
|---|---|
| baseline (prefill) | 74.67 (71/77/76) |
| steer K50 α16 | 58.00 (64/60/50) |
| Steer-CD K50 λ0.25 | 49.33 (60/52/36) |
| Steer-CD K50 λ0.5 | 41.33 (53/45/26) |
| Steer-CD K50 λ0.75 | 32.00 (45/38/13) |
| Steer-CD K50 λ1.0 | 13.67 (19/9/13) |

One λ knob: on conflict data more contrast = more gain; on clean data more
contrast = more harm. The method is a conflict-resolution instrument by design.

## Protocol note (be transparent in the paper)

Steer-CD is a logit readout, not generate+parse. Equivalence check: per-config
prefill-argmax accuracy equals generate+parse accuracy (34.67 vs 34.67 for the
champion on dev300; baseline 7.00 vs 8.67; test 34.43 vs 33.57), although
per-item agreement is only 72% — the two readouts differ per sample but not in
aggregate. Report the CD numbers as logit-readout with this equivalence check
alongside. Official generate+parse steering numbers remain in
`methods_exp/out/test700*/methods_summary.json`.

## Negative results from this round (all with data)

1. **Affine intervention family is saturated** (dev300, meanplustvar selector,
   baseline 8.67, champion scale K50 α16 = 34.67):
   contrast-around-noise-ref 32.67 (α24: 30.33), blank-contrast 34.00,
   bidirectional (amplify top-K + attenuate bottom-K ×0.25) 34.67 (tie),
   suppress-only 8.67 (exact no-op), graded gain K400 20.00,
   contrast+bidir 32.33, scale α24 32.67, scale K100 32.33.
   The fc2 pre-hook is affine (z·gain + add), so this covers the whole
   per-neuron affine class reachable at that hook.
2. **Confidence-based per-clip config selection fails**: argmax-margin over
   {baseline + 5 steered} = 15.67 because the baseline is anti-calibrated —
   its margin is HIGHER when wrong (2.81) than when right (0.70): the language
   prior corrupts confidence too. Steered-only argmax-margin 32.67 < 34.67;
   default+switch never exceeds the default.
3. **Majority-vote ensembles capture almost none of the oracle headroom**:
   oracle union over 5 steered configs = 44.43 (test700) but best vote = 34.71
   (dynamics-family 3-way; +1.14). Steer-CD reaches 49.4+ and eats that
   headroom properly.
4. (Prior rounds, `iaan-tvar-selector` memory): selector family saturated at
   meanplustvar; label-aware causal attribution upper bound 33.33 < 34.67.

## Files and reproduction

Runners (all need the `_steer_shim` PYTHONPATH; omni conda env):

```
SCR=/livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/scripts
export PYTHONPATH="$PWD/_steer_shim:$SCR/steering:$SCR/ablation:$SCR/pipeline:$PWD"
PY=/home/khlu/miniconda3/envs/omni/bin/python
```

- `methods_exp/collect_letter_logits.py` — letter logits for variants
  real/blank/noise + steered configs (VoxParadox). Produced:
  `out/cll_test700.jsonl`, `out/cll_dev300_audioablate.jsonl`.
  ```
  $PY methods_exp/collect_letter_logits.py --data_path data/voxparadox_test700.json \
    --audio_base data --out methods_exp/out/cll_test700.jsonl \
    --variants real,blank,noise,meanplustvar:scale:50:16,meanplustvar:scale:100:16
  ```
- `methods_exp/collect_letter_logits_mmau.py` — same for MMAU. Produced
  `out/cll_mmau300.jsonl`.
- `methods_exp/omni_perq_confpick.py` — per-config margins + official generates
  (dev300 study). Produced `out/confpick_dev300/` (margins.jsonl has the
  steered letter logits used for dev CD numbers).
- `methods_exp/omni_perq_interventions.py` — affine intervention sweep
  (config grammar `method:op:K:alpha[:beta]`, ops scale/contrast/blankcontrast/
  bidir/suppress/graded/contrastbidir). Produced `out/intv_dev300/`.
- Offline analysis: `eval_letter_cd.py` (CD sweeps), `eval_confpick.py`
  (selection rules + calibration), `overlap_analysis.py` (oracle union),
  `ensemble_vote.py` (majority vote).
- CD evaluation itself is ~20 lines: parse letters from the jsonl records and
  take the contrasted argmax (see the inline snippets in the session or
  eval_letter_cd.py).

Data splits: `data/voxparadox_dev300.json` (op-point selection),
`data/voxparadox_test700.json` (held out, balanced 70/task, seed 0),
`audio_caption/MMAU/mmau_sub300.json`.

## Per-token CD generation (official protocol restored)

`methods_exp/omni_perq_tokencd.py`: two KV-cache branches over the identical
token sequence (steered vs unsteered prefill; the encoder only runs at prefill,
so decode steps differ only through the caches); each greedy step picks
argmax[logits_steered − λ·logits_unsteered]. λ=0 reproduces the hooked
`model.generate` outputs token-for-token (verified). Evaluated with the official
generate+parse eval.py:

| dev300, expert mpt K50 α16 | gt_acc | ALA |
|---|---|---|
| λ=0 (= steered greedy) | 34.67 | 44.00 |
| λ=0.5 | 50.33 | 23.00 |
| **λ=0.75** | **51.00** | 14.00 |
| λ=1.0 | 9.00 (degenerate text) | 5.67 |

So the gain survives the official generation protocol: 51.00 vs 34.67. It is a
few points below the pure letter readout (56–57.7) because contrasting every
token also perturbs non-answer tokens; λ=1 derails fluency entirely. The
standard adaptive-plausibility constraint (contrast only among tokens with
p_expert ≥ 0.1·max) fixes both: dev300 λ0.75 → 52.67, λ1.0 → 52.33 (from 9.00).

**test700, official generate+parse, dev-chosen point (λ0.75, plaus 0.1):
48.71 / ALA 15.57** — within 0.7pp of the letter readout (49.43), so the
protocol question is closed: the improvement does not depend on the logit
readout. Cost: 2 prefills + 2× decode compute.

## λ dev-transfer: MMSU_para (negative, diagnosed)

MMSU_para dev600 (12 tasks, 9/10 overlap with VoxParadox task types):
baseline 38.83; steering alone slightly hurts (36.2–37.5); Steer-CD hurts
monotonically in λ (λ1.5 → 17.7). Despite the task overlap, MMSU_para lacks the
transcript-conflict structure, so its baseline is healthy and subtracting it
removes signal — the same mirror-image behavior as MMAU. Consequence: λ cannot
be tuned on an external no-conflict benchmark (it would pick λ=0); what must
transfer is the conflict pathology, not the task list (extends the earlier
LISTEN K/α transfer failure). Practical guidance: the λ plateau is broad
(0.5–1.0 all within ~1pp of each other on test700), so a global default λ=0.75
is safe wherever the method's scope condition holds (floored baseline + high
ALA, i.e. the conflict pathology is present).

## Cross-model validation (AF3, Kimi-Audio)

Same recipe, per model: dev300 picks the operating point (selector, K, α, λ),
test700 is held out. Selection on each model's audio-encoder fc2 (AF3
audio_tower, Kimi Whisper speech_encoder), real-vs-noise per clip, z←αz on
top-K, amateur = unsteered same model on the same real audio.

test700, dev-chosen operating points:

| model (audio encoder) | baseline gt/ALA | steer-only gt/ALA | Steer-CD gt/ALA | Δgt | op point |
|---|---|---|---|---|---|
| Qwen2.5-Omni-7B (Whisper+thinker) | 9.86 / 77.4 | 34.43 / 43.4 | **49.43** / 13.4 | +15.0 | mpt K100 λ0.75 |
| Audio-Flamingo-3 7B (own tower)   | 15.00 / 67.4 | 32.00 / 36.7 | **48.00** / 10.4 | +16.0 | mpt K50α8 λ1.0 |
| Kimi-Audio 7B (Whisper enc)       | 20.29 / 65.7 | 30.43 / 44.1 | **32.71** / 27.1 | +2.3  | content K200α12 λ0.5 |

The method transfers to three different architectures: Steer-CD strips the
prior in all three (large ALA drops: 43→13, 37→10, 44→27) and never hurts gt
vs steering alone at the λ plateau. The *size* of the gt gain tracks how much
correct audio evidence steering has surfaced under the prior: Qwen and AF3
(floored baselines 9.9/15.0, ALA 77/67) gain +15/+16; Kimi (the least
pathological, baseline gt already 20.3 and its steering recovers most of the
gap on its own) gains +2.3 — CD there drops ALA a lot but reveals little extra
correct evidence. Consistent with the mirror-image control: benefit exists only
where a floored, prior-dominated baseline hides audio evidence.

- AF3 dev300: real 13.67; steer-only (mpt K50α8) 31.33; CD sweep peaks λ=1.0 at
  49.00 (broad λ∈[0.75,1.25]). test700 at that point: 48.00 (steer-only 32.00).
  9/10 tasks up.
- Kimi dev300: real 19.67; steer-only (content K200α12) 29.00; CD peaks λ=0.5 at
  33.00 (broad λ∈[0.25,1.5], all 31.7–33.0). test700 at that point: 32.71
  (steer-only 30.43, best post-hoc λ0.25 = 33.29). 6/10 tasks up, emotion the
  one clear regression (18.6→5.7).

Higher λ plateau on AF3 (≈1.0) than Qwen (≈0.75) than Kimi (≈0.5) is consistent
with heavier prior → more contrast tolerated.

Collectors: `methods_exp/af3_steer_cd_collect.py` (af3 HF-path venv
`/tmp/af3_venv/bin/python`, forced-"(" letter readout, AudioFlamingo3 audio
tower) and `methods_exp/kimi_steer_cd_collect.py` (kimi conda env, bare-letter
readout via `alm.forward` text_logits, Whisper speech_encoder; TikToken
tokenizer `encode(s, bos=False, eos=False)`). Both write the same letters.jsonl
schema, evaluated with `eval_cd_letters.py`. Outputs: `out/cll_af3_{dev300,
test700}.jsonl`, `out/cll_kimi_{dev300,test700}.jsonl`.
