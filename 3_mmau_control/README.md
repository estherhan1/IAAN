# 3_mmau_control — no-conflict mechanism check

MMAU is a general audio-QA benchmark where the correct answer does **not** conflict with
the language prior: the baseline model is already audio-grounded (68% overall). This is
the control that tells us whether the VoxParadox win is a generic "steering helps" effect
or specifically a conflict-resolution effect. It is the latter.

## Scripts

| Script | Purpose |
|---|---|
| `omni_mmau_methods.py` | Same selection + intervention machinery as `omni_perq_methods.py`, on MMAU. Reports overall and per-task (sound / music / speech) accuracy. |
| `collect_letter_logits_mmau.py` | MMAU letter-logit collection for the offline Steer-CD readout, self-contained (records carry the item's choices / answer / task). |

## Results

`results/mmau300` (300 balanced MMAU items), overall accuracy:

| config | overall | music | sound | speech |
|---|---|---|---|---|
| baseline | 68.0 | 57 | 71 | 76 |
| content `K50 a8` | 64.0 | 50 | 70 | 72 |
| content `K50 a16` | 49.67 | 51 | 53 | 45 |
| tvar `K50 a16` | 53.33 | 50 | 55 | 55 |
| meanplustvar `K50 a16` | 54.33 | 49 | 61 | 53 |

Steering monotonically **hurts**, worse with larger α, across all three sub-tasks.

`results/cll_mmau300.jsonl` holds the letter logits for the CD readout. Applying
Steer-CD here makes it worse still and monotonically in λ (documented in
`../STEER_CD_REPORT.md`): where the audio and the prior agree, the unsteered
distribution is the signal, and subtracting it destroys accuracy.

## Interpretation

This is the exact mirror image of the VoxParadox result. On VoxParadox (audio conflicts
with the prior) steering + CD lifts accuracy from ~7 to ~49; on MMAU (no conflict) the
same operations drop it from 68 toward the 40s. The clean dose-response in both
directions is the evidence that Steer-CD resolves audio-vs-prior conflict rather than
generically boosting audio attention.

Re-run: `$PY omni_mmau_methods.py --data_path $MMAU/mmau_sub300.json --audio_base $MMAU --output_dir methods_exp/out/mmau300 --methods content,tvar,meanplustvar --ks 50 --alphas 8,16`
