# IAAN SLT 2026 reproduction interface

**Status: command-line interface implemented, complete paper reproduction not yet runnable** without the original model-specific official greedy-generation runners and VoxParadox dataset. All changes are in this draft branch; Steer-CD originals remain untouched.

## One-line invocation

Run from the repository root:

```bash
# All three models, all 10 VoxParadox tasks, published paper's K/g
bash reproduce_main_table.sh

# One model, two tasks
bash reproduce_main_table.sh --model qwen --tasks emotion,gender

# Multiple models, selected tasks, manually chosen IAAN hyperparameters
bash reproduce_main_table.sh --model af3,kimi --tasks pitch,volume,speed --k 100 --gain 8

# Print the exact selection and K/g without running any model
bash reproduce_main_table.sh --model qwen --tasks emotion --dry-run

# Show all options
bash reproduce_main_table.sh --help
```

**Published paper defaults (Table II, † selected on LISTEN):**

| Model | K | gain |
|---|---:|---:|
| AF3 | 50 | 16 |
| Qwen2.5-Omni | 50 | 12 |
| Kimi-Audio | 200 | 12 |

`--k` and `--gain` override defaults for every selected model.
A subset of tasks produces a subset report, **not** the full paper Table I.
There is **no dev sweep** or tuning stage; the paper-selected settings are
used unless overridden.

The task selectors are `age,gender,emotion,pitch,volume,speed,range,intonation,speaker_id,speaker_count`; `all` is the default.

The command writes a filtered `selected_dataset.json` and expects original model
runners to emit `baseline_report.json` and `iaan_report.json` in
`outputs/main_table/<model>/` in the official `data/eval.py` JSON schema.
`main_table.py` then assembles `outputs/main_table/Main_Table.md`.

## Required setup — not yet included in repository

The original `af3_perq_official.py`, `omni_perq_official.py`, and
`kimi_perq_official.py` generation scripts, their model-specific dependencies,
and the complete VoxParadox data were not found in this GitHub repository.
The existing `2_steer_cd/` collectors compute letter logits and should not be
silently substituted for greedy generation.

1. Supply the complete VoxParadox JSON and audio files (no data bundled).
2. Copy `paper_reproduction/main_table.env.example` to
   `paper_reproduction/main_table.env` and set `IAAN_VOXPARADOX_JSON`,
   `IAAN_AUDIO_BASE`, and the three `IAAN_*_CMD` environment variables.
3. Each command must consume `IAAN_DATA_PATH`, `IAAN_K`, `IAAN_GAIN`
   and emit official JSON reports to `IAAN_MODEL_OUTPUT_DIR`.
4. Run one of the one-line commands above. The launcher rejects missing
   runners and reports instead of presenting unverified results.

All models run with **greedy model-specific generation**. Unamplified baseline
is evaluated for the selected task subset, then IAAN. The signed score
`mean(real) - mean(noise)` selects top-K neurons across the entire encoder;
selected FFN activations are scaled by gain. No Steer-CD / contrastive decoding.

The `iaan_core.py` file contains adapter-independent IAAN primitives.
These are not full model adapters. CPU checks:

```bash
cd paper_reproduction
python -m unittest discover -s tests -v
```

## Paper reference

*Enhancing How Large Audio-Language Models Hear: Identifying and Amplifying
Acoustic Neurons in the Audio Encoder*, IEEE SLT 2026.
Full-test published baseline → IAAN: AF3 13.1 → 38.8; Qwen 7.6 → 29.0;
Kimi 21.2 → 30.9. This launcher never copies those values into reports.
