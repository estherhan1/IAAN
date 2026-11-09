# Reproducing IAAN Table I

This directory contains an IAAN baseline-versus-intervention inference pipeline for **Audio-Flamingo-3, Qwen2.5-Omni, and Kimi-Audio**, selecting any combination of the 10 VoxParadox tasks. No Steer-CD experiments, calibration sweeps, or extra benchmarks run here.

> **Validation status:** The model adapters are based on public model inference APIs and previous IAAN experiments, but the three complete GPU pipelines have **not been tested end to end** with the paper's exact model revisions. The paper numbers are not guaranteed. The first Qwen/AF3/Kimi runs should be treated as validation runs and reviewed before calling them a faithful replication.

## Environment setup

Python 3.10 or newer and a CUDA-enabled GPU with sufficient VRAM for the chosen model. Install PyTorch for your CUDA version, then the dependencies for each model. If package versions conflict, create **separate environments** and point the launcher to each Python interpreter.

```bash
# Basic CLI / evaluation dependencies
python -m pip install huggingface_hub numpy tqdm

# Qwen2.5-Omni environment
python -m pip install torch transformers accelerate qwen-omni-utils librosa soundfile numpy tqdm

# Audio-Flamingo-3 environment (separate env recommended)
python -m pip install torch transformers accelerate librosa soundfile numpy tqdm

# Kimi-Audio environment (separate env recommended)
python -m pip install torch numpy librosa soundfile tqdm
python -m pip install git+https://github.com/MoonshotAI/Kimi-Audio.git
```

These are dependency families rather than frozen tested versions; resolve any model-specific requirements following the upstream model repositories.

If you use separate environments, set their Python executables once:

```bash
export IAAN_AF3_PYTHON=/path/to/af3-env/bin/python
export IAAN_QWEN_PYTHON=/path/to/qwen-env/bin/python
export IAAN_KIMI_PYTHON=/path/to/kimi-env/bin/python
```

The launcher itself can run from any Python with `huggingface_hub` installed. Each inference subprocess uses its model-specific interpreter.

## Dataset

By default, the first run downloads the **VoxParadox** dataset from the [IHP-Lab Hugging Face repository](https://huggingface.co/datasets/IHP-Lab/VoxParadox), including `voxparadox.json`, the `audio/` files, and the official `eval.py`.

If the dataset is already available locally, avoid downloading it again:

```bash
bash reproduce_main_table.sh --data /path/to/voxparadox.json --audio-base /path/to/dataset-root
```

`--audio-base` is the root against which each item's `audio_path` is resolved. The official scorer is downloaded separately if `eval.py` is not present beside the dataset; alternatively supply `--eval-script /path/to/eval.py`.

The dataset carries its own research-use terms. No audio or model weights are redistributed here.

## Usage

```bash
# Paper defaults: all 3 models, all 10 tasks
bash reproduce_main_table.sh

# Select model / tasks
bash reproduce_main_table.sh --model qwen --tasks gender,emotion
bash reproduce_main_table.sh --model af3,kimi --tasks pitch,speed,volume

# Override published K/g for the chosen models
bash reproduce_main_table.sh --model kimi --tasks all --k 100 --gain 8

# Review CLI selection without downloading the dataset or loading a model
bash reproduce_main_table.sh --model qwen --tasks emotion --dry-run

# Optional checkpoint override (one model at a time)
bash reproduce_main_table.sh --model qwen --checkpoint Qwen/Qwen2.5-Omni-7B

# Show all flags
bash reproduce_main_table.sh --help
```

Model aliases are `af3`, `qwen`, `kimi`; task aliases are
`age`, `gender`, `emotion`, `pitch`, `volume`, `speed`, `range`,
`intonation`, `speaker_id`, `speaker_count`. `all` is the default.

The paper-selected **fixed** settings from Table II are:

| Model | K | Gain g |
|---|---:|---:|
| AF3 | 50 | 16 |
| Qwen2.5-Omni | 50 | 12 |
| Kimi-Audio | 200 | 12 |

**No dev selection is performed.** `--k` and `--gain` override these settings; results on a subset of tasks are not the complete paper Table I.

## Output

```text
outputs/main_table/
  selected_dataset.json
  af3/
    baseline_predictions.jsonl
    iaan_predictions.jsonl
    baseline_report.json
    iaan_report.json
  qwen/
    ... (same structure)
  kimi/
    ... (same structure)
  Main_Table.md
```

Each `predictions.jsonl` row has the original VoxParadox `id` and a raw model `response` string. The upstream official `eval.py` computes ground-truth accuracy and adversarial-label agreement (ALA), per task and overall. No accuracy percentages are hard-coded into report files.

## How IAAN runs

For each clip, the runner:

1. Generates a **baseline response** from the real audio and collects encoder FFN neuron means before the `fc2` projection.
2. Generates once on a same-length **Gaussian reference waveform** to measure background activation. (This runs an additional whole-model pass for implementation simplicity, unlike the paper's encoder-only scoring passes.)
3. Computes **signed** `mean(real) - mean(reference)` across *all* encoder layers, selects global top-`K`.
4. Multiplies those activations by gain `g` in a fresh **real-audio** generation and saves the IAAN response.
5. Evaluates baseline and IAAN with the **official VoxParadox `eval.py`** and creates a Markdown table.

This is a main-table *reimplementation*, not a byte-for-byte archive of the original `*_perq_official.py` scripts. Differences in prompt phrasing, random-number generation, model version and processing may change measured scores. To achieve exact replication, recover and pin those original configurations. Kimi uses its upstream text generation API; AF3 and Qwen use their Hugging Face generation APIs.

## Code checks

```bash
cd paper_reproduction
python -m unittest discover -s tests -v
```

CPU unit tests cover signed selection, hook restoration, task filtering and reporting. They **do not** test the model inference adapters or reproduce the Table I numbers.

## Reference

[IAAN paper](https://arxiv.org/abs/2607.11801) · [OpenReview](https://openreview.net/forum?id=aZaFtJs6dr) · [VoxParadox dataset and official evaluator](https://huggingface.co/datasets/IHP-Lab/VoxParadox) · [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio)
