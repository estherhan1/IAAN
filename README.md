<div align="center">

# IAAN: Identifying and Amplifying Acoustic Neurons

**Enhancing How Large Audio-Language Models Hear: Identifying and Amplifying Acoustic Neurons in the Audio Encoder**

**IEEE Spoken Language Technology Workshop (SLT), 2026**

Yu-Han Huang · Chih-Kai Yang · Ke-Han Lu · An-Yu Cheng · Hung-yi Lee

[**Paper**](https://arxiv.org/abs/2607.11801) · [**OpenReview**](https://openreview.net/forum?id=aZaFtJs6dr) · [**Project Page source**](docs/index.html) · [**VoxParadox**](https://huggingface.co/datasets/IHP-Lab/VoxParadox)

</div>

## Overview

IAAN improves acoustic understanding in large audio-language models without training or labeled calibration data. For each clip, it contrasts real-audio and Gaussian-reference activations in the **audio encoder**, ranks feed-forward neurons by their signed activation difference, and amplifies only the top-`K` neurons during inference.

This repository's **main reproduction entry point focuses only on Table I (baseline vs. IAAN)**. It supports Audio-Flamingo-3, Qwen2.5-Omni, and Kimi-Audio, selecting any combination of the ten VoxParadox tasks.

## Quick start

Install the model-specific dependencies described in [Reproduction setup](paper_reproduction/README.md) and use a CUDA GPU. The launcher can fetch the VoxParadox dataset and its official evaluator from Hugging Face.

```bash
# All three models, all 10 tasks; paper-selected parameters
bash reproduce_main_table.sh

# One model, two tasks
bash reproduce_main_table.sh --model qwen --tasks emotion,gender

# Custom parameter sweep point
bash reproduce_main_table.sh --model af3,kimi --tasks pitch,speed --k 100 --gain 8
```

Paper-selected defaults (Table II):

| Model | K | Gain |
|---|---:|---:|
| Audio-Flamingo-3 | 50 | 16 |
| Qwen2.5-Omni | 50 | 12 |
| Kimi-Audio | 200 | 12 |

**Published VoxParadox full-test main-table results** (accuracy %, baseline → IAAN):

| Model | Baseline | IAAN | Difference |
|---|---:|---:|---:|
| Audio-Flamingo-3 | 13.1 | 38.8 | +25.7 |
| Qwen2.5-Omni | 7.6 | 29.0 | +21.4 |
| Kimi-Audio | 21.2 | 30.9 | +9.7 |

These published values are for reference; they are not injected into re-run outputs. **Model integrations have not yet been GPU-tested against the published table.** Different model/library versions, prompts or preprocessing may change results.

## Repository layout

```text
reproduce_main_table.sh            # User-facing one-command entry point
paper_reproduction/
  infer_main_table.py              # 3 model adapters + baseline/IAAN inference
  iaan_core.py                     # FFN neuron selection and amplification
  run_main_table.py                # Model/task CLI, dataset, official evaluation
  main_table.py                    # Results table generator
  tests/                           # Unit tests
docs/
  index.html                       # Academic project page
  static/css/                     # Academic Project Page Template + IAAN styling
```

This release retains only the IAAN Main Table reproduction code and the project website. Previous Steer-CD and ablation experiment archives have been removed from the current branch; their history remains accessible through earlier Git commits.

## Installation, data and outputs

Full instructions and model-specific setup: **[paper_reproduction/README.md](paper_reproduction/README.md)**.

The runner writes `outputs/main_table/<model>/baseline_predictions.jsonl`,
`iaan_predictions.jsonl`, and corresponding official `*_report.json` files,
then aggregates selected tasks into `outputs/main_table/Main_Table.md`.

## Citation

```bibtex
@inproceedings{huang2026iaan,
  title={Enhancing How Large Audio-Language Models Hear: Identifying and Amplifying Acoustic Neurons in the Audio Encoder},
  author={Huang, Yu-Han and Yang, Chih-Kai and Lu, Ke-Han and Cheng, An-Yu and Lee, Hung-yi},
  booktitle={Proceedings of the IEEE Spoken Language Technology Workshop (SLT)},
  year={2026}
}
```

## Acknowledgments

The project page adapts the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (CC BY-SA 4.0). The code README structure follows the simple installation / quick start / outputs pattern of [LLaVAShield](https://github.com/leost123456/LLaVAShield). VoxParadox data and official evaluation are maintained by [IHP-Lab](https://huggingface.co/datasets/IHP-Lab/VoxParadox), under its own license.
