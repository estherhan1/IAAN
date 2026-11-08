#!/usr/bin/env python3
"""Validate and aggregate official VoxParadox reports into SLT Table I.

Each model requires baseline_report.json and iaan_report.json with the
official data/eval.py shape:
  {"n_examples": 2000, "n_missing": 0, "overall":{"gt_acc":..., "ala":...},
   "per_task":{"age_prediction":{"n":200,"gt_acc":..., "ala":...}, ...}}
Outputs raw reproduced metrics, not hard-coded paper results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TASKS = (
    ("Age", "age_prediction"),
    ("Gender", "gender_prediction"),
    ("Emotion", "emotion_recognition"),
    ("Pitch", "pitch_comparison"),
    ("Volume", "volume_comparison"),
    ("Speed", "speed_comparison"),
    ("Range", "vocal_range_comparison"),
    ("Intonation", "intonation_perception"),
    ("Spk ID", "speaker_identity_recognition"),
    ("Spk Cnt", "total_speaker_counting"),
)
MODELS = ("af3", "qwen", "kimi")


def read_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("n_examples") != 2000 or report.get("n_missing") != 0:
        raise ValueError(f"{path}: expected all 2000 items, no missing predictions")
    tasks = report.get("per_task", {})
    actual = set(tasks)
    expected = {task for _, task in TASKS}
    if actual != expected:
        raise ValueError(f"{path}: task mismatch; missing={expected-actual}; extra={actual-expected}")
    for _, task in TASKS:
        row = tasks[task]
        if row.get("n") != 200:
            raise ValueError(f"{path}: {task} requires exactly 200 evaluated items")
        for metric in ("gt_acc", "ala"):
            if not isinstance(row.get(metric), (int, float)):
                raise ValueError(f"{path}: missing numeric {task}.{metric}")
    for metric in ("gt_acc", "ala"):
        if not isinstance(report.get("overall", {}).get(metric), (int, float)):
            raise ValueError(f"{path}: overall.{metric} is missing")
    # Match the paper's 10-task macro averages, not a masked subset.
    for metric in ("gt_acc", "ala"):
        macro = sum(tasks[name][metric] for _, name in TASKS) / len(TASKS)
        if abs(macro - report["overall"][metric]) > 0.15:
            raise ValueError(f"{path}: overall {metric} differs from 10-task macro average")
    return report


def render(results: Path) -> str:
    header = ["Model", "Method", *[name for name, _ in TASKS], "Avg.", "ALA"]
    rows = ["# Reproduced VoxParadox Main Table", "",
            "All values below are computed from complete 2,000-item official evaluation reports.",
            "The published paper's values are **not** substituted.", "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|"]
    for model in MODELS:
        for mode in ("baseline", "iaan"):
            report = read_report(results / model / f"{mode}_report.json")
            row = [model, mode]
            row += [f"{report['per_task'][key]['gt_acc']:.1f}" for _, key in TASKS]
            row += [f"{report['overall']['gt_acc']:.1f}", f"{report['overall']['ala']:.1f}"]
            rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    table = render(args.results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
