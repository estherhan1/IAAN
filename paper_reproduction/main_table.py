#!/usr/bin/env python3
"""Aggregate official greedy-generation VoxParadox JSON reports.

Accepts subsets for exploratory runs. Reports real observed values, not the
published paper numbers. Full Table I is available when all tasks were run.
"""
import argparse
import json
from pathlib import Path

TASKS = (
    ("Age", "age_prediction"), ("Gender", "gender_prediction"),
    ("Emotion", "emotion_recognition"), ("Pitch", "pitch_comparison"),
    ("Volume", "volume_comparison"), ("Speed", "speed_comparison"),
    ("Range", "vocal_range_comparison"), ("Intonation", "intonation_perception"),
    ("Spk ID", "speaker_identity_recognition"),
    ("Spk Cnt", "total_speaker_counting"),
)
MODELS = ("af3", "qwen", "kimi")


def read_report(path: Path, tasks=None) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    task_names = list(tasks or [key for _, key in TASKS])
    data = report.get("per_task", {})
    if set(data) != set(task_names):
        raise ValueError(f"{path}: per-task mismatch (expected {task_names}, found {list(data)})")
    count = sum(data[name].get("n", 0) for name in task_names)
    if count <= 0 or report.get("n_examples") != count or report.get("n_missing") != 0:
        raise ValueError(f"{path}: missing examples or inconsistent n_examples")
    for name in task_names:
        if data[name].get("n", 0) <= 0:
            raise ValueError(f"{path}: empty task {name}")
        for metric in ("gt_acc", "ala"):
            if not isinstance(data[name].get(metric), (int, float)):
                raise ValueError(f"{path}: missing {name}.{metric}")
    # Macro mean matches the paper when each task has equal count; for
    # unequal subsets we explicitly compute the mean across selected tasks.
    report["_macro"] = {
        key: sum(data[name][key] for name in task_names) / len(task_names)
        for key in ("gt_acc", "ala")
    }
    return report


def render(results: Path, models=None, tasks=None) -> str:
    models = list(models or MODELS)
    task_keys = list(tasks or [key for _, key in TASKS])
    labels = [label for label, key in TASKS if key in task_keys]
    cols = ["Model", "Method", *labels, "Avg.", "ALA"]
    lines = [
        "# IAAN VoxParadox Results", "",
        "Actual results from official model greedy-generation runs; not hard-coded paper scores.",
        "Avg. and ALA are macro averages across selected tasks.", "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for model in models:
        for kind in ("baseline", "iaan"):
            rep = read_report(results / model / f"{kind}_report.json", task_keys)
            values = [model, kind]
            values += [f"{rep['per_task'][key]['gt_acc']:.1f}"
                       for _, key in TASKS if key in task_keys]
            values += [f"{rep['_macro']['gt_acc']:.1f}", f"{rep['_macro']['ala']:.1f}"]
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = render(args.results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
