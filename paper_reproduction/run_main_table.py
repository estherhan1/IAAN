#!/usr/bin/env python3
"""One-command IAAN main-table runner. Supports any model/task subset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

TASK_NAMES = (
    "age_prediction", "gender_prediction", "emotion_recognition",
    "pitch_comparison", "volume_comparison", "speed_comparison",
    "vocal_range_comparison", "intonation_perception",
    "speaker_identity_recognition", "total_speaker_counting",
)
ALIASES = {
    "age": TASK_NAMES[0], "gender": TASK_NAMES[1], "emotion": TASK_NAMES[2],
    "pitch": TASK_NAMES[3], "volume": TASK_NAMES[4], "speed": TASK_NAMES[5],
    "range": TASK_NAMES[6], "intonation": TASK_NAMES[7],
    "speaker_id": TASK_NAMES[8], "speaker_count": TASK_NAMES[9],
    "spk_id": TASK_NAMES[8], "spk_cnt": TASK_NAMES[9],
}
ALIASES.update({name: name for name in TASK_NAMES})
DEFAULTS = {"af3": (50, 16.), "qwen": (50, 12.), "kimi": (200, 12.)}


def parse_list(value: str, valid: set[str], aliases: dict[str, str] | None = None) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULTS if aliases is None else TASK_NAMES)
    mapping = aliases if aliases is not None else {x: x for x in valid}
    selected = []
    for raw in value.split(","):
        key = raw.strip().lower().replace("-", "_")
        item = mapping.get(key)
        if item is None or item not in valid:
            raise ValueError(f"Unknown choice '{raw}'. Options: {', '.join(sorted(mapping))}")
        if item not in selected:
            selected.append(item)
    if not selected:
        raise ValueError("Empty selection")
    return selected


def select_data(source: Path, target: Path, tasks: list[str]) -> int:
    items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("VoxParadox dataset must be a list of JSON records")
    if len({str(it["id"]) for it in items}) != len(items):
        raise ValueError("Duplicate sample IDs")
    selected = []
    count = {key: 0 for key in tasks}
    for i, item in enumerate(items):
        task = item.get("task_name") or str(item["id"]).split("__", 1)[0]
        if task not in tasks:
            continue
        row = dict(item)
        row["_iaan_index"] = i  # Preserve full-dataset index for per-clip RNG on subsets.
        selected.append(row)
        count[task] += 1
    if not selected or any(num == 0 for num in count.values()):
        raise ValueError(f"Missing task(s) in dataset: {count}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(selected)} samples: {count}", flush=True)
    return len(selected)


def obtain_dataset():
    from huggingface_hub import snapshot_download
    print("Downloading VoxParadox metadata, evaluator, and audio (first run only)...", flush=True)
    return Path(snapshot_download(
        repo_id="IHP-Lab/VoxParadox", repo_type="dataset",
        allow_patterns=["voxparadox.json", "eval.py", "audio/**"],
    ))


def official_evaluator(dataset_root: Path | None, specified: Path | None):
    if specified:
        if not specified.is_file():
            raise FileNotFoundError(specified)
        return specified
    if dataset_root and (dataset_root / "eval.py").is_file():
        return dataset_root / "eval.py"
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id="IHP-Lab/VoxParadox", repo_type="dataset", filename="eval.py"
    ))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate baseline and IAAN on VoxParadox (main paper table only).")
    ap.add_argument("--model", "--models", default="all",
                    help="af3,qwen,kimi or all (default all)")
    ap.add_argument("--tasks", default="all",
                    help="age,gender,emotion,pitch,volume,speed,range,intonation,speaker_id,speaker_count or all")
    ap.add_argument("--k", type=int, default=None,
                    help="Override paper K for all selected models")
    ap.add_argument("--gain", "--g", type=float, default=None,
                    help="Override paper gain for all selected models")
    ap.add_argument("--data", type=Path, default=None,
                    help="Local voxparadox.json; defaults to HF dataset download")
    ap.add_argument("--audio-base", type=Path, default=None,
                    help="Dataset directory containing audio/; defaults to parent of --data")
    ap.add_argument("--eval-script", type=Path, default=None,
                    help="Local official VoxParadox eval.py; otherwise download from HF")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/main_table"))
    ap.add_argument("--checkpoint", default=None,
                    help="Model checkpoint override (only when --model chooses one model)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        models = parse_list(args.model, set(DEFAULTS))
        tasks = parse_list(args.tasks, set(TASK_NAMES), ALIASES)
        if args.k is not None and args.k < 1:
            ap.error("--k must be positive")
        if args.gain is not None and args.gain <= 1:
            ap.error("--gain must be > 1")
        if args.checkpoint and len(models) != 1:
            ap.error("--checkpoint requires selecting exactly one model")
        if args.max_new_tokens < 1:
            ap.error("--max-new-tokens must be positive")
        settings = {m: (args.k if args.k is not None else DEFAULTS[m][0],
                        args.gain if args.gain is not None else DEFAULTS[m][1])
                    for m in models}
        print("Selected models:", ", ".join(models))
        print("Selected tasks:", ", ".join(tasks))
        print("IAAN settings:", settings, flush=True)
        if args.dry_run:
            return

        root = obtain_dataset() if args.data is None else args.data.resolve().parent
        data = args.data.resolve() if args.data else root / "voxparadox.json"
        if not data.is_file():
            raise FileNotFoundError(f"Dataset JSON missing: {data}")
        audio_base = (args.audio_base.resolve() if args.audio_base else root)
        if not audio_base.is_dir():
            raise FileNotFoundError(f"Audio directory missing: {audio_base}")
        evaluator = official_evaluator(root, args.eval_script)
        out = args.output_dir.resolve()
        subset = out / "selected_dataset.json"
        select_data(data, subset, tasks)
        inf_script = Path(__file__).with_name("infer_main_table.py")
        for model, (k, gain) in settings.items():
            model_dir = out / model
            model_dir.mkdir(parents=True, exist_ok=True)
            py = os.environ.get(f"IAAN_{model.upper()}_PYTHON", sys.executable)
            cmd = [
                py, str(inf_script), "--model", model, "--data", str(subset),
                "--audio-base", str(audio_base), "--out", str(model_dir),
                "--k", str(k), "--gain", str(gain),
                "--max-new-tokens", str(args.max_new_tokens),
            ]
            if args.checkpoint:
                cmd.extend(["--checkpoint", args.checkpoint])
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            for kind in ("baseline", "iaan"):
                pred = model_dir / f"{kind}_predictions.jsonl"
                report = model_dir / f"{kind}_report.json"
                subprocess.run([sys.executable, str(evaluator),
                                "--dataset", str(subset),
                                "--predictions", str(pred),
                                "--report", str(report)], check=True)
        from main_table import render
        table = render(out, models=models, tasks=tasks)
        (out / "Main_Table.md").write_text(table, encoding="utf-8")
        print(table)
        print("Saved:", out / "Main_Table.md")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError,
            ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
