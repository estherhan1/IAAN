#!/usr/bin/env python3
"""Select models/tasks and dispatch the original IAAN inference commands.

Published SLT 2026 Table II defaults, NOT the best configurations on VoxParadox:
AF3 K50 g16, Qwen K50 g12, Kimi K200 g12.
The original model-specific official runners are external dependencies.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
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
    output = []
    for raw in value.split(","):
        key = raw.strip().lower().replace("-", "_")
        item = mapping.get(key)
        if item is None or item not in valid:
            raise ValueError(f"Unknown choice '{raw}'. Options: {', '.join(sorted(mapping))}")
        if item not in output:
            output.append(item)
    if not output:
        raise ValueError("Empty selection")
    return output


def select_data(source: Path, target: Path, tasks: list[str]) -> int:
    with source.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("VoxParadox dataset must be a JSON list of samples")
    if len({str(it["id"]) for it in items}) != len(items):
        raise ValueError("Duplicate sample ids")
    # Official data encodes task in sample id: e.g. emotion_recognition__70.
    selected = [it for it in items if str(it["id"]).split("__", 1)[0] in tasks]
    counts = {task: 0 for task in tasks}
    for it in selected:
        counts[str(it["id"]).split("__", 1)[0]] += 1
    if not selected or any(n == 0 for n in counts.values()):
        raise ValueError(f"Selected tasks missing in dataset. Counts: {counts}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected", len(selected), "examples:", counts, flush=True)
    return len(selected)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "--models", default="all",
                    help="af3,qwen,kimi or all (default all)")
    ap.add_argument("--tasks", default="all",
                    help="comma-separated: age,gender,emotion,pitch,volume,speed,range,intonation,speaker_id,speaker_count; or all")
    ap.add_argument("--k", type=int, default=None, help="Override K for selected models")
    ap.add_argument("--gain", "--g", type=float, default=None,
                    help="Override gain g for selected models")
    ap.add_argument("--data", type=Path, default=None,
                    help="full VoxParadox JSON (or set IAAN_VOXPARADOX_JSON)")
    ap.add_argument("--audio-base", type=Path, default=None,
                    help="audio directory (or set IAAN_AUDIO_BASE)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan without executing model runners")
    args = ap.parse_args()
    try:
        models = parse_list(args.model, set(DEFAULTS))
        tasks = parse_list(args.tasks, set(TASK_NAMES), ALIASES)
        if args.k is not None and args.k < 1:
            ap.error("--k must be positive")
        if args.gain is not None and args.gain <= 1:
            ap.error("--gain must exceed 1")
        dataset = args.data or (Path(os.environ["IAAN_VOXPARADOX_JSON"])
                                if os.environ.get("IAAN_VOXPARADOX_JSON") else None)
        audio_base = args.audio_base or (Path(os.environ["IAAN_AUDIO_BASE"])
                                         if os.environ.get("IAAN_AUDIO_BASE") else None)
        out = args.output_dir or Path(os.environ.get("IAAN_OUTPUT_DIR", "outputs/main_table"))
        settings = {model: (args.k or DEFAULTS[model][0],
                            args.gain if args.gain is not None else DEFAULTS[model][1])
                    for model in models}
        print("Models:", ", ".join(models))
        print("Tasks:", ", ".join(tasks))
        print("Parameters:", settings)
        if args.dry_run:
            for model in models:
                key = f"IAAN_{model.upper()}_CMD"
                print(key, "configured" if os.environ.get(key) else "MISSING original official runner")
            return
        if dataset is None or not dataset.is_file():
            ap.error("Provide --data /path/to/full_voxparadox.json")
        if audio_base is None or not audio_base.is_dir():
            ap.error("Provide --audio-base /path/to/wavs")
        for model in models:
            if not os.environ.get(f"IAAN_{model.upper()}_CMD"):
                ap.error(f"Missing IAAN_{model.upper()}_CMD: original greedy generation runner "
                         "is not included in this repository; see main_table.env.example")
        out = out.resolve()
        subset = out / "selected_dataset.json"
        select_data(dataset, subset, tasks)
        for model, (k, gain) in settings.items():
            env = os.environ.copy()
            env.update({
                "IAAN_MODEL": model, "IAAN_K": str(k), "IAAN_GAIN": str(gain),
                "IAAN_DATA_PATH": str(subset.resolve()),
                "IAAN_VOXPARADOX_JSON": str(subset.resolve()),
                "IAAN_AUDIO_BASE": str(audio_base.resolve()),
                "IAAN_OUTPUT_DIR": str(out),
                "IAAN_MODEL_OUTPUT_DIR": str(out / model),
                "IAAN_TASKS": ",".join(tasks),
            })
            (out / model).mkdir(parents=True, exist_ok=True)
            print(f"Running {model}: K={k}, gain={gain}", flush=True)
            subprocess.run(["bash", "-lc", env[f"IAAN_{model.upper()}_CMD"]],
                           env=env, check=True)
            for kind in ("baseline", "iaan"):
                report = out / model / f"{kind}_report.json"
                if not report.is_file():
                    raise FileNotFoundError(f"Original runner must produce {report}")
        from main_table import render
        table = render(out, models=models, tasks=tasks)
        (out / "Main_Table.md").write_text(table, encoding="utf-8")
        print(table)
        print("Saved:", out / "Main_Table.md")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
