#!/usr/bin/env bash
# One-command orchestrator for *original greedy-generation* IAAN results.
# Does not silently use letter-logit Steer-CD or 300/700-item subsets.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f paper_reproduction/main_table.env ]]; then
  # Local, trusted shell configuration containing your original model commands.
  source paper_reproduction/main_table.env
fi
OUT="${IAAN_OUTPUT_DIR:-$ROOT/outputs/main_table}"
DATA="${IAAN_VOXPARADOX_JSON:-}"
AUDIO="${IAAN_AUDIO_BASE:-}"
if [[ -z "$DATA" || ! -f "$DATA" ]]; then
  echo "ERROR: IAAN_VOXPARADOX_JSON must point to the complete 2,000-item VoxParadox JSON." >&2
  exit 2
fi
if [[ -z "$AUDIO" || ! -d "$AUDIO" ]]; then
  echo "ERROR: IAAN_AUDIO_BASE must point to the directory with audio files." >&2
  exit 2
fi
# Known repository scripts are not the paper's original official runners.
# Fail fast rather than silently presenting a Steer-CD result as SLT Table I.
for model in af3 qwen kimi; do
  key="IAAN_${model^^}_CMD"
  if [[ -z "${!key:-}" ]]; then
    echo "ERROR: Missing $key. The original *greedy-generation* script for $model" >&2
    echo "is not present in this GitHub repository. See paper_reproduction/main_table.env.example." >&2
    exit 2
  fi
done
mkdir -p "$OUT"
export IAAN_OUTPUT_DIR="$OUT" IAAN_VOXPARADOX_JSON="$DATA" IAAN_AUDIO_BASE="$AUDIO"
for model in af3 qwen kimi; do
  key="IAAN_${model^^}_CMD"
  echo "=== Running original IAAN $model generation/evaluation ==="
  bash -lc "${!key}"
  for kind in baseline iaan; do
    report="$OUT/$model/${kind}_report.json"
    if [[ ! -f "$report" ]]; then
      echo "ERROR: $key must create $report (official data/eval.py JSON format)." >&2
      exit 3
    fi
  done
done
python paper_reproduction/main_table.py --results "$OUT" --out "$OUT/Main_Table.md"
echo "Complete: $OUT/Main_Table.md"
