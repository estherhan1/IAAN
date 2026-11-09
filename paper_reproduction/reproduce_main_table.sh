#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/paper_reproduction/main_table.env" ]]; then
  source "$ROOT/paper_reproduction/main_table.env"
fi
exec python "$ROOT/paper_reproduction/run_main_table.py" "$@"
