#!/usr/bin/env bash
# Cursor afterFileEdit hook. Project hooks run from the repo root.
# Reads the afterFileEdit JSON payload on stdin and runs the same
# convention gate as CI (`tools/convention_check.py`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT/.cursor/hooks/convention_gate.py"
