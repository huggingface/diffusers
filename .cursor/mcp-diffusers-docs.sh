#!/usr/bin/env bash
# Thin wrapper around the Python launcher (Cloud dashboard can use either).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 -u "$DIR/mcp-diffusers-docs.py" --serve "$@"
