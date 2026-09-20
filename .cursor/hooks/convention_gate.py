#!/usr/bin/env python3
"""afterFileEdit hook — run the same convention gate CI uses.

Cursor sends JSON on stdin:
  {"file_path": "<absolute path>", "edits": [...]}

This is a notification (the write already happened). Findings are printed
to stderr so the agent can fix them immediately. We always exit 0 so a
red gate never crashes the editor loop; blocking findings still show up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_kit_tools = None
for _cand in (ROOT / "ramp-kit" / "tools", ROOT / "tools"):
    if (_cand / "convention_check.py").is_file():
        _kit_tools = _cand
        break
sys.path.insert(0, str(_kit_tools or (ROOT / "tools")))

from convention_check import check_file, load_rules, render_human  # noqa: E402

# Don't fire the gate on the kit's own machinery or the teaching fixture
# (the fixture is scanned explicitly by `make demo`).
_SKIP_SUBSTR = (
    "/tools/",
    "/.cursor/",
    "/_templates/",
    "/templates/",
    "/__pycache__/",
    "/ramp-kit/examples/candidate_scheduler/",
)


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    file_path = payload.get("file_path") or ""
    path = Path(file_path)
    if path.suffix != ".py" or not path.is_file():
        return 0
    posix = path.as_posix()
    if any(s in posix for s in _SKIP_SUBSTR):
        return 0
    findings = check_file(path, load_rules())
    sys.stderr.write(render_human(findings, 1))
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
