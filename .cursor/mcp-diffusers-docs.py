#!/usr/bin/env python3
"""Launch diffusers-docs MCP from any cwd.

Cloud Agent stdio cannot set `cwd` and does not expand `${workspaceFolder}`.
This launcher walks cwd, git root, /workspace, and its own path until it finds
`tools/docs_mcp_server.py`, then execs it. Desktop and Cloud then share one
command: `python3 -u .cursor/mcp-diffusers-docs.py`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _candidates():
    out: list[Path] = []
    here = Path(__file__).resolve().parent
    out.append(here.parent)  # repo root when this file lives in .cursor/
    out.append(here.parent / "ramp-kit")
    cwd = Path.cwd().resolve()
    out.extend([cwd, cwd / "ramp-kit", *cwd.parents])
    for key in ("CURSOR_PROJECT_DIR", "CURSOR_WORKSPACE", "WORKSPACE"):
        val = os.environ.get(key)
        if val:
            out.append(Path(val))
    out.append(Path("/workspace"))
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if top:
            out.append(Path(top))
    except (OSError, subprocess.CalledProcessError):
        pass
    seen: set[Path] = set()
    for raw in out:
        try:
            path = raw.resolve()
        except OSError:
            continue
        if path in seen:
            continue
        seen.add(path)
        yield path


def main(argv: list[str] | None = None) -> int:
    extra = [a for a in (argv if argv is not None else sys.argv[1:]) if a != "--serve"]
    for root in _candidates():
        script = root / "tools" / "docs_mcp_server.py"
        if script.is_file():
            os.chdir(root)
            sys.argv = [str(script), "--serve", *extra]
            # runpy would work; exec keeps stdin/stdout as the MCP pipes.
            os.execv(sys.executable, [sys.executable, "-u", str(script), "--serve", *extra])
    sys.stderr.write(
        "diffusers-docs MCP: could not find tools/docs_mcp_server.py "
        f"(cwd={Path.cwd()}). From the repo root run: "
        "python3 -u .cursor/mcp-diffusers-docs.py\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
