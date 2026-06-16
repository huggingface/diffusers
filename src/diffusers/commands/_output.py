# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Output formatting for ``diffusers-cli``.

Commands print through the singleton ``out`` instead of calling ``print`` directly. ``out`` picks the right format
(human, agent, or json) based on the top-level ``--format`` flag, so commands don't have to check the mode themselves.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Sequence


# Environment variables set by known AI coding agents. If any of these is set, `--format auto`
# picks AGENT mode instead of HUMAN.
_AGENT_ENV_VARS = (
    "CLAUDECODE",  # Claude Code
    "CLAUDE_CODE",  # alt spelling
    "CODEX_SANDBOX",  # Codex
    "CURSOR_AI",  # Cursor
    "AIDER_AI_CONTEXT",  # Aider
    "GH_COPILOT_AGENT",  # GitHub Copilot Agent
)


def is_agent() -> bool:
    """Return True if the CLI is being run by an AI coding agent."""
    return any(os.environ.get(v) for v in _AGENT_ENV_VARS)


class OutputFormat(str, Enum):
    AUTO = "auto"
    HUMAN = "human"
    AGENT = "agent"
    JSON = "json"


class Output:
    """Picks the print format for each method based on the active mode (human / agent / json)."""

    mode: OutputFormat

    def __init__(self) -> None:
        self.set_mode(OutputFormat.AUTO)

    def set_mode(self, mode: OutputFormat) -> None:
        """Set the active output mode. AUTO becomes AGENT or HUMAN based on is_agent()."""
        if mode == OutputFormat.AUTO:
            mode = OutputFormat.AGENT if is_agent() else OutputFormat.HUMAN
        self.mode = mode

    # ------------------------------------------------------------------ stdout

    def text(self, msg: str) -> None:
        """Print a line of text. Same in every mode."""
        print(msg)

    def dict(self, data: dict[str, Any]) -> None:
        """Print a dict as JSON. Indented for HUMAN, compact for AGENT and JSON."""
        indent = 2 if self.mode == OutputFormat.HUMAN else None
        print(json.dumps(data, indent=indent, default=str))

    def result(self, message: str, **data: Any) -> None:
        """Print a result summary.

        - HUMAN: the message line followed by `` key: value`` lines.
        - AGENT: ``key=value`` pairs separated by spaces on one line.
        - JSON: compact JSON of the data dict.
        """
        if self.mode == OutputFormat.HUMAN:
            print(message)
            for k, v in data.items():
                if v is not None:
                    print(f"  {k}: {v}")
        elif self.mode == OutputFormat.AGENT:
            parts = [f"{k}={v}" for k, v in data.items() if v is not None]
            print(" ".join(parts) if parts else message)
        elif self.mode == OutputFormat.JSON:
            print(json.dumps(data, default=str))

    def table(
        self,
        items: Sequence[dict[str, Any]],
        *,
        headers: list[str] | None = None,
    ) -> None:
        """Print a list of dicts as a table.

        - HUMAN: columns padded so each column lines up.
        - AGENT: tab-separated values, one row per line.
        - JSON: the list itself as a JSON array.

        ``headers`` defaults to the keys of the first item.
        """
        if not items:
            if self.mode in (OutputFormat.HUMAN, OutputFormat.AGENT):
                print("No results.")
            elif self.mode == OutputFormat.JSON:
                print("[]")
            return

        if headers is None:
            headers = list(items[0].keys())

        if self.mode == OutputFormat.JSON:
            print(json.dumps(list(items), default=str))
            return

        rows = [[_cell(item.get(h)) for h in headers] for item in items]
        if self.mode == OutputFormat.AGENT:
            print("\t".join(headers))
            for row in rows:
                print("\t".join(row))
            return

        # HUMAN: pad each column to its widest cell so they line up.
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        for row in rows:
            print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


# Shared instance imported by every subcommand.
out = Output()
