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
"""Dual-audience output sink for ``diffusers-cli``.

Every subcommand routes user-visible output through the singleton ``out``. The mode is one of ``human`` (default for
terminals), ``agent`` (auto-selected when an AI coding agent is detected), or ``json`` (machine-parseable). The set of
methods on ``out`` covers the shapes our commands actually produce — free-form text, key/value results, structured
dicts, and tabular schemas — so leaf commands never branch on ``args.json`` themselves.
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from typing import Any, Sequence


# Environment variables set by known AI coding agents. Presence of any one triggers AGENT mode
# under `--format auto`.
_AGENT_ENV_VARS = (
    "CLAUDECODE",  # Claude Code
    "CLAUDE_CODE",  # alt spelling
    "CODEX_SANDBOX",  # Codex
    "CURSOR_AI",  # Cursor
    "AIDER_AI_CONTEXT",  # Aider
    "GH_COPILOT_AGENT",  # GitHub Copilot Agent
)


def is_agent() -> bool:
    """Return True if the process appears to be invoked by an AI coding agent."""
    return any(os.environ.get(v) for v in _AGENT_ENV_VARS)


class OutputFormat(str, Enum):
    AUTO = "auto"
    HUMAN = "human"
    AGENT = "agent"
    JSON = "json"


class Output:
    """Singleton output sink. Resolve mode once at startup, then call ``out.<method>``."""

    mode: OutputFormat

    def __init__(self) -> None:
        self.set_mode(OutputFormat.AUTO)

    def set_mode(self, mode: OutputFormat) -> None:
        """Set the active output mode. AUTO resolves to AGENT or HUMAN via ``is_agent()``."""
        if mode == OutputFormat.AUTO:
            mode = OutputFormat.AGENT if is_agent() else OutputFormat.HUMAN
        self.mode = mode

    # ------------------------------------------------------------------ stdout

    def text(self, msg: str) -> None:
        """Free-form line. Printed plain in every mode."""
        print(msg)

    def dict(self, data: dict[str, Any]) -> None:
        """Structured object — JSON in every mode (indented for HUMAN, compact otherwise).

        Use for payloads that don't decompose cleanly into key/value pairs (e.g. describe schemas).
        """
        indent = 2 if self.mode == OutputFormat.HUMAN else None
        print(json.dumps(data, indent=indent, default=str))

    def result(self, message: str, **data: Any) -> None:
        """Success summary.

        - HUMAN: ``message`` followed by `` key: value`` lines.
        - AGENT: ``key=value`` pairs space-separated on one line (TSV-ish, parser-friendly).
        - JSON: compact JSON of ``data``.
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
        """Tabular data — HUMAN gets padded columns, AGENT gets TSV, JSON gets the list.

        Headers default to the keys of the first item.
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

        # HUMAN: pad each column to its widest cell for readable alignment.
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        for row in rows:
            print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))

    # ------------------------------------------------------------------ stderr

    def hint(self, message: str) -> None:
        """Next-step suggestion. Always goes to stderr so it never pollutes parseable stdout."""
        print(f"Hint: {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        """Non-fatal warning — stderr, every mode."""
        print(f"Warning: {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Error — stderr, every mode."""
        print(f"Error: {message}", file=sys.stderr)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


# Module-level singleton imported by every subcommand.
out = Output()
