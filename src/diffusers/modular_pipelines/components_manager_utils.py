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

"""Display helpers for `ComponentsManager`: human-readable sizes, text tables, per-layer value summaries."""

from __future__ import annotations

from typing import Any


def format_size(num_bytes: int | None) -> str:
    """Bytes as a human-readable size, so that both a 20GB transformer and a KB-sized test model read sensibly."""
    if num_bytes is None:
        return "-"
    for unit in ("B", "KB", "MB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} GB"


def format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """
    Align `headers` and `rows` into text-table lines (header first), each column as wide as its longest cell and
    cells joined by " | ". The last column is not padded, so a long final cell (a reason, a load id) runs free
    without stretching the table. Separator lines are the caller's to add — `len(lines[0])` is the table width.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row[:-1]):
            widths[index] = max(widths[index], len(cell))

    def line(cells: list[str]) -> str:
        return " | ".join([cell.ljust(width) for cell, width in zip(cells[:-1], widths)] + [cells[-1]])

    return [line(headers), *(line(row) for row in rows)]


def summarize_dict_by_value_and_parts(d: dict[str, Any]) -> dict[str, Any]:
    """
    Summarize a dict with dot-separated keys by grouping keys that share a value under their longest common prefix.

    For example IP-Adapter scales per attention processor: {
        'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor': [0.6],
        'down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor': [0.6],
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor': [0.3],
    } becomes {'down_blocks.1.attentions.1.transformer_blocks': [0.6], 'up_blocks': [0.3]}.
    """
    value_to_keys: dict[Any, list[str]] = {}
    for key, value in d.items():
        hashable = tuple(value) if isinstance(value, list) else value
        value_to_keys.setdefault(hashable, []).append(key)

    summary = {}
    for keys in value_to_keys.values():
        split_keys = [key.split(".") for key in keys]
        common_parts = []
        for parts in zip(*split_keys):
            if len(set(parts)) != 1:
                break
            common_parts.append(parts[0])
        value = d[keys[0]]
        summary[".".join(common_parts)] = list(value) if isinstance(value, list) else value
    return summary
