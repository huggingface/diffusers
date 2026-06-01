# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""``diffusers-cli tasks`` — list every registered agentic subcommand.

Designed so an agent can discover the surface area without parsing
``--help`` output.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction

from .. import BaseDiffusersCLICommand


AGENTIC_TASK_NAMES: tuple[str, ...] = (
    "text-to-image",
    "image-to-image",
    "inpaint",
    "text-to-video",
    "image-to-video",
    "text-to-audio",
    "modular",
)


def register(subparsers: _SubParsersAction) -> None:
    ListTasksCommand.register_subcommand(subparsers, subparsers)


def list_agentic_tasks(subparsers: _SubParsersAction) -> list[dict]:
    """Return ``[{name, description}, ...]`` for every registered agentic task.

    Reads metadata directly from the live argparse subparsers so the list
    can never drift from the actual commands.
    """
    choices = getattr(subparsers, "choices", {}) or {}
    actions = [a for a in getattr(subparsers, "_choices_actions", [])]
    descriptions = {a.dest: a.help for a in actions}

    out: list[dict] = []
    for name in AGENTIC_TASK_NAMES:
        if name not in choices:
            continue
        out.append({"name": name, "description": descriptions.get(name, "")})
    return out


class ListTasksCommand(BaseDiffusersCLICommand):
    task = "tasks"

    # The live subparsers object is captured at registration time so ``run``
    # can introspect it without needing access to ``main``'s locals.
    _root_subparsers: _SubParsersAction | None = None

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction, root_subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "tasks",
            help="List every registered agentic task with a one-line description.",
        )
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
        parser.set_defaults(func=ListTasksCommand)
        ListTasksCommand._root_subparsers = root_subparsers

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        tasks = list_agentic_tasks(self._root_subparsers) if self._root_subparsers else []
        if self.args.json:
            json.dump({"tasks": tasks}, sys.stdout)
            sys.stdout.write("\n")
            return
        width = max((len(t["name"]) for t in tasks), default=0)
        for entry in tasks:
            print(f"{entry['name']:<{width}}  {entry['description'] or ''}")
