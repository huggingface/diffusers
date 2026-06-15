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

"""``diffusers-cli describe`` — print the input schema for any pipeline repo.

Tries ``DiffusionPipeline.config_name`` first (so standard repos get their ``__call__`` signature introspected); falls
back to ``ModularPipelineBlocks.from_pretrained`` for modular repos. No weights are downloaded — only the small index
file (and any custom block code if ``--trust-remote-code`` is set).
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import Any, Optional

from . import BaseDiffusersCLICommand
from ._common import try_fetch_config
from ._output import OutputFormat, out


def _describe(args: Namespace) -> None:
    """Print the pipeline's input schema.

    Tries ``DiffusionPipeline.config_name`` (= ``model_index.json``) first; if present, introspects the declared
    pipeline class's ``__call__`` signature. Otherwise falls back to ``ModularPipelineBlocks.from_pretrained`` and
    reads the block-declared ``inputs``. No weights downloaded either way.
    """
    import inspect

    import diffusers

    model_index = try_fetch_config(args, diffusers.DiffusionPipeline.config_name)
    if model_index is not None:
        with open(model_index) as f:
            index = json.load(f)
        class_name = index.get("_class_name")
        if class_name is None:
            raise SystemExit(
                f"{diffusers.DiffusionPipeline.config_name} for {args.model!r} has no `_class_name` field."
            )
        pipeline_cls = getattr(diffusers, class_name, None)
        if pipeline_cls is None:
            raise SystemExit(
                f"Pipeline class {class_name!r} declared in {diffusers.DiffusionPipeline.config_name} "
                "is not exported by the installed diffusers."
            )

        sig = inspect.signature(pipeline_cls.__call__)
        descriptions = _parse_docstring_args(pipeline_cls.__call__.__doc__) if args.verbose else {}
        schema: list[dict[str, Any]] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            has_default = param.default is not inspect.Parameter.empty
            schema.append(
                {
                    "name": name,
                    "type_hint": str(param.annotation) if param.annotation is not inspect.Parameter.empty else None,
                    "default": param.default if has_default else None,
                    "required": not has_default,
                    "description": descriptions.get(name, ""),
                }
            )
    else:
        kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
        if args.revision:
            kwargs["revision"] = args.revision
        if args.token:
            kwargs["token"] = args.token
        try:
            blocks = diffusers.ModularPipelineBlocks.from_pretrained(args.model, **kwargs)
        except Exception as e:
            raise SystemExit(
                f"Could not describe {args.model!r}: no {diffusers.DiffusionPipeline.config_name} and "
                f"loading as a modular pipeline failed ({type(e).__name__}: {e}). "
                "Is this a diffusers pipeline repo? Pass --trust-remote-code if it ships custom block code."
            ) from e

        class_name = type(blocks).__name__
        schema = [
            {
                "name": p.name,
                "type_hint": str(p.type_hint) if p.type_hint is not None else None,
                "default": p.default,
                "required": p.required,
                "description": p.description,
            }
            for p in blocks.inputs
        ]

    if args.json:
        out.set_mode(OutputFormat.JSON)

    if out.mode in (OutputFormat.JSON, OutputFormat.AGENT):
        # Agents get the structured schema (full payload for JSON, the inputs table for AGENT).
        if out.mode == OutputFormat.JSON:
            out.dict({"task": "describe", "model": args.model, "pipeline_class": class_name, "inputs": schema})
        else:
            out.table(schema, headers=["name", "required", "type_hint", "default", "description"])
        return

    _print_schema(class_name, args.model, schema)


def _parse_docstring_args(docstring: Optional[str]) -> dict[str, str]:
    """Extract per-argument descriptions from a Google-style ``Args:`` block.

    Returns a ``{name: description}`` mapping. Best-effort — unrecognised formats just yield an empty dict rather than
    raising.
    """
    if not docstring:
        return {}

    import re

    lines = docstring.expandtabs().splitlines()
    start = None
    section_indent = 0
    for i, line in enumerate(lines):
        if line.strip() in ("Args:", "Arguments:", "Parameters:"):
            start = i + 1
            section_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return {}

    descriptions: dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []
    arg_indent: Optional[int] = None
    name_pattern = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:?\s*(.*)$")

    def _flush() -> None:
        if current_name and current_lines:
            descriptions[current_name] = " ".join(s.strip() for s in current_lines).strip()

    for line in lines[start:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        # A new top-level section ends the Args block.
        if indent <= section_indent and line.strip().endswith(":"):
            break
        if arg_indent is None:
            arg_indent = indent
        if indent == arg_indent:
            _flush()
            current_lines = []
            match = name_pattern.match(line.strip())
            if match:
                current_name = match.group(1)
                tail = match.group(2).strip()
                if tail:
                    current_lines.append(tail)
            else:
                current_name = None
        elif current_name is not None and indent > arg_indent:
            current_lines.append(line.strip())
    _flush()
    return descriptions


def _print_schema(class_name: str, model: str, schema: list[dict[str, Any]]) -> None:
    print(f"{class_name} ({model}) inputs:")
    for entry in schema:
        tag = "required" if entry["required"] else f"optional, default={entry['default']!r}"
        print(f"  {entry['name']}  ({tag})")
        if entry["type_hint"]:
            print(f"    type: {entry['type_hint']}")
        if entry["description"]:
            print(f"    desc: {entry['description']}")


class DescribeCommand(BaseDiffusersCLICommand):
    task = "describe"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "describe",
            help="Print the input schema for a diffusers pipeline repo. No weights downloaded.",
            usage="\n  diffusers-cli describe [options]",
        )
        parser._optionals.title = "Options"
        parser.add_argument(
            "--model",
            "-m",
            required=True,
            help="Model id on the Hugging Face Hub or local path.",
        )
        parser.add_argument(
            "--revision",
            default=None,
            help="Model revision (branch, tag, or commit SHA).",
        )
        parser.add_argument(
            "--token",
            default=None,
            help="Hugging Face token for gated/private models.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Allow custom code from the Hub (required for modular pipelines that ship block code).",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help=(
                "Also include per-argument descriptions from the pipeline's __call__ docstring. "
                "Modular pipelines always include block-declared descriptions; --verbose populates "
                "the equivalent field for standard pipelines by parsing the Google-style Args: block."
            ),
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit a machine-readable JSON summary on stdout.",
        )
        parser.set_defaults(func=DescribeCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        _describe(self.args)
