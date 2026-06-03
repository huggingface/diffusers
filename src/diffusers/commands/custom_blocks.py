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

"""``diffusers-cli custom_blocks`` — save a custom ``ModularPipelineBlocks`` subclass.

Parses a local ``block.py``, finds a ``ModularPipelineBlocks`` subclass,
dynamically imports it, and calls ``save_pretrained`` in the current
working directory so the result can be pushed to the Hub and consumed by
``diffusers-cli inference``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

from ..utils import logging
from . import BaseDiffusersCLICommand


_EXPECTED_BASE_CLASSES = ("ModularPipelineBlocks",)


class CustomBlocksCommand(BaseDiffusersCLICommand):
    task = "custom_blocks"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "custom_blocks",
            help="Save a custom ModularPipelineBlocks subclass via save_pretrained.",
        )
        parser.add_argument(
            "--block-module-name",
            default="block.py",
            help="Module filename in which the custom block is implemented (default: block.py).",
        )
        parser.add_argument(
            "--block-class-name",
            default=None,
            help="Name of the custom block class. If None, the first ModularPipelineBlocks subclass found is used.",
        )
        parser.set_defaults(func=CustomBlocksCommand)

    def __init__(self, args: Namespace):
        self.logger = logging.get_logger("diffusers-cli/custom_blocks")
        self.block_module_name = Path(args.block_module_name)
        self.block_class_name = args.block_class_name

    def run(self) -> None:
        candidates = self._get_class_names(self.block_module_name)
        classes_found = list({cls for cls, _ in candidates})

        if not candidates:
            raise ValueError(
                f"No ModularPipelineBlocks subclass found in {self.block_module_name}. "
                "Ensure your block class inherits from `ModularPipelineBlocks` directly."
            )

        if self.block_class_name is not None:
            child_class = next((cls for cls, _ in candidates if cls == self.block_class_name), None)
            if child_class is None:
                raise ValueError(
                    f"--block-class-name {self.block_class_name!r} not found in "
                    f"{self.block_module_name}. Available: {classes_found}"
                )
        else:
            self.logger.info(
                f"Found classes: {classes_found} — using {classes_found[0]}. "
                "Re-run with --block-class-name to override."
            )
            child_class, _ = candidates[0]

        module_name = f"__dynamic__{self.block_module_name.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(self.block_module_name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        getattr(module, child_class)().save_pretrained(os.getcwd())

    def _get_class_names(self, file_path: Path) -> list[tuple[str, str]]:
        source = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            raise ValueError(f"Could not parse {file_path!r}: {e}") from e

        results: list[tuple[str, str]] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = [bname for b in node.bases if (bname := self._get_base_name(b)) is not None]
            for allowed in _EXPECTED_BASE_CLASSES:
                if allowed in base_names:
                    results.append((node.name, allowed))
        return results

    @staticmethod
    def _get_base_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            val = CustomBlocksCommand._get_base_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None
