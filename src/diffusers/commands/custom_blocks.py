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

"""
Usage example:
    TODO
"""

import ast
import importlib.util
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from ..utils import logging
from . import BaseDiffusersCLICommand


EXPECTED_PARENT_CLASSES = ["ModularPipelineBlocks"]
CONFIG = "config.json"


def conversion_command_factory(args: Namespace):
    return CustomBlocksCommand(args.block_module_name, args.block_class_name)


class CustomBlocksCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        conversion_parser = parser.add_parser("custom_blocks")
        conversion_parser.add_argument(
            "--block_module_name",
            type=str,
            default="block.py",
            help="Module filename in which the custom block will be implemented.",
        )
        conversion_parser.add_argument(
            "--block_class_name",
            type=str,
            default=None,
            help="Name of the custom block. If provided None, we will try to infer it.",
        )
        conversion_parser.set_defaults(func=conversion_command_factory)

    def __init__(self, block_module_name: str = "block.py", block_class_name: str = None):
        self.logger = logging.get_logger("diffusers-cli/custom_blocks")
        self.block_module_name = Path(block_module_name)
        self.block_class_name = block_class_name

    def run(self):
        # determine the block to be saved.
        out = self._get_class_names(self.block_module_name)
        classes_found = list({cls for cls, _ in out})

        if self.block_class_name is not None:
            child_class, parent_class = self._choose_block(out, self.block_class_name)
            if child_class is None and parent_class is None:
                raise ValueError(
                    "`block_class_name` could not be retrieved. Available classes from "
                    f"{self.block_module_name}:\n{classes_found}"
                )
        else:
            self.logger.info(
                f"Found classes: {classes_found} will be using {classes_found[0]}. "
                "If this needs to be changed, re-run the command specifying `block_class_name`."
            )
            child_class, parent_class = out[0][0], out[0][1]

        # dynamically get the custom block and initialize it to call `save_pretrained` in the current directory.
        # the user is responsible for running it, so I guess that is safe?
        module_name = f"__dynamic__{self.block_module_name.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(self.block_module_name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        getattr(module, child_class)().save_pretrained(os.getcwd())

        # or, we could create it manually.
        # automap = self._create_automap(parent_class=parent_class, child_class=child_class)
        # with open(CONFIG, "w") as f:
        #     json.dump(automap, f)
        with open("requirements.txt", "w") as f:
            f.write("")

    def _choose_block(self, candidates, chosen=None):
        for cls, base in candidates:
            if cls == chosen:
                return cls, base
        return None, None

    def _get_class_names(self, file_path):
        source = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            raise ValueError(f"Could not parse {file_path!r}: {e}") from e

        results: list[tuple[str, str]] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            # extract all base names for this class
            base_names = [bname for b in node.bases if (bname := self._get_base_name(b)) is not None]

            # for each allowed base that appears in the class's bases, emit a tuple
            for allowed in EXPECTED_PARENT_CLASSES:
                if allowed in base_names:
                    results.append((node.name, allowed))

        return results

    def _get_base_name(self, node: ast.expr):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_base_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None

    def _create_automap(self, parent_class, child_class):
        module = str(self.block_module_name).replace(".py", "").rsplit(".", 1)[-1]
        auto_map = {f"{parent_class}": f"{module}.{child_class}"}
        return {"auto_map": auto_map}
