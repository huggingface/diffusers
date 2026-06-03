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

from argparse import ArgumentParser, Namespace, _SubParsersAction

from . import BaseDiffusersCLICommand
from .inference import _describe


class DescribeCommand(BaseDiffusersCLICommand):
    task = "describe"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "describe",
            help="Print the input schema for a diffusers pipeline repo. No weights downloaded.",
        )
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
