#!/usr/bin/env python
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

from argparse import ArgumentParser

from ._output import OutputFormat, out
from .custom_blocks import CustomBlocksCommand
from .describe import DescribeCommand
from .env import EnvironmentCommand
from .fp16_safetensors import FP16SafetensorsCommand
from .generate import GenerateCommand


def main():
    parser = ArgumentParser(
        prog="diffusers-cli",
        usage="\n  diffusers-cli [--format <fmt>] <command> [options]",
    )
    parser._optionals.title = "Options"
    parser.add_argument(
        "--format",
        choices=[m.value for m in OutputFormat],
        default=OutputFormat.AUTO.value,
        help=(
            "Output format. 'auto' (default) picks 'agent' when an AI coding agent is detected "
            "(via CLAUDECODE/CURSOR_AI/AIDER_AI_CONTEXT/... env vars) and 'human' otherwise. "
            "Must appear before the subcommand."
        ),
    )
    commands_parser = parser.add_subparsers(title="Commands", metavar="<command>")

    # Register commands
    EnvironmentCommand.register_subcommand(commands_parser)
    FP16SafetensorsCommand.register_subcommand(commands_parser)
    CustomBlocksCommand.register_subcommand(commands_parser)
    GenerateCommand.register_subcommand(commands_parser)
    DescribeCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    out.set_mode(OutputFormat(args.format))

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
