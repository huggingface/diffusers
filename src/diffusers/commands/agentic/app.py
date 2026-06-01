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

"""Single integration point for the agentic CLI.

Removing the call to ``register_agentic_commands`` from
``diffusers_cli.py`` disables the entire surface with no side effects.
"""

from __future__ import annotations

from argparse import _SubParsersAction

from . import audio as audio_commands
from . import image as image_commands
from . import modular as modular_commands
from . import tasks as tasks_commands
from . import video as video_commands


def register_agentic_commands(subparsers: _SubParsersAction) -> None:
    """Register every agentic subcommand on the top-level ``diffusers-cli`` parser."""
    image_commands.register(subparsers)
    video_commands.register(subparsers)
    audio_commands.register(subparsers)
    modular_commands.register(subparsers)
    tasks_commands.register(subparsers)
