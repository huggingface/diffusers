# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""Output containers for the Kandinsky 6 Diffusers pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...utils import BaseOutput


@dataclass
class Kandinsky6TI2VAPipelineOutput(BaseOutput):
    """Output of the K6 video-and-audio pipeline."""

    frames: Any
    audio: list[np.ndarray] | None


@dataclass
class Kandinsky6SRPipelineOutput(BaseOutput):
    """Output of the K6 super-resolution pipeline."""

    frames: Any


__all__ = [
    "Kandinsky6SRPipelineOutput",
    "Kandinsky6TI2VAPipelineOutput",
]
