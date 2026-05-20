# Copyright 2025 Ant Group and The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class UniLLaDaPipelineOutput(BaseOutput):
    """
    Output class for UniLLaDA pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`, *optional*):
            List of denoised PIL images or numpy array. Present for text-to-image and image editing tasks.
        text (`str` or `list[str]`, *optional*):
            Generated text response. Present for image understanding tasks.
    """

    images: list[PIL.Image.Image] | np.ndarray | None = None
    text: str | list[str] | None = None
