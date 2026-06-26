# Copyright 2025 The Google and HuggingFace Teams. All rights reserved.
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

import torch

from ...utils import BaseOutput


@dataclass
class DiffusionGemmaPipelineOutput(BaseOutput):
    """
    Output class for DiffusionGemma block-diffusion generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, gen_length)`):
            The generated token IDs (the prompt is stripped off).
        texts (`list[str]`, *optional*):
            The decoded text, one string per sequence. Only set for `output_type="text"`.
    """

    sequences: torch.LongTensor
    texts: list[str] | None = None
