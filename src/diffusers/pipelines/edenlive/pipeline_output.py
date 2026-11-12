# Copyright 2026 The VAST-AI Team. All rights reserved.
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

from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class EdenLivePipelineOutput(BaseOutput):
    """Output of [`EdenLivePipeline`].

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]`):
            Generated videos selected by `output_type`. `"np"` returns shape `(B, T, H, W, 3)` and `"pt"` returns `(B,
            T, 3, H, W)`, both with floating-point RGB values in [0, 1]. `"pil"` returns one list of RGB images per
            video. `"latent"` bypasses decoding and returns normalized diffusion latents with shape `(B, C, T_latent,
            H_latent, W_latent)`; these values are not bounded RGB pixels. `B` is the video batch size and `T` is the
            decoded frame count. The latent channel count `C` is 48 for 5B and 16 for 14B.
    """

    frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
