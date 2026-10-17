# Copyright 2026 The Hugging Face Team. All rights reserved.
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
import torch
from PIL import Image

from ...utils import BaseOutput


@dataclass
class TripoSplatPipelineOutput(BaseOutput):
    """Generated Gaussian parameters and the conditioning images.

    Args:
        gaussians (`torch.Tensor`, `numpy.ndarray`, or `list`, *optional*):
            Parameters shaped `(batch, num_gaussians, 14)`, or one such batch per requested density. `None` for latent
            output. Columns contain xyz position, degree-zero SH color, scale, wxyz rotation, and opacity.
        latents (`torch.Tensor`, *optional*):
            Denoised Gaussian latents.
        camera_latents (`torch.Tensor`, *optional*):
            Denoised camera latents.
        preprocessed_images (`list[PIL.Image.Image]`, *optional*):
            Cropped RGB images composited onto black canvases.
    """

    gaussians: torch.Tensor | np.ndarray | list[torch.Tensor] | list[np.ndarray] | None
    latents: torch.Tensor | None = None
    camera_latents: torch.Tensor | None = None
    preprocessed_images: list[Image.Image] | None = None
