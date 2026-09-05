# Copyright 2026 HuggingFace Inc. All rights reserved.
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
class RAEDiTPipelineOutput(BaseOutput):
    """
    Output class for RAE DiT image generation pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`)
            Denoised images as PIL images, a NumPy array of shape `(batch_size, height, width, num_channels)`, or a
            PyTorch tensor of shape `(batch_size, num_channels, height, width)`. Torch tensors may also represent
            latent outputs when `output_type="latent"`.
    """

    images: list[PIL.Image.Image] | np.ndarray | torch.Tensor
