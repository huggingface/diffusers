# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import List, Union

import numpy as np
import torch

from ...utils import BaseOutput


@dataclass
class MagiPipelineOutput(BaseOutput):
    """
    Output class for MAGI-1 pipeline.

    Args:
        frames (`torch.Tensor` or `np.ndarray`):
            List of denoised frames from the diffusion process, as a NumPy array of shape `(batch_size, num_frames, height, width, num_channels)` or a PyTorch tensor of shape `(batch_size, num_channels, num_frames, height, width)`.
    """

    frames: Union[torch.Tensor, np.ndarray, List[List[np.ndarray]]]
