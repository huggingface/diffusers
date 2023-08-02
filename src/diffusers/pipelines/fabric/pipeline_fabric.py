# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .cross_attention import AttnProcessor
from .embeddings import TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AttentionBasedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def generate(
        self, 
        prompt: Union[str, List[str]] = "a photo of an astronaut riding a horse on mars",
        negative_prompt: Union[str, List[str]] = "",
        liked: List[Image.Image] = [],
        disliked: List[Image.Image] = [],
        seed: int = 42,
        n_images: int = 1,
        guidance_scale: float = 8.0,
        denoising_steps: int = 20,
        feedback_start: float = 0.33,
        feedback_end: float = 0.66,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        neg_scale: float = 0.5,
        pos_bottleneck_scale: float = 1.0,
        neg_bottleneck_scale: float = 1.0,
    )
        pass




