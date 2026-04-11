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


import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import LTXVideoLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)


class LTXVideoPachifier(ConfigMixin):
    """
    A class to pack and unpack latents for LTX Video.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(self, patch_size: int = 1, patch_size_t: int = 1):
        super().__init__()

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_frames, height, width = latents.shape
        patch_size = self.config.patch_size
        patch_size_t = self.config.patch_size_t
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    def unpack_latents(self, latents: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
        batch_size = latents.size(0)
        patch_size = self.config.patch_size
        patch_size_t = self.config.patch_size_t
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents


class LTXModularPipeline(
    ModularPipeline,
    LTXVideoLoraLoaderMixin,
):
    """
    A ModularPipeline for LTX Video.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "LTXAutoBlocks"

    @property
    def vae_spatial_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.spatial_compression_ratio
        return 32

    @property
    def vae_temporal_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.temporal_compression_ratio
        return 8

    @property
    def requires_unconditional_embeds(self):
        if hasattr(self, "guider") and self.guider is not None:
            return self.guider._enabled and self.guider.num_conditions > 1
        return False
