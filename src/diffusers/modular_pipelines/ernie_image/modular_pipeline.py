# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ErnieImagePachifier(ConfigMixin):
    """
    A class to pack and unpack latents for ErnieImage.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(self, patch_size: int = 2):
        super().__init__()

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = latents.shape
        patch_size = self.config.patch_size

        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Latent height and width must be divisible by {patch_size}, but got {height} and {width}"
            )

        latents = latents.view(
            batch_size, num_channels, height // patch_size, patch_size, width // patch_size, patch_size
        )
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(
            batch_size, num_channels * patch_size * patch_size, height // patch_size, width // patch_size
        )

    def unpack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = latents.shape
        patch_size = self.config.patch_size

        latents = latents.reshape(
            batch_size, num_channels // (patch_size * patch_size), patch_size, patch_size, height, width
        )
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(
            batch_size, num_channels // (patch_size * patch_size), height * patch_size, width * patch_size
        )


class ErnieImageModularPipeline(ModularPipeline):
    """
    A ModularPipeline for ErnieImage.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "ErnieImageAutoBlocks"

    @property
    def default_height(self):
        return 1024

    @property
    def default_width(self):
        return 1024

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 16
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** len(self.vae.config.block_out_channels)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 128
        if hasattr(self, "transformer") and self.transformer is not None:
            num_channels_latents = self.transformer.config.in_channels
        return num_channels_latents

    @property
    def text_in_dim(self):
        text_in_dim = 3584
        if hasattr(self, "transformer") and self.transformer is not None:
            text_in_dim = self.transformer.config.text_in_dim
        return text_in_dim

    @property
    def requires_unconditional_embeds(self):
        requires_unconditional_embeds = False
        if hasattr(self, "guider") and self.guider is not None:
            requires_unconditional_embeds = self.guider._enabled and self.guider.num_conditions > 1
        return requires_unconditional_embeds
