# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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


from ...loaders import ZImageLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ZImageModularPipeline(
    ModularPipeline,
    ZImageLoraLoaderMixin,
):
    """
    A ModularPipeline for Z-Image.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "ZImageAutoBlocks"

    @property
    def default_height(self):
        return 1024

    @property
    def default_width(self):
        return 1024

    @property
    def vae_scale_factor_spatial(self):
        vae_scale_factor_spatial = 16
        if hasattr(self, "image_processor") and self.image_processor is not None:
            vae_scale_factor_spatial = self.image_processor.config.vae_scale_factor
        return vae_scale_factor_spatial

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 16
        if hasattr(self, "transformer") and self.transformer is not None:
            num_channels_latents = self.transformer.config.in_channels
        return num_channels_latents

    @property
    def requires_unconditional_embeds(self):
        requires_unconditional_embeds = False

        if hasattr(self, "guider") and self.guider is not None:
            requires_unconditional_embeds = self.guider._enabled and self.guider.num_conditions > 1

        return requires_unconditional_embeds
