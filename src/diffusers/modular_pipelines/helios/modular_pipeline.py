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


from ...loaders import HeliosLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HeliosModularPipeline(
    ModularPipeline,
    HeliosLoraLoaderMixin,
):
    """
    A ModularPipeline for Helios text-to-video generation.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "HeliosAutoBlocks"

    @property
    def vae_scale_factor_spatial(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = self.vae.config.scale_factor_spatial
        return vae_scale_factor

    @property
    def vae_scale_factor_temporal(self):
        vae_scale_factor = 4
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = self.vae.config.scale_factor_temporal
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        # YiYi TODO: find out default value
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


class HeliosPyramidModularPipeline(HeliosModularPipeline):
    """
    A ModularPipeline for Helios pyramid (progressive resolution) video generation.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "HeliosPyramidAutoBlocks"


class HeliosPyramidDistilledModularPipeline(HeliosModularPipeline):
    """
    A ModularPipeline for Helios distilled pyramid video generation using DMD scheduler.

    Uses guidance_scale=1.0 (no CFG) and supports is_amplify_first_chunk for the DMD scheduler.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "HeliosPyramidDistilledAutoBlocks"
