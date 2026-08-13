# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ...loaders import WanLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAnimate2ModularPipeline(ModularPipeline, WanLoraLoaderMixin):
    """
    A ModularPipeline for Wan-Animate-2 character animation.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "WanAnimate2Blocks"

    @property
    def vae_scale_factor_spatial(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        return vae_scale_factor

    @property
    def vae_scale_factor_temporal(self):
        vae_scale_factor = 4
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** sum(self.vae.temperal_downsample)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 16
        if hasattr(self, "vae") and self.vae is not None:
            num_channels_latents = self.vae.config.z_dim
        return num_channels_latents

    @property
    def requires_unconditional_embeds(self):
        requires_unconditional_embeds = False

        if hasattr(self, "guider") and self.guider is not None:
            requires_unconditional_embeds = self.guider._enabled and self.guider.num_conditions > 1

        return requires_unconditional_embeds


class WanAnimate2DistilledModularPipeline(WanAnimate2ModularPipeline):
    """
    A ModularPipeline for the distilled Wan-Animate-2 model, which samples in few steps without classifier-free
    guidance.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "WanAnimate2DistilledBlocks"
