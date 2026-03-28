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

from ...loaders import LTX2LoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)


class LTX2ModularPipeline(
    ModularPipeline,
    LTX2LoraLoaderMixin,
):
    """
    A ModularPipeline for LTX2 video generation (T2V, I2V, Conditional/FLF2V).

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "LTX2AutoBlocks"

    @property
    def default_height(self):
        return 512

    @property
    def default_width(self):
        return 768

    @property
    def default_num_frames(self):
        return 121

    @property
    def vae_scale_factor_spatial(self):
        vae_scale_factor = 32
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = self.vae.spatial_compression_ratio
        return vae_scale_factor

    @property
    def vae_scale_factor_temporal(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = self.vae.temporal_compression_ratio
        return vae_scale_factor

    @property
    def transformer_spatial_patch_size(self):
        patch_size = 1
        if hasattr(self, "transformer") and self.transformer is not None:
            patch_size = self.transformer.config.patch_size
        return patch_size

    @property
    def transformer_temporal_patch_size(self):
        patch_size = 1
        if hasattr(self, "transformer") and self.transformer is not None:
            patch_size = self.transformer.config.patch_size_t
        return patch_size

    @property
    def requires_unconditional_embeds(self):
        requires = False
        if hasattr(self, "guider") and self.guider is not None:
            requires = self.guider._enabled and self.guider.num_conditions > 1
        return requires


class LTX2UpsampleModularPipeline(ModularPipeline):
    """
    A ModularPipeline for LTX2 latent upsampling.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "LTX2UpsampleBlocks"

    @property
    def default_height(self):
        return 512

    @property
    def default_width(self):
        return 768

    @property
    def vae_scale_factor_spatial(self):
        vae_scale_factor = 32
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = self.vae.spatial_compression_ratio
        return vae_scale_factor

    @property
    def vae_scale_factor_temporal(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = self.vae.temporal_compression_ratio
        return vae_scale_factor
