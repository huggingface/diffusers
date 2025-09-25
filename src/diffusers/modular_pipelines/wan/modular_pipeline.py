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


from ...loaders import WanLoraLoaderMixin
from ...pipelines.pipeline_utils import StableDiffusionMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanModularPipeline(
    ModularPipeline,
    StableDiffusionMixin,
    WanLoraLoaderMixin,
):
    """
    A ModularPipeline for Wan.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "WanAutoBlocks"

    @property
    def default_height(self):
        return self.default_sample_height * self.vae_scale_factor_spatial

    @property
    def default_width(self):
        return self.default_sample_width * self.vae_scale_factor_spatial

    @property
    def default_num_frames(self):
        return (self.default_sample_num_frames - 1) * self.vae_scale_factor_temporal + 1

    @property
    def default_sample_height(self):
        return 60

    @property
    def default_sample_width(self):
        return 104

    @property
    def default_sample_num_frames(self):
        return 21

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
    def num_channels_transformer(self):
        num_channels_transformer = 16
        if hasattr(self, "transformer") and self.transformer is not None:
            num_channels_transformer = self.transformer.config.in_channels
        return num_channels_transformer

    @property
    def num_channels_latents(self):
        num_channels_latents = 16
        if hasattr(self, "vae") and self.vae is not None:
            num_channels_latents = self.vae.config.z_dim
        return num_channels_latents
