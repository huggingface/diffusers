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

from ...loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)


class StableDiffusion3ModularPipeline(ModularPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    """
    A ModularPipeline for Stable Diffusion 3.

    >[!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "StableDiffusion3AutoBlocks"

    @property
    def default_height(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_width(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_sample_size(self):
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.sample_size
        return 128

    @property
    def patch_size(self):
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.patch_size
        return 2

    @property
    def tokenizer_max_length(self):
        if getattr(self, "tokenizer", None) is not None:
            return self.tokenizer.model_max_length
        return 77

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if getattr(self, "vae", None) is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.in_channels
        return 16
