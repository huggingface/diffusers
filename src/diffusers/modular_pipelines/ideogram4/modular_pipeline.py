# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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

from ...loaders import Ideogram4LoraLoaderMixin
from ..modular_pipeline import ModularPipeline


class Ideogram4ModularPipeline(ModularPipeline, Ideogram4LoraLoaderMixin):
    """
    A ModularPipeline for Ideogram4.

    > [!WARNING] > This is an experimental feature!
    """

    default_blocks_name = "Ideogram4AutoBlocks"

    # Ideogram4 patchifies the VAE output by a factor of 2 before feeding the transformer.
    @property
    def patch_size(self):
        return 2

    @property
    def default_height(self):
        return 2048

    @property
    def default_width(self):
        return 2048

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if getattr(self, "vae", None) is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor
