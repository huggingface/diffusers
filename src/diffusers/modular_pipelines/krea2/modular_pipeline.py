# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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

from ...loaders import Krea2LoraLoaderMixin
from ..modular_pipeline import ModularPipeline


class Krea2ModularPipeline(ModularPipeline, Krea2LoraLoaderMixin):
    """
    A ModularPipeline for Krea 2.

    > [!WARNING] > This is an experimental feature!
    """

    default_blocks_name = "Krea2AutoBlocks"

    @property
    def patch_size(self):
        return 2

    @property
    def default_height(self):
        return 1024

    @property
    def default_width(self):
        return 1024

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if getattr(self, "vae", None) is not None:
            vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        return vae_scale_factor

    @property
    def requires_unconditional_embeds(self):
        requires_unconditional_embeds = False
        if hasattr(self, "guider") and self.guider is not None:
            requires_unconditional_embeds = self.guider._enabled and self.guider.num_conditions > 1
        return requires_unconditional_embeds
