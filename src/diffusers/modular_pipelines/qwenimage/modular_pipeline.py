# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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


from ...loaders import QwenImageLoraLoaderMixin
from ..modular_pipeline import ModularPipeline


class QwenImageModularPipeline(ModularPipeline, QwenImageLoraLoaderMixin):
    """
    A ModularPipeline for QwenImage.

    <Tip warning={true}>

        This is an experimental feature and is likely to change in the future.

    </Tip>
    """

    @property
    def default_height(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_width(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_sample_size(self):
        return 128

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 16
        if hasattr(self, "transformer") and self.transformer is not None:
            num_channels_latents = self.transformer.config.in_channels // 4
        return num_channels_latents

    @property
    def is_guidance_distilled(self):
        is_guidance_distilled = False
        if hasattr(self, "transformer") and self.transformer is not None:
            is_guidance_distilled = self.transformer.config.guidance_embeds
        return is_guidance_distilled

    @property
    def requires_unconditional_embeds(self):
        requires_unconditional_embeds = False

        if hasattr(self, "guider") and self.guider is not None:
            requires_unconditional_embeds = self.guider._enabled and self.guider.num_conditions > 1

        return requires_unconditional_embeds
