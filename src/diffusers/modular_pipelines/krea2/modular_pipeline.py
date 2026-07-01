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

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "Krea2AutoBlocks"

    @property
    def default_height(self):
        return 1024

    @property
    def default_width(self):
        return 1024

    @property
    def patch_size(self):
        return self.config.get("patch_size", 2)

    @property
    def text_encoder_select_layers(self):
        return tuple(
            self.config.get(
                "text_encoder_select_layers",
                (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
            )
        )

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
            num_channels_latents = self.transformer.config.in_channels // (self.patch_size**2)
        return num_channels_latents

    def pack_latents(self, latents):
        patch_size = self.patch_size
        batch_size, num_channels_latents, latent_height, latent_width = latents.shape
        latents = latents.view(
            batch_size,
            num_channels_latents,
            latent_height // patch_size,
            patch_size,
            latent_width // patch_size,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size,
            (latent_height // patch_size) * (latent_width // patch_size),
            num_channels_latents * patch_size * patch_size,
        )
        return latents

    def unpack_latents(self, latents, height, width):
        batch_size, _, channels = latents.shape
        patch_size = self.patch_size
        height = patch_size * (int(height) // (self.vae_scale_factor * patch_size))
        width = patch_size * (int(width) // (self.vae_scale_factor * patch_size))

        latents = latents.view(
            batch_size,
            height // patch_size,
            width // patch_size,
            channels // (patch_size * patch_size),
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (patch_size * patch_size), 1, height, width)
        return latents
