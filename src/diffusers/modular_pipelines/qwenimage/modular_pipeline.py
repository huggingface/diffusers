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


from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import QwenImageLoraLoaderMixin
from ..modular_pipeline import ModularPipeline


class QwenImagePachifier(ConfigMixin):
    """
    A class to pack and unpack latents for QwenImage.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(self, patch_size: int = 2):
        super().__init__()

    def pack_latents(self, latents):
        if latents.ndim != 4 and latents.ndim != 5:
            raise ValueError(f"Latents must have 4 or 5 dimensions, but got {latents.ndim}")

        if latents.ndim == 4:
            latents = latents.unsqueeze(2)

        batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width = latents.shape
        patch_size = self.config.patch_size

        if latent_height % patch_size != 0 or latent_width % patch_size != 0:
            raise ValueError(
                f"Latent height and width must be divisible by {patch_size}, but got {latent_height} and {latent_width}"
            )

        latents = latents.view(
            batch_size,
            num_channels_latents,
            latent_height // patch_size,
            patch_size,
            latent_width // patch_size,
            patch_size,
        )
        latents = latents.permute(
            0, 2, 4, 1, 3, 5
        )  # Batch_size, num_patches_height, num_patches_width, num_channels_latents, patch_size, patch_size
        latents = latents.reshape(
            batch_size,
            (latent_height // patch_size) * (latent_width // patch_size),
            num_channels_latents * patch_size * patch_size,
        )

        return latents

    def unpack_latents(self, latents, height, width, vae_scale_factor=8):
        if latents.ndim != 3:
            raise ValueError(f"Latents must have 3 dimensions, but got {latents.ndim}")

        batch_size, num_patches, channels = latents.shape
        patch_size = self.config.patch_size

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = patch_size * (int(height) // (vae_scale_factor * patch_size))
        width = patch_size * (int(width) // (vae_scale_factor * patch_size))

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


class QwenImageModularPipeline(ModularPipeline, QwenImageLoraLoaderMixin):
    """
    A ModularPipeline for QwenImage.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "QwenImageAutoBlocks"

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


class QwenImageEditModularPipeline(ModularPipeline, QwenImageLoraLoaderMixin):
    """
    A ModularPipeline for QwenImage-Edit.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "QwenImageEditAutoBlocks"

    # YiYi TODO: qwen edit should not provide default height/width, should be derived from the resized input image (after adjustment) produced by the resize step.
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


class QwenImageEditPlusModularPipeline(QwenImageEditModularPipeline):
    """
    A ModularPipeline for QwenImage-Edit Plus.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "QwenImageEditPlusAutoBlocks"
