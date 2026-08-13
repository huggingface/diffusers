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


import torch

from ...configuration_utils import FrozenDict
from ...image_processor import InpaintProcessor, VaeImageProcessor
from ...models import AutoencoderKLQwenImage
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _decode_latents(components: Krea2ModularPipeline, latents: torch.Tensor, height: int, width: int):
    vae = components.vae
    p = components.patch_size
    batch_size, _, channels = latents.shape
    latent_height = p * (height // (components.vae_scale_factor * p))
    latent_width = p * (width // (components.vae_scale_factor * p))
    latents = latents.view(batch_size, latent_height // p, latent_width // p, channels // (p * p), p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (p * p), 1, latent_height, latent_width).to(vae.dtype)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    latents = latents * latents_std + latents_mean
    return vae.decode(latents, return_dict=False)[0][:, :, 0]


# auto_docstring
class Krea2DecodeStep(ModularPipelineBlocks):
    """
    Step that unpacks the denoised packed latents back to the spatial grid, de-normalizes them with the VAE's
    per-channel statistics, and decodes them through the Qwen-Image VAE into images.

      Components:
          vae (`AutoencoderKLQwenImage`) image_processor (`VaeImageProcessor`)

      Inputs:
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          latents (`Tensor`):
              The denoised packed latents (B, image_seq_len, in_channels) from the denoising loop.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that unpacks the denoised packed latents back to the spatial grid, de-normalizes them with the "
            "VAE's per-channel statistics, and decodes them through the Qwen-Image VAE into images."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                # Effective pixel-to-token downsampling factor: vae_scale_factor (8) * patch_size (2).
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("output_type", default="pil"),
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised packed latents (B, image_seq_len, in_channels) from the denoising loop.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image = _decode_latents(components, block_state.latents, int(block_state.height), int(block_state.width))
        block_state.images = components.image_processor.postprocess(image, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2InpaintDecodeStep(ModularPipelineBlocks):
    """
    Decode Krea 2 inpainting latents and optionally overlay a cropped result on the original image.

      Components:
          vae (`AutoencoderKLQwenImage`) image_mask_processor (`InpaintProcessor`)

      Inputs:
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          mask_overlay_kwargs (`dict`, *optional*):
              Arguments used to overlay a cropped inpainting result on the original image.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Decode Krea 2 inpainting latents and optionally overlay a cropped result on the original image."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("output_type", default="pil"),
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam.template("latents", required=True),
            InputParam(
                name="mask_overlay_kwargs",
                type_hint=dict,
                description="Arguments used to overlay a cropped inpainting result on the original image.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image = _decode_latents(components, block_state.latents, int(block_state.height), int(block_state.width))
        overlay_kwargs = block_state.mask_overlay_kwargs or {}
        block_state.images = components.image_mask_processor.postprocess(
            image, output_type=block_state.output_type, **overlay_kwargs
        )

        self.set_block_state(state, block_state)
        return components, state
