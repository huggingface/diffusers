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
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLQwenImage
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
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
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
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

        vae = components.vae
        p = components.patch_size
        latents = block_state.latents

        # Unpack the patchified token sequence back to the spatial latent grid (mirrors Krea2Pipeline._unpack_latents).
        batch_size, _, channels = latents.shape
        height = p * (int(block_state.height) // (components.vae_scale_factor * p))
        width = p * (int(block_state.width) // (components.vae_scale_factor * p))
        latents = latents.view(batch_size, height // p, width // p, channels // (p * p), p, p)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (p * p), 1, height, width)

        # De-normalize with the per-channel VAE statistics, then decode (the VAE produces a single-frame video latent).
        latents = latents.to(vae.dtype)
        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]
        block_state.images = components.image_processor.postprocess(image, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
