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

import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLQwenImage21
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImage21ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    # (batch_size, height * width, channels) -> (batch_size, channels, 1, height, width)
    batch_size, _, channels = latents.shape
    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    return latents.transpose(1, 2).reshape(batch_size, channels, 1, latent_height, latent_width)


# auto_docstring
class QwenImage21UnpackLatentsStep(ModularPipelineBlocks):
    """
    Step that unpacks the latents from (batch_size, sequence_length, channels) into (batch_size, channels, 1, height,
    width)

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`):
              The packed latents to unpack, can be generated in the denoise step.

      Outputs:
          latents (`Tensor`):
              The denoised latents unpacked to B, C, 1, H, W
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return "Step that unpacks the latents from (batch_size, sequence_length, channels) into (batch_size, channels, 1, height, width)"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The packed latents to unpack, can be generated in the denoise step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="latents", type_hint=torch.Tensor, description="The denoised latents unpacked to B, C, 1, H, W"
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        block_state.latents = unpack_latents(
            block_state.latents, block_state.height, block_state.width, components.vae_scale_factor
        )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21DecodeStep(ModularPipelineBlocks):
    """
    Step that decodes the latents to RGBA images and postprocesses them

      Components:
          vae (`AutoencoderKLQwenImage21`) image_processor (`VaeImageProcessor`)

      Inputs:
          latents (`Tensor`):
              The denoised latents to decode, can be generated in the denoise step and unpacked in the unpack latents
              step.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return "Step that decodes the latents to RGBA images and postprocesses them"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage21),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents to decode, can be generated in the denoise step and unpacked in the unpack latents step.",
            ),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @staticmethod
    def check_inputs(output_type):
        if output_type not in ["pil", "np", "pt"]:
            raise ValueError(f"Invalid output_type: {output_type}")

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.output_type)

        if block_state.latents.ndim == 4:
            block_state.latents = block_state.latents.unsqueeze(dim=2)
        elif block_state.latents.ndim != 5:
            raise ValueError(
                f"expect latents to be a 4D or 5D tensor but got: {block_state.latents.shape}. Please make sure the latents are unpacked before decode step."
            )

        latents = block_state.latents.to(components.vae.dtype)
        latents_mean = (
            torch.tensor(components.vae.config.latents_mean)
            .view(1, components.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(components.vae.config.latents_std)
            .view(1, components.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        images = components.vae.decode(latents, return_dict=False)[0][:, :, 0]
        block_state.images = components.image_processor.postprocess(images, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
