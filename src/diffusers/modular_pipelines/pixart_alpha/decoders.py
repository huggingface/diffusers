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
from ...models import AutoencoderKL
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import PixArtAlphaModularPipeline


logger = logging.get_logger(__name__)


# decode step


# auto_docstring
class PixArtAlphaDecodeStep(ModularPipelineBlocks):
    """
    Step that decodes the denoised latents into an image tensor with the VAE.

      Components:
          vae (`AutoencoderKL`)

      Inputs:
          latents (`Tensor`):
              The denoised latents to decode, can be generated in the denoise step.

      Outputs:
          images (`list`):
              Generated images. (tensor output of the vae decoder.)
    """

    model_name = "pixart-alpha"

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into an image tensor with the VAE."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents to decode, can be generated in the denoise step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images", note="tensor output of the vae decoder.")]

    @torch.no_grad()
    def __call__(self, components: PixArtAlphaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents / components.vae.config.scaling_factor
        block_state.images = components.vae.decode(latents, return_dict=False)[0]

        self.set_block_state(state, block_state)
        return components, state


# postprocess the decoded images


# auto_docstring
class PixArtAlphaProcessImagesOutputStep(ModularPipelineBlocks):
    """
    Step that postprocesses the decoded image tensor into the requested output format.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          images (`Tensor`):
              The image tensor from the decode step.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "pixart-alpha"

    @property
    def description(self) -> str:
        return "Step that postprocesses the decoded image tensor into the requested output format."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "images",
                required=True,
                type_hint=torch.Tensor,
                description="The image tensor from the decode step.",
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
    def __call__(self, components: PixArtAlphaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.output_type)

        block_state.images = components.image_processor.postprocess(
            block_state.images, output_type=block_state.output_type
        )

        self.set_block_state(state, block_state)
        return components, state
