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

import PIL
import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


class StableDiffusion3DecodeStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return[
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8, "vae_latent_channels": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "output_type",
                default="pil",
                description="The output format of the generated image (e.g., 'pil', 'pt', 'np').",
            ),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents to be decoded.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("images", type_hint=list[PIL.Image.Image] | torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        if not block_state.output_type == "latent":
            latents = (block_state.latents / vae.config.scaling_factor) + vae.config.shift_factor
            block_state.images = vae.decode(latents, return_dict=False)[0]
            block_state.images = components.image_processor.postprocess(
                block_state.images, output_type=block_state.output_type
            )
        else:
            block_state.images = block_state.latents

        self.set_block_state(state, block_state)
        return components, state
