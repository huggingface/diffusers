# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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
from ...models import AutoencoderKLFlux2
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ErnieImageModularPipeline, ErnieImagePachifier


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ErnieImageVaeDecoderStep(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into images (unpachify, BN denormalization, VAE decode)."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLFlux2),
            ComponentSpec(
                "pachifier",
                ErnieImagePachifier,
                config=FrozenDict({"patch_size": 2}),
                default_creation_method="from_config",
            ),
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
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to decode into images.",
            ),
            InputParam(
                "output_type",
                type_hint=str,
                default="pil",
                description="Output format: 'pil', 'np', or 'pt'.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("images", type_hint=list, description="The generated images.")]

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae
        device = block_state.latents.device

        latents = block_state.latents
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=latents.dtype)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device=device, dtype=latents.dtype)
        latents = latents * bn_std + bn_mean

        latents = components.pachifier.unpack_latents(latents)

        images = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        block_state.images = components.image_processor.postprocess(images, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
