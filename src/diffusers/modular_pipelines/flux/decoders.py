# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Any, List, Tuple, Union

import numpy as np
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL
from ...utils import logging
from ...video_processor import VaeImageProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


class FluxDecodeStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into images"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("output_type", default="pil"),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents from the denoising step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "images",
                type_hint=Union[List[PIL.Image.Image], torch.Tensor, np.ndarray],
                description="The generated images, can be a list of PIL.Image.Image, torch.Tensor or a numpy array",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        if not block_state.output_type == "latent":
            latents = block_state.latents
            latents = _unpack_latents(latents, block_state.height, block_state.width, components.vae_scale_factor)
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            block_state.images = vae.decode(latents, return_dict=False)[0]
            block_state.images = components.image_processor.postprocess(
                block_state.images, output_type=block_state.output_type
            )
        else:
            block_state.images = block_state.latents

        self.set_block_state(state, block_state)

        return components, state
