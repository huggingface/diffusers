# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import Any

import numpy as np
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ZImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ZImageVaeDecoderStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8 * 2}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into images"

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam(
                "latents",
                required=True,
            ),
            InputParam(
                name="output_type",
                default="pil",
                type_hint=str,
                description="The type of the output images, can be 'pil', 'np', 'pt'",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[str]:
        return [
            OutputParam(
                "images",
                type_hint=list[PIL.Image.Image, list[torch.Tensor], list[np.ndarray]],
                description="The generated images, can be a PIL.Image.Image, torch.Tensor or a numpy array",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae_dtype = components.vae.dtype

        latents = block_state.latents.to(vae_dtype)
        latents = (
            latents / components.vae.config.scaling_factor
            + components.vae.config.shift_factor
        )

        block_state.images = components.vae.decode(latents, return_dict=False)[0]
        block_state.images = components.image_processor.postprocess(
            block_state.images, output_type=block_state.output_type
        )

        self.set_block_state(state, block_state)

        return components, state


class ZImageInpaintOverlayMaskStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8 * 2}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Overlays a cropped inpaint result onto the original image."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("images", required=True),
            InputParam("image", required=True, type_hint=PIL.Image.Image),
            InputParam("mask_image", required=True, type_hint=PIL.Image.Image),
            InputParam.template("padding_mask_crop"),
            InputParam("crops_coords", type_hint=tuple[int, int, int, int] | None),
            InputParam("output_type", default="pil", type_hint=str),
        ]

    @torch.no_grad()
    def __call__(
        self, components: ZImageModularPipeline, state: PipelineState
    ) -> PipelineState:
        block_state = self.get_block_state(state)
        if block_state.padding_mask_crop is not None:
            if block_state.output_type != "pil":
                raise ValueError(
                    "`output_type` must be 'pil' when `padding_mask_crop` is provided."
                )
            block_state.images = [
                components.image_processor.apply_overlay(
                    block_state.mask_image,
                    block_state.image,
                    image,
                    block_state.crops_coords,
                )
                for image in block_state.images
            ]
        self.set_block_state(state, block_state)
        return components, state
