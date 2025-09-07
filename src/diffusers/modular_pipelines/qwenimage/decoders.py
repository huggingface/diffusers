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

from typing import List, Union

import numpy as np
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...image_processor import InpaintProcessor, VaeImageProcessor
from ...models import AutoencoderKLQwenImage
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline, QwenImagePachifier


logger = logging.get_logger(__name__)


class QwenImageDecoderStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that decodes the latents to images"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        components = [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

        return components

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to decode, can be generated in the denoise step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "images",
                type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]],
                description="The generated images, can be a PIL.Image.Image, torch.Tensor or a numpy array",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # YiYi Notes: remove support for output_type = "latents', we can just skip decode/encode step in modular
        block_state.latents = components.pachifier.unpack_latents(
            block_state.latents, block_state.height, block_state.width
        )
        block_state.latents = block_state.latents.to(components.vae.dtype)

        latents_mean = (
            torch.tensor(components.vae.config.latents_mean)
            .view(1, components.vae.config.z_dim, 1, 1, 1)
            .to(block_state.latents.device, block_state.latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(components.vae.config.latents_std).view(
            1, components.vae.config.z_dim, 1, 1, 1
        ).to(block_state.latents.device, block_state.latents.dtype)
        block_state.latents = block_state.latents / latents_std + latents_mean
        block_state.images = components.vae.decode(block_state.latents, return_dict=False)[0][:, :, 0]

        self.set_block_state(state, block_state)
        return components, state


class QwenImageProcessImagesOutputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "postprocess the generated image"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("images", required=True, description="the generated image from decoders step"),
            InputParam(
                name="output_type",
                default="pil",
                type_hint=str,
                description="The type of the output images, can be 'pil', 'np', 'pt'",
            ),
        ]

    @staticmethod
    def check_inputs(output_type):
        if output_type not in ["pil", "np", "pt"]:
            raise ValueError(f"Invalid output_type: {output_type}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.output_type)

        block_state.images = components.image_processor.postprocess(
            image=block_state.images,
            output_type=block_state.output_type,
        )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageInpaintProcessImagesOutputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "postprocess the generated image, optional apply the mask overally to the original image.."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("images", required=True, description="the generated image from decoders step"),
            InputParam(
                name="output_type",
                default="pil",
                type_hint=str,
                description="The type of the output images, can be 'pil', 'np', 'pt'",
            ),
            InputParam("mask_overlay_kwargs"),
        ]

    @staticmethod
    def check_inputs(output_type, mask_overlay_kwargs):
        if output_type not in ["pil", "np", "pt"]:
            raise ValueError(f"Invalid output_type: {output_type}")

        if mask_overlay_kwargs and output_type != "pil":
            raise ValueError("only support output_type 'pil' for mask overlay")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.output_type, block_state.mask_overlay_kwargs)

        if block_state.mask_overlay_kwargs is None:
            mask_overlay_kwargs = {}
        else:
            mask_overlay_kwargs = block_state.mask_overlay_kwargs

        block_state.images = components.image_mask_processor.postprocess(
            image=block_state.images,
            **mask_overlay_kwargs,
        )

        self.set_block_state(state, block_state)
        return components, state
