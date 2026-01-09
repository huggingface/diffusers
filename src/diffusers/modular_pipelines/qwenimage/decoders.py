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
from .modular_pipeline import QwenImageLayeredPachifier, QwenImageModularPipeline, QwenImagePachifier


logger = logging.get_logger(__name__)


# after denoising loop (unpack latents)
class QwenImageAfterDenoiseStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that unpack the latents from 3D tensor (batch_size, sequence_length, channels) into 5D tensor (batch_size, channels, 1, height, width)"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        components = [
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

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        vae_scale_factor = components.vae_scale_factor
        block_state.latents = components.pachifier.unpack_latents(
            block_state.latents, block_state.height, block_state.width, vae_scale_factor=vae_scale_factor
        )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageLayeredAfterDenoiseStep(ModularPipelineBlocks):
    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Unpack latents from (B, seq, C*4) to (B, C, layers+1, H, W) after denoising."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImageLayeredPachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("height", required=True, type_hint=int),
            InputParam("width", required=True, type_hint=int),
            InputParam("layers", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Unpack: (B, seq, C*4) -> (B, C, layers+1, H, W)
        block_state.latents = components.pachifier.unpack_latents(
            block_state.latents,
            block_state.height,
            block_state.width,
            block_state.layers,
            components.vae_scale_factor,
        )

        self.set_block_state(state, block_state)
        return components, state


# decode step
class QwenImageDecoderStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that decodes the latents to images"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        components = [
            ComponentSpec("vae", AutoencoderKLQwenImage),
        ]

        return components

    @property
    def inputs(self) -> List[InputParam]:
        return [
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
        if block_state.latents.ndim == 4:
            block_state.latents = block_state.latents.unsqueeze(dim=1)
        elif block_state.latents.ndim != 5:
            raise ValueError(
                f"expect latents to be a 4D or 5D tensor but got: {block_state.latents.shape}. Please make sure the latents are unpacked before decode step."
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


class QwenImageLayeredDecoderStep(ModularPipelineBlocks):
    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Decode unpacked latents (B, C, layers+1, H, W) into layer images."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage),
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
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("output_type", default="pil", type_hint=str),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="images", type_hint=List[List[PIL.Image.Image]]),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents

        # 1. VAE normalization
        latents = latents.to(components.vae.dtype)
        latents_mean = (
            torch.tensor(components.vae.config.latents_mean)
            .view(1, components.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(components.vae.config.latents_std).view(
            1, components.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        # 2. Reshape for batch decoding: (B, C, layers+1, H, W) -> (B*layers, C, 1, H, W)
        b, c, f, h, w = latents.shape
        # 3. Remove first frame (composite), keep layers frames
        latents = latents[:, :, 1:]
        latents = latents.permute(0, 2, 1, 3, 4).reshape(-1, c, 1, h, w)

        # 4. Decode: (B*layers, C, 1, H, W) -> (B*layers, C, H, W)
        image = components.vae.decode(latents, return_dict=False)[0]
        image = image.squeeze(2)

        # 5. Postprocess - returns flat list of B*layers images
        image = components.image_processor.postprocess(image, output_type=block_state.output_type)

        # 6. Chunk into list per batch item
        images = []
        for bidx in range(b):
            images.append(image[bidx * f : (bidx + 1) * f])

        block_state.images = images

        self.set_block_state(state, block_state)
        return components, state


# postprocess the decoded images
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
