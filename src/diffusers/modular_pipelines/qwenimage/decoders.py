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


from typing import Any

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


# auto_docstring
class QwenImageAfterDenoiseStep(ModularPipelineBlocks):
    """
    Step that unpack the latents from 3D tensor (batch_size, sequence_length, channels) into 5D tensor (batch_size,
    channels, 1, height, width)

      Components:
          pachifier (`QwenImagePachifier`)

      Inputs:
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          latents (`Tensor`):
              The latents to decode, can be generated in the denoise step.

      Outputs:
          latents (`Tensor`):
              The denoisedlatents unpacked to B, C, 1, H, W
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that unpack the latents from 3D tensor (batch_size, sequence_length, channels) into 5D tensor (batch_size, channels, 1, height, width)"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        components = [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

        return components

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to decode, can be generated in the denoise step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="latents", type_hint=torch.Tensor, description="The denoisedlatents unpacked to B, C, 1, H, W"
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


# auto_docstring
class QwenImageLayeredAfterDenoiseStep(ModularPipelineBlocks):
    """
    Unpack latents from (B, seq, C*4) to (B, C, layers+1, H, W) after denoising.

      Components:
          pachifier (`QwenImageLayeredPachifier`)

      Inputs:
          latents (`Tensor`):
              The denoised latents to decode, can be generated in the denoise step.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          layers (`int`, *optional*, defaults to 4):
              Number of layers to extract from the image

      Outputs:
          latents (`Tensor`):
              Denoised latents. (unpacked to B, C, layers+1, H, W)
    """

    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Unpack latents from (B, seq, C*4) to (B, C, layers+1, H, W) after denoising."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImageLayeredPachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents to decode, can be generated in the denoise step.",
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("layers"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents", note="unpacked to B, C, layers+1, H, W"),
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


# auto_docstring
class QwenImageDecoderStep(ModularPipelineBlocks):
    """
    Step that decodes the latents to images

      Components:
          vae (`AutoencoderKLQwenImage`)

      Inputs:
          latents (`Tensor`):
              The denoised latents to decode, can be generated in the denoise step and unpacked in the after denoise
              step.

      Outputs:
          images (`list`):
              Generated images. (tensor output of the vae decoder.)
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that decodes the latents to images"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        components = [
            ComponentSpec("vae", AutoencoderKLQwenImage),
        ]

        return components

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents to decode, can be generated in the denoise step and unpacked in the after denoise step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images", note="tensor output of the vae decoder.")]

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


# auto_docstring
class QwenImageLayeredDecoderStep(ModularPipelineBlocks):
    """
    Decode unpacked latents (B, C, layers+1, H, W) into layer images.

      Components:
          vae (`AutoencoderKLQwenImage`) image_processor (`VaeImageProcessor`)

      Inputs:
          latents (`Tensor`):
              The denoised latents to decode, can be generated in the denoise step and unpacked in the after denoise
              step.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Decode unpacked latents (B, C, layers+1, H, W) into layer images."

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents to decode, can be generated in the denoise step and unpacked in the after denoise step.",
            ),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

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


# auto_docstring
class QwenImageProcessImagesOutputStep(ModularPipelineBlocks):
    """
    postprocess the generated image

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          images (`Tensor`):
              the generated image tensor from decoders step
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "postprocess the generated image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
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
                name="images",
                required=True,
                type_hint=torch.Tensor,
                description="the generated image tensor from decoders step",
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
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.output_type)

        block_state.images = components.image_processor.postprocess(
            image=block_state.images,
            output_type=block_state.output_type,
        )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageInpaintProcessImagesOutputStep(ModularPipelineBlocks):
    """
    postprocess the generated image, optional apply the mask overally to the original image..

      Components:
          image_mask_processor (`InpaintProcessor`)

      Inputs:
          images (`Tensor`):
              the generated image tensor from decoders step
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          mask_overlay_kwargs (`dict`, *optional*):
              The kwargs for the postprocess step to apply the mask overlay. generated in
              InpaintProcessImagesInputStep.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "postprocess the generated image, optional apply the mask overally to the original image.."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
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
            InputParam(
                name="images",
                required=True,
                type_hint=torch.Tensor,
                description="the generated image tensor from decoders step",
            ),
            InputParam.template("output_type"),
            InputParam(
                name="mask_overlay_kwargs",
                type_hint=dict[str, Any],
                description="The kwargs for the postprocess step to apply the mask overlay. generated in InpaintProcessImagesInputStep.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

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
