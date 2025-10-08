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

from typing import List

import torch

from ...pipelines import FluxPipeline
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam

# TODO: consider making these common utilities for modular if they are not pipeline-specific.
from ..qwenimage.inputs import calculate_dimension_from_latents, repeat_tensor_to_batch_size
from .modular_pipeline import FluxModularPipeline


class FluxTextInputStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def description(self) -> str:
        return (
            "Text input processing step that standardizes text embeddings for the pipeline.\n"
            "This step:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Ensures all text embeddings have consistent batch sizes (batch_size * num_images_per_prompt)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "prompt_embeds",
                required=True,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Pre-generated pooled text embeddings. Can be generated from text_encoder step.",
            ),
            # TODO: support negative embeddings?
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (determined by `prompt_embeds`)",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="pooled text embeddings used to guide the image generation",
            ),
            # TODO: support negative embeddings?
        ]

    def check_inputs(self, components, block_state):
        if block_state.prompt_embeds is not None and block_state.pooled_prompt_embeds is not None:
            if block_state.prompt_embeds.shape[0] != block_state.pooled_prompt_embeds.shape[0]:
                raise ValueError(
                    "`prompt_embeds` and `pooled_prompt_embeds` must have the same batch size when passed directly, but"
                    f" got: `prompt_embeds` {block_state.prompt_embeds.shape} != `pooled_prompt_embeds`"
                    f" {block_state.pooled_prompt_embeds.shape}."
                )

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        # TODO: consider adding negative embeddings?
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
        )
        self.set_block_state(state, block_state)

        return components, state


# Adapted from `QwenImageInputsDynamicStep`
class FluxInputsDynamicStep(ModularPipelineBlocks):
    model_name = "flux"

    def __init__(
        self,
        image_latent_inputs: List[str] = ["image_latents"],
        additional_batch_inputs: List[str] = [],
    ):
        if not isinstance(image_latent_inputs, list):
            image_latent_inputs = [image_latent_inputs]
        if not isinstance(additional_batch_inputs, list):
            additional_batch_inputs = [additional_batch_inputs]

        self._image_latent_inputs = image_latent_inputs
        self._additional_batch_inputs = additional_batch_inputs
        super().__init__()

    @property
    def description(self) -> str:
        # Functionality section
        summary_section = (
            "Input processing step that:\n"
            "  1. For image latent inputs: Updates height/width if None, patchifies latents, and expands batch size\n"
            "  2. For additional batch inputs: Expands batch dimensions to match final batch size"
        )

        # Inputs info
        inputs_info = ""
        if self._image_latent_inputs or self._additional_batch_inputs:
            inputs_info = "\n\nConfigured inputs:"
            if self._image_latent_inputs:
                inputs_info += f"\n  - Image latent inputs: {self._image_latent_inputs}"
            if self._additional_batch_inputs:
                inputs_info += f"\n  - Additional batch inputs: {self._additional_batch_inputs}"

        # Placement guidance
        placement_section = "\n\nThis block should be placed after the encoder steps and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def inputs(self) -> List[InputParam]:
        inputs = [
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="batch_size", required=True),
            InputParam(name="height"),
            InputParam(name="width"),
        ]

        # Add image latent inputs
        for image_latent_input_name in self._image_latent_inputs:
            inputs.append(InputParam(name=image_latent_input_name))

        # Add additional batch inputs
        for input_name in self._additional_batch_inputs:
            inputs.append(InputParam(name=input_name))

        return inputs

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="image_height", type_hint=int, description="The height of the image latents"),
            OutputParam(name="image_width", type_hint=int, description="The width of the image latents"),
        ]

    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Process image latent inputs (height/width calculation, patchify, and batch expansion)
        for image_latent_input_name in self._image_latent_inputs:
            image_latent_tensor = getattr(block_state, image_latent_input_name)
            if image_latent_tensor is None:
                continue

            # 1. Calculate height/width from latents
            height, width = calculate_dimension_from_latents(image_latent_tensor, components.vae_scale_factor)
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

            if not hasattr(block_state, "image_height"):
                block_state.image_height = height
            if not hasattr(block_state, "image_width"):
                block_state.image_width = width

            # 2. Patchify the image latent tensor
            # TODO: Implement patchifier for Flux.
            latent_height, latent_width = image_latent_tensor.shape[2:]
            image_latent_tensor = FluxPipeline._pack_latents(
                image_latent_tensor, block_state.batch_size, image_latent_tensor.shape[1], latent_height, latent_width
            )

            # 3. Expand batch size
            image_latent_tensor = repeat_tensor_to_batch_size(
                input_name=image_latent_input_name,
                input_tensor=image_latent_tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, image_latent_input_name, image_latent_tensor)

        # Process additional batch inputs (only batch expansion)
        for input_name in self._additional_batch_inputs:
            input_tensor = getattr(block_state, input_name)
            if input_tensor is None:
                continue

            # Only expand batch size
            input_tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=input_tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_name, input_tensor)

        self.set_block_state(state, block_state)
        return components, state
