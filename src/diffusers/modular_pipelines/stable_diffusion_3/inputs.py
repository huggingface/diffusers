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

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from ..qwenimage.inputs import calculate_dimension_from_latents, repeat_tensor_to_batch_size
from .modular_pipeline import StableDiffusion3ModularPipeline


logger = logging.get_logger(__name__)


class StableDiffusion3TextInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def description(self) -> str:
        return (
            "Text input processing step that standardizes text embeddings for SD3, applying CFG duplication if needed."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_images_per_prompt", default=1),
            InputParam("guidance_scale", default=7.0),
            InputParam("skip_guidance_layers", type_hint=list),
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("pooled_prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[str]:
        return [
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
            OutputParam("do_classifier_free_guidance", type_hint=bool),
            OutputParam("prompt_embeds", type_hint=torch.Tensor),
            OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            OutputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype
        block_state.do_classifier_free_guidance = block_state.guidance_scale > 1.0

        _, seq_len, _ = block_state.prompt_embeds.shape
        prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = block_state.pooled_prompt_embeds.repeat(1, block_state.num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, -1
        )

        if block_state.do_classifier_free_guidance and block_state.negative_prompt_embeds is not None:
            _, neg_seq_len, _ = block_state.negative_prompt_embeds.shape
            negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, neg_seq_len, -1
            )

            negative_pooled_prompt_embeds = block_state.negative_pooled_prompt_embeds.repeat(
                1, block_state.num_images_per_prompt
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, -1
            )

            block_state.prompt_embeds = prompt_embeds
            block_state.pooled_prompt_embeds = pooled_prompt_embeds
            block_state.negative_prompt_embeds = negative_prompt_embeds
            block_state.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        else:
            block_state.prompt_embeds = prompt_embeds
            block_state.pooled_prompt_embeds = pooled_prompt_embeds
            block_state.negative_prompt_embeds = None
            block_state.negative_pooled_prompt_embeds = None

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusion3AdditionalInputsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    def __init__(self, image_latent_inputs: list[str] = ["image_latents"], additional_batch_inputs: list[str] = []):
        self._image_latent_inputs = (
            image_latent_inputs if isinstance(image_latent_inputs, list) else [image_latent_inputs]
        )
        self._additional_batch_inputs = (
            additional_batch_inputs if isinstance(additional_batch_inputs, list) else [additional_batch_inputs]
        )
        super().__init__()

    @property
    def description(self) -> str:
        return "Updates height/width if None, and expands batch size. SD3 does not pack latents on pipeline level."

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam("num_images_per_prompt", default=1),
            InputParam("batch_size", required=True),
            InputParam("height"),
            InputParam("width"),
        ]
        for name in self._image_latent_inputs + self._additional_batch_inputs:
            inputs.append(InputParam(name))
        return inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("image_height", type_hint=int),
            OutputParam("image_width", type_hint=int),
        ]

    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        for input_name in self._image_latent_inputs:
            tensor = getattr(block_state, input_name)
            if tensor is None:
                continue

            height, width = calculate_dimension_from_latents(tensor, components.vae_scale_factor)
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

            if not hasattr(block_state, "image_height"):
                block_state.image_height = height
            if not hasattr(block_state, "image_width"):
                block_state.image_width = width

            tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )
            setattr(block_state, input_name, tensor)

        for input_name in self._additional_batch_inputs:
            tensor = getattr(block_state, input_name)
            if tensor is None:
                continue
            tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )
            setattr(block_state, input_name, tensor)

        self.set_block_state(state, block_state)
        return components, state
