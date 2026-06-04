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
from .modular_pipeline import StableDiffusion3ModularPipeline


logger = logging.get_logger(__name__)


# Copied from diffusers.modular_pipelines.qwenimage.inputs.repeat_tensor_to_batch_size
def repeat_tensor_to_batch_size(
    input_name: str,
    input_tensor: torch.Tensor,
    batch_size: int,
    num_images_per_prompt: int = 1,
) -> torch.Tensor:
    """Repeat tensor elements to match the final batch size.

    This function expands a tensor's batch dimension to match the final batch size (batch_size * num_images_per_prompt)
    by repeating each element along dimension 0.

    The input tensor must have batch size 1 or batch_size. The function will:
    - If batch size is 1: repeat each element (batch_size * num_images_per_prompt) times
    - If batch size equals batch_size: repeat each element num_images_per_prompt times

    Args:
        input_name (str): Name of the input tensor (used for error messages)
        input_tensor (torch.Tensor): The tensor to repeat. Must have batch size 1 or batch_size.
        batch_size (int): The base batch size (number of prompts)
        num_images_per_prompt (int, optional): Number of images to generate per prompt. Defaults to 1.

    Returns:
        torch.Tensor: The repeated tensor with final batch size (batch_size * num_images_per_prompt)

    Raises:
        ValueError: If input_tensor is not a torch.Tensor or has invalid batch size

    Examples:
        tensor = torch.tensor([[1, 2, 3]]) # shape: [1, 3] repeated = repeat_tensor_to_batch_size("image", tensor,
        batch_size=2, num_images_per_prompt=2) repeated # tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]) - shape:
        [4, 3]

        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]) # shape: [2, 3] repeated = repeat_tensor_to_batch_size("image",
        tensor, batch_size=2, num_images_per_prompt=2) repeated # tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])
        - shape: [4, 3]
    """
    # make sure input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError(f"`{input_name}` must be a tensor")

    # make sure input tensor e.g. image_latents has batch size 1 or batch_size same as prompts
    if input_tensor.shape[0] == 1:
        repeat_by = batch_size * num_images_per_prompt
    elif input_tensor.shape[0] == batch_size:
        repeat_by = num_images_per_prompt
    else:
        raise ValueError(
            f"`{input_name}` must have have batch size 1 or {batch_size}, but got {input_tensor.shape[0]}"
        )

    # expand the tensor to match the batch_size * num_images_per_prompt
    input_tensor = input_tensor.repeat_interleave(repeat_by, dim=0)

    return input_tensor


# Copied from diffusers.modular_pipelines.qwenimage.inputs.calculate_dimension_from_latents
def calculate_dimension_from_latents(latents: torch.Tensor, vae_scale_factor: int) -> tuple[int, int]:
    """Calculate image dimensions from latent tensor dimensions.

    This function converts latent space dimensions to image space dimensions by multiplying the latent height and width
    by the VAE scale factor.

    Args:
        latents (torch.Tensor): The latent tensor. Must have 4 or 5 dimensions.
            Expected shapes: [batch, channels, height, width] or [batch, channels, frames, height, width]
        vae_scale_factor (int): The scale factor used by the VAE to compress images.
            Typically 8 for most VAEs (image is 8x larger than latents in each dimension)

    Returns:
        tuple[int, int]: The calculated image dimensions as (height, width)

    Raises:
        ValueError: If latents tensor doesn't have 4 or 5 dimensions

    """
    # make sure the latents are not packed
    if latents.ndim != 4 and latents.ndim != 5:
        raise ValueError(f"unpacked latents must have 4 or 5 dimensions, but got {latents.ndim}")

    latent_height, latent_width = latents.shape[-2:]

    height = latent_height * vae_scale_factor
    width = latent_width * vae_scale_factor

    return height, width


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
            InputParam(
                "num_images_per_prompt",
                default=1,
                description="The number of images to generate per prompt.",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-generated pooled text embeddings.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Pre-generated negative text embeddings.",
            ),
            InputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="Pre-generated negative pooled text embeddings.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="The batch size for the inference.",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="The expected data type for latents.",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="The processed text embeddings.",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="The processed pooled text embeddings.",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="The processed negative text embeddings.",
            ),
            OutputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="The processed negative pooled text embeddings.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape
        prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = block_state.pooled_prompt_embeds.repeat(1, block_state.num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, -1
        )

        if getattr(block_state, "negative_prompt_embeds", None) is not None:
            _, neg_seq_len, _ = block_state.negative_prompt_embeds.shape
            negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt,
                neg_seq_len,
                -1,
            )

            negative_pooled_prompt_embeds = block_state.negative_pooled_prompt_embeds.repeat(
                1, block_state.num_images_per_prompt
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, -1
            )

            block_state.negative_prompt_embeds = negative_prompt_embeds
            block_state.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        else:
            block_state.negative_prompt_embeds = None
            block_state.negative_pooled_prompt_embeds = None

        block_state.prompt_embeds = prompt_embeds
        block_state.pooled_prompt_embeds = pooled_prompt_embeds

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusion3AdditionalInputsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    def __init__(
        self,
        image_latent_inputs: list[str] = ["image_latents"],
        additional_batch_inputs: list[str] = [],
    ):
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
            InputParam(
                "num_images_per_prompt",
                default=1,
                description="The number of images to generate per prompt.",
            ),
            InputParam("batch_size", required=True, description="The batch size."),
            InputParam("height", description="The height in pixels of the generated image."),
            InputParam("width", description="The width in pixels of the generated image."),
        ]
        for name in self._image_latent_inputs + self._additional_batch_inputs:
            inputs.append(InputParam(name, description=f"Latent input {name} to be processed."))
        return inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_height",
                type_hint=int,
                description="The height of the generated image.",
            ),
            OutputParam(
                "image_width",
                type_hint=int,
                description="The width of the generated image.",
            ),
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
