# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import math

import torch

from ...models import Flux3Transformer2DModel
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Flux3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
        raise ValueError(f"`{input_name}` must have batch size 1 or {batch_size}, but got {input_tensor.shape[0]}")

    # expand the tensor to match the batch_size * num_images_per_prompt
    input_tensor = input_tensor.repeat_interleave(repeat_by, dim=0)

    return input_tensor


class Flux3TextInputStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return (
            "This step:\n"
            "  1. Determines `batch_size` from `prompt_embeds` and the compute `dtype` from the transformer\n"
            "  2. Ensures all text embeddings have consistent batch sizes (batch_size * num_images_per_prompt)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Flux3Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam(
                "prompt_embeds",
                required=True,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "negative_prompt_embeds",
                required=False,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Pre-generated negative text embeddings. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (the transformer's dtype)",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Negative text embeddings used to guide the image generation",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        # the denoiser consumes every tensor this chain prepares, so its dtype is the compute dtype:
        # the text encoder may run in a different precision (e.g. a bf16 encoder with an fp32 transformer)
        block_state.dtype = components.transformer.dtype

        block_state.prompt_embeds = repeat_tensor_to_batch_size(
            "prompt_embeds", block_state.prompt_embeds, block_state.batch_size, block_state.num_images_per_prompt
        ).to(block_state.dtype)

        if block_state.negative_prompt_embeds is not None:
            block_state.negative_prompt_embeds = repeat_tensor_to_batch_size(
                "negative_prompt_embeds",
                block_state.negative_prompt_embeds,
                block_state.batch_size,
                block_state.num_images_per_prompt,
            ).to(block_state.dtype)
        elif components.requires_unconditional_embeds:
            raise ValueError(
                "The guider needs unconditional embeddings but `negative_prompt_embeds` is None. Pass `negative_prompt` "
                "(an empty string for plain classifier-free guidance) to the text encoder step, or disable the guider."
            )

        self.set_block_state(state, block_state)
        return components, state


class Flux3DistilledTextInputStep(ModularPipelineBlocks):
    model_name = "flux3-distilled"

    @property
    def description(self) -> str:
        return (
            "This step:\n"
            "  1. Determines `batch_size` from `prompt_embeds` and the compute `dtype` from the transformer\n"
            "  2. Repeats the text embeddings to the batch size (batch_size * num_images_per_prompt)\n"
            "For the step-distilled FLUX.3 model, which samples without classifier-free guidance (no negative "
            "embeddings)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Flux3Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam(
                "prompt_embeds",
                required=True,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (the transformer's dtype)",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Text embeddings used to guide the image generation",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        # the denoiser consumes every tensor this chain prepares, so its dtype is the compute dtype
        block_state.dtype = components.transformer.dtype

        block_state.prompt_embeds = repeat_tensor_to_batch_size(
            "prompt_embeds", block_state.prompt_embeds, block_state.batch_size, block_state.num_images_per_prompt
        ).to(block_state.dtype)

        self.set_block_state(state, block_state)
        return components, state


class Flux3ImageCondInputStep(ModularPipelineBlocks):
    model_name = "flux3"

    def __init__(self, default_height: int = 1024, default_width: int = 1024):
        # the generation size defaults to this area at the first reference image's aspect ratio
        self.default_height = default_height
        self.default_width = default_width
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Step that expands the reference image latents to the final batch size and sets the generation "
            f"`height`/`width` when not provided: {self.default_height} x {self.default_width} pixels of area at the "
            "first reference image's aspect ratio, on the 16-pixel grid (a given side overrides its default). The "
            "reference images keep their own resolution, the generation size is independent of them."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image_cond_latents",
                required=True,
                type_hint=list[torch.Tensor],
                description="Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16) per image",
            ),
            InputParam(
                "height",
                type_hint=int,
                description="The height in pixels of the generated image; defaults to the first reference image's "
                "aspect ratio at the default area",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="The width in pixels of the generated image; defaults to the first reference image's "
                "aspect ratio at the default area",
            ),
            InputParam.template("num_images_per_prompt"),
            InputParam("batch_size", required=True, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_cond_latents",
                type_hint=list[torch.Tensor],
                description="Normalized packed latents of the reference images, expanded to the final batch size",
            ),
            OutputParam("height", type_hint=int, description="The height in pixels of the generated image"),
            OutputParam("width", type_hint=int, description="The width in pixels of the generated image"),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # the generation size defaults to the default area at the first reference's aspect ratio, rounded to a
        # multiple of 16 pixels (one token); a given `height` or `width` overrides its side
        if block_state.height is None or block_state.width is None:
            multiple_of = components.vae_scale_factor * 2
            latent_height, latent_width = block_state.image_cond_latents[0].shape[-2:]
            scale = math.sqrt(self.default_height * self.default_width / (latent_height * latent_width)) / multiple_of
            block_state.height = block_state.height or max(multiple_of, round(latent_height * scale) * multiple_of)
            block_state.width = block_state.width or max(multiple_of, round(latent_width * scale) * multiple_of)

        block_state.image_cond_latents = [
            repeat_tensor_to_batch_size(
                "image_cond_latents", latents, block_state.batch_size, block_state.num_images_per_prompt
            )
            for latents in block_state.image_cond_latents
        ]

        self.set_block_state(state, block_state)
        return components, state
