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

from typing import List, Tuple

import torch

from ...models import QwenImageMultiControlNetModel
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline, QwenImagePachifier


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


def calculate_dimension_from_latents(latents: torch.Tensor, vae_scale_factor: int) -> Tuple[int, int]:
    """Calculate image dimensions from latent tensor dimensions.

    This function converts latent space dimensions to image space dimensions by multiplying the latent height and width
    by the VAE scale factor.

    Args:
        latents (torch.Tensor): The latent tensor. Must have 4 or 5 dimensions.
            Expected shapes: [batch, channels, height, width] or [batch, channels, frames, height, width]
        vae_scale_factor (int): The scale factor used by the VAE to compress images.
            Typically 8 for most VAEs (image is 8x larger than latents in each dimension)

    Returns:
        Tuple[int, int]: The calculated image dimensions as (height, width)

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


class QwenImageTextInputsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        summary_section = (
            "Text input processing step that standardizes text embeddings for the pipeline.\n"
            "This step:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Ensures all text embeddings have consistent batch sizes (batch_size * num_images_per_prompt)"
        )

        # Placement guidance
        placement_section = "\n\nThis block should be placed after all encoder steps to process the text embeddings before they are used in subsequent pipeline steps."

        return summary_section + placement_section

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="prompt_embeds", required=True, kwargs_type="denoiser_input_fields"),
            InputParam(name="prompt_embeds_mask", required=True, kwargs_type="denoiser_input_fields"),
            InputParam(name="negative_prompt_embeds", kwargs_type="denoiser_input_fields"),
            InputParam(name="negative_prompt_embeds_mask", kwargs_type="denoiser_input_fields"),
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
        ]

    @staticmethod
    def check_inputs(
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
    ):
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("`negative_prompt_embeds_mask` is required when `negative_prompt_embeds` is not None")

        if negative_prompt_embeds is None and negative_prompt_embeds_mask is not None:
            raise ValueError("cannot pass `negative_prompt_embeds_mask` without `negative_prompt_embeds`")

        if prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`prompt_embeds_mask` must have the same batch size as `prompt_embeds`")

        elif negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`negative_prompt_embeds` must have the same batch size as `prompt_embeds`")

        elif (
            negative_prompt_embeds_mask is not None and negative_prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]
        ):
            raise ValueError("`negative_prompt_embeds_mask` must have the same batch size as `prompt_embeds`")

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(
            prompt_embeds=block_state.prompt_embeds,
            prompt_embeds_mask=block_state.prompt_embeds_mask,
            negative_prompt_embeds=block_state.negative_prompt_embeds,
            negative_prompt_embeds_mask=block_state.negative_prompt_embeds_mask,
        )

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape

        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
        )

        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len
        )

        if block_state.negative_prompt_embeds is not None:
            _, seq_len, _ = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(
                1, block_state.num_images_per_prompt, 1
            )
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
            )

            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.repeat(
                1, block_state.num_images_per_prompt, 1
            )
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len
            )

        self.set_block_state(state, block_state)

        return components, state


class QwenImageInputsDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    def __init__(
        self,
        image_latent_inputs: List[str] = ["image_latents"],
        additional_batch_inputs: List[str] = [],
    ):
        """Initialize a configurable step that standardizes the inputs for the denoising step. It:\n"

        This step handles multiple common tasks to prepare inputs for the denoising step:
        1. For encoded image latents, use it update height/width if None, patchifies, and expands batch size
        2. For additional_batch_inputs: Only expands batch dimensions to match final batch size

        This is a dynamic block that allows you to configure which inputs to process.

        Args:
            image_latent_inputs (List[str], optional): Names of image latent tensors to process.
                These will be used to determine height/width, patchified, and batch-expanded. Can be a single string or
                list of strings. Defaults to ["image_latents"]. Examples: ["image_latents"], ["control_image_latents"]
            additional_batch_inputs (List[str], optional):
                Names of additional conditional input tensors to expand batch size. These tensors will only have their
                batch dimensions adjusted to match the final batch size. Can be a single string or list of strings.
                Defaults to []. Examples: ["processed_mask_image"]

        Examples:
            # Configure to process image_latents (default behavior) QwenImageInputsDynamicStep()

            # Configure to process multiple image latent inputs
            QwenImageInputsDynamicStep(image_latent_inputs=["image_latents", "control_image_latents"])

            # Configure to process image latents and additional batch inputs QwenImageInputsDynamicStep(
                image_latent_inputs=["image_latents"], additional_batch_inputs=["processed_mask_image"]
            )
        """
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

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
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
            image_latent_tensor = components.pachifier.pack_latents(image_latent_tensor)

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


class QwenImageControlNetInputsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "prepare the `control_image_latents` for controlnet. Insert after all the other inputs steps."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="control_image_latents", required=True),
            InputParam(name="batch_size", required=True),
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="height"),
            InputParam(name="width"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if isinstance(components.controlnet, QwenImageMultiControlNetModel):
            control_image_latents = []
            # loop through each control_image_latents
            for i, control_image_latents_ in enumerate(block_state.control_image_latents):
                # 1. update height/width if not provided
                height, width = calculate_dimension_from_latents(control_image_latents_, components.vae_scale_factor)
                block_state.height = block_state.height or height
                block_state.width = block_state.width or width

                # 2. pack
                control_image_latents_ = components.pachifier.pack_latents(control_image_latents_)

                # 3. repeat to match the batch size
                control_image_latents_ = repeat_tensor_to_batch_size(
                    input_name=f"control_image_latents[{i}]",
                    input_tensor=control_image_latents_,
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    batch_size=block_state.batch_size,
                )

                control_image_latents.append(control_image_latents_)

            block_state.control_image_latents = control_image_latents

        else:
            # 1. update height/width if not provided
            height, width = calculate_dimension_from_latents(
                block_state.control_image_latents, components.vae_scale_factor
            )
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

            # 2. pack
            block_state.control_image_latents = components.pachifier.pack_latents(block_state.control_image_latents)

            # 3. repeat to match the batch size
            block_state.control_image_latents = repeat_tensor_to_batch_size(
                input_name="control_image_latents",
                input_tensor=block_state.control_image_latents,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            block_state.control_image_latents = block_state.control_image_latents

        self.set_block_state(state, block_state)

        return components, state
