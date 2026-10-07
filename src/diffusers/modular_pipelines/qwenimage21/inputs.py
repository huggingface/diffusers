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
from .modular_pipeline import QwenImage21ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    # Qwen-Image 2.1 consumes latents unpatched, so packing is a plain spatial flatten:
    # (batch_size, channels, 1, height, width) -> (batch_size, height * width, channels)
    batch_size, channels, _, height, width = latents.shape
    return latents.view(batch_size, channels, height * width).transpose(1, 2)


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


# auto_docstring
class QwenImage21TextInputsStep(ModularPipelineBlocks):
    """
    Text input processing step that standardizes text embeddings for the pipeline.
      This step:
        1. Determines `batch_size` and `dtype` based on `prompt_embeds`
        2. Expands all text embeddings and masks to batch_size * num_images_per_prompt
        3. Drops an attention mask that carries no padding, as the attention backends then skip masking

      This block should be placed after all encoder steps to process the text embeddings before they are used in
      subsequent pipeline steps.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.
          negative_image_pad_mask (`Tensor`, *optional*):
              Bool mask marking the vision positions of the negative prompt embeddings. Can be generated from
              text_encoder step.

      Outputs:
          batch_size (`int`):
              The batch size of the prompt embeddings
          dtype (`dtype`):
              The data type of the prompt embeddings
          prompt_embeds (`Tensor`):
              The prompt embeddings. (batch-expanded)
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask. (batch-expanded, None when nothing is padded)
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings. (batch-expanded)
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask. (batch-expanded, None when nothing is padded)
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings (batch-expanded)
          negative_image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the negative prompt embeddings (batch-expanded)
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Text input processing step that standardizes text embeddings for the pipeline.\n"
            "This step:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Expands all text embeddings and masks to batch_size * num_images_per_prompt\n"
            "  3. Drops an attention mask that carries no padding, as the attention backends then skip masking\n\n"
            "This block should be placed after all encoder steps to process the text embeddings before they are used "
            "in subsequent pipeline steps."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("prompt_embeds"),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds"),
            InputParam.template("negative_prompt_embeds_mask"),
            InputParam(
                name="image_pad_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                name="negative_image_pad_mask",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the negative prompt embeddings. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="batch_size", type_hint=int, description="The batch size of the prompt embeddings"),
            OutputParam(name="dtype", type_hint=torch.dtype, description="The data type of the prompt embeddings"),
            OutputParam.template("prompt_embeds", note="batch-expanded"),
            OutputParam.template("prompt_embeds_mask", note="batch-expanded, None when nothing is padded"),
            OutputParam.template("negative_prompt_embeds", note="batch-expanded"),
            OutputParam.template("negative_prompt_embeds_mask", note="batch-expanded, None when nothing is padded"),
            OutputParam(
                name="image_pad_mask",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the prompt embeddings (batch-expanded)",
            ),
            OutputParam(
                name="negative_image_pad_mask",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the negative prompt embeddings (batch-expanded)",
            ),
        ]

    @staticmethod
    def check_inputs(
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        image_pad_mask,
        negative_image_pad_mask,
    ):
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("`negative_prompt_embeds_mask` is required when `negative_prompt_embeds` is not None")
        if negative_prompt_embeds is None and negative_prompt_embeds_mask is not None:
            raise ValueError("cannot pass `negative_prompt_embeds_mask` without `negative_prompt_embeds`")
        if negative_prompt_embeds is not None and negative_image_pad_mask is None:
            raise ValueError("`negative_image_pad_mask` is required when `negative_prompt_embeds` is not None")

        batch_size = prompt_embeds.shape[0]
        for name, value in (
            ("prompt_embeds_mask", prompt_embeds_mask),
            ("image_pad_mask", image_pad_mask),
            ("negative_prompt_embeds", negative_prompt_embeds),
            ("negative_prompt_embeds_mask", negative_prompt_embeds_mask),
            ("negative_image_pad_mask", negative_image_pad_mask),
        ):
            if value is not None and value.shape[0] != batch_size:
                raise ValueError(f"`{name}` must have the same batch size as `prompt_embeds`")

    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(
            prompt_embeds=block_state.prompt_embeds,
            prompt_embeds_mask=block_state.prompt_embeds_mask,
            negative_prompt_embeds=block_state.negative_prompt_embeds,
            negative_prompt_embeds_mask=block_state.negative_prompt_embeds_mask,
            image_pad_mask=block_state.image_pad_mask,
            negative_image_pad_mask=block_state.negative_image_pad_mask,
        )

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        for name in (
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "image_pad_mask",
            "negative_image_pad_mask",
        ):
            value = getattr(block_state, name)
            if value is not None:
                value = value.repeat_interleave(block_state.num_images_per_prompt, dim=0)
            setattr(block_state, name, value)

        # Without padding there is nothing to mask, and a mask that carries no information costs the attention
        # backends that reject one outright.
        if block_state.prompt_embeds_mask.all():
            block_state.prompt_embeds_mask = None
        if block_state.negative_prompt_embeds_mask is not None and block_state.negative_prompt_embeds_mask.all():
            block_state.negative_prompt_embeds_mask = None

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21AdditionalInputsStep(ModularPipelineBlocks):
    """
    Input processing step for image-conditioned generation that:
        1. Records the pixel height/width of each condition image from its latents
        2. Packs each image latent, concatenates them along the sequence dimension and expands the batch
        3. Defaults `height`/`width` to the size of the last condition image

      This block should be placed after the encoder steps and the text input step.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_latents (`list`):
              Normalized latents of each condition image. Can be generated from vae_encoder step.

      Outputs:
          image_height (`list`):
              The pixel heights of the condition images, calculated from the image latents
          image_width (`list`):
              The pixel widths of the condition images, calculated from the image latents
          height (`int`):
              if not provided, updated to the last image height
          width (`int`):
              if not provided, updated to the last image width
          image_latents (`Tensor`):
              Condition image latents packed, concatenated along the sequence dimension and batch-expanded
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Input processing step for image-conditioned generation that:\n"
            "  1. Records the pixel height/width of each condition image from its latents\n"
            "  2. Packs each image latent, concatenates them along the sequence dimension and expands the batch\n"
            "  3. Defaults `height`/`width` to the size of the last condition image\n\n"
            "This block should be placed after the encoder steps and the text input step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("batch_size"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam(
                name="image_latents",
                required=True,
                type_hint=list[torch.Tensor],
                description="Normalized latents of each condition image. Can be generated from vae_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="image_height",
                type_hint=list[int],
                description="The pixel heights of the condition images, calculated from the image latents",
            ),
            OutputParam(
                name="image_width",
                type_hint=list[int],
                description="The pixel widths of the condition images, calculated from the image latents",
            ),
            OutputParam(name="height", type_hint=int, description="if not provided, updated to the last image height"),
            OutputParam(name="width", type_hint=int, description="if not provided, updated to the last image width"),
            OutputParam(
                name="image_latents",
                type_hint=torch.Tensor,
                description="Condition image latents packed, concatenated along the sequence dimension and batch-expanded",
            ),
        ]

    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        image_latents = block_state.image_latents
        if not isinstance(image_latents, list):
            image_latents = [image_latents]

        image_heights = []
        image_widths = []
        packed_image_latents = []
        for i, image_latent in enumerate(image_latents):
            latent_height, latent_width = image_latent.shape[-2:]
            image_heights.append(latent_height * components.vae_scale_factor)
            image_widths.append(latent_width * components.vae_scale_factor)
            packed_image_latents.append(
                repeat_tensor_to_batch_size(
                    input_name=f"image_latents[{i}]",
                    input_tensor=pack_latents(image_latent),
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    batch_size=block_state.batch_size,
                )
            )

        block_state.image_latents = torch.cat(packed_image_latents, dim=1)
        block_state.image_height = image_heights
        block_state.image_width = image_widths
        block_state.height = block_state.height or image_heights[-1]
        block_state.width = block_state.width or image_widths[-1]

        self.set_block_state(state, block_state)
        return components, state
