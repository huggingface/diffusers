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


import torch

from ...models import QwenImageMultiControlNetModel
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageLayeredPachifier, QwenImageModularPipeline, QwenImagePachifier


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


# auto_docstring
class QwenImageTextInputsStep(ModularPipelineBlocks):
    """
    Text input processing step that standardizes text embeddings for the pipeline.
      This step:
        1. Determines `batch_size` and `dtype` based on `prompt_embeds`
        2. Ensures all text embeddings have consistent batch sizes (batch_size * num_images_per_prompt)

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

      Outputs:
          batch_size (`int`):
              The batch size of the prompt embeddings
          dtype (`dtype`):
              The data type of the prompt embeddings
          prompt_embeds (`Tensor`):
              The prompt embeddings. (batch-expanded)
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask. (batch-expanded)
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings. (batch-expanded)
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask. (batch-expanded)
    """

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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("prompt_embeds"),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds"),
            InputParam.template("negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="batch_size", type_hint=int, description="The batch size of the prompt embeddings"),
            OutputParam(name="dtype", type_hint=torch.dtype, description="The data type of the prompt embeddings"),
            OutputParam.template("prompt_embeds", note="batch-expanded"),
            OutputParam.template("prompt_embeds_mask", note="batch-expanded"),
            OutputParam.template("negative_prompt_embeds", note="batch-expanded"),
            OutputParam.template("negative_prompt_embeds_mask", note="batch-expanded"),
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


# auto_docstring
class QwenImageAdditionalInputsStep(ModularPipelineBlocks):
    """
    Input processing step that:
        1. For image latent inputs: Updates height/width if None, patchifies, and expands batch size
        2. For additional batch inputs: Expands batch dimensions to match final batch size

      Configured inputs:
        - Image latent inputs: ['image_latents']

      This block should be placed after the encoder steps and the text input step.

      Components:
          pachifier (`QwenImagePachifier`)

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
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.

      Outputs:
          image_height (`int`):
              The image height calculated from the image latents dimension
          image_width (`int`):
              The image width calculated from the image latents dimension
          height (`int`):
              if not provided, updated to image height
          width (`int`):
              if not provided, updated to image width
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (patchified and
              batch-expanded)
    """

    model_name = "qwenimage"

    def __init__(
        self,
        image_latent_inputs: list[InputParam] | None = None,
        additional_batch_inputs: list[InputParam] | None = None,
    ):
        # by default, process `image_latents`
        if image_latent_inputs is None:
            image_latent_inputs = [InputParam.template("image_latents")]
        if additional_batch_inputs is None:
            additional_batch_inputs = []

        if not isinstance(image_latent_inputs, list):
            raise ValueError(f"image_latent_inputs must be a list, but got {type(image_latent_inputs)}")
        else:
            for input_param in image_latent_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(f"image_latent_inputs must be a list of InputParam, but got {type(input_param)}")

        if not isinstance(additional_batch_inputs, list):
            raise ValueError(f"additional_batch_inputs must be a list, but got {type(additional_batch_inputs)}")
        else:
            for input_param in additional_batch_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(
                        f"additional_batch_inputs must be a list of InputParam, but got {type(input_param)}"
                    )

        self._image_latent_inputs = image_latent_inputs
        self._additional_batch_inputs = additional_batch_inputs
        super().__init__()

    @property
    def description(self) -> str:
        summary_section = (
            "Input processing step that:\n"
            "  1. For image latent inputs: Updates height/width if None, patchifies, and expands batch size\n"
            "  2. For additional batch inputs: Expands batch dimensions to match final batch size"
        )

        inputs_info = ""
        if self._image_latent_inputs or self._additional_batch_inputs:
            inputs_info = "\n\nConfigured inputs:"
            if self._image_latent_inputs:
                inputs_info += f"\n  - Image latent inputs: {[p.name for p in self._image_latent_inputs]}"
            if self._additional_batch_inputs:
                inputs_info += f"\n  - Additional batch inputs: {[p.name for p in self._additional_batch_inputs]}"

        placement_section = "\n\nThis block should be placed after the encoder steps and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("batch_size"),
            InputParam.template("height"),
            InputParam.template("width"),
        ]
        # default is `image_latents`
        inputs += self._image_latent_inputs + self._additional_batch_inputs

        return inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        outputs = [
            OutputParam(
                name="image_height",
                type_hint=int,
                description="The image height calculated from the image latents dimension",
            ),
            OutputParam(
                name="image_width",
                type_hint=int,
                description="The image width calculated from the image latents dimension",
            ),
        ]

        # `height`/`width` are not new outputs, but they will be updated if any image latent inputs are provided
        if len(self._image_latent_inputs) > 0:
            outputs.append(
                OutputParam(name="height", type_hint=int, description="if not provided, updated to image height")
            )
            outputs.append(
                OutputParam(name="width", type_hint=int, description="if not provided, updated to image width")
            )

        # image latent inputs are modified in place (patchified and batch-expanded)
        for input_param in self._image_latent_inputs:
            outputs.append(
                OutputParam(
                    name=input_param.name,
                    type_hint=input_param.type_hint,
                    description=input_param.description + " (patchified and batch-expanded)",
                )
            )

        # additional batch inputs (batch-expanded only)
        for input_param in self._additional_batch_inputs:
            outputs.append(
                OutputParam(
                    name=input_param.name,
                    type_hint=input_param.type_hint,
                    description=input_param.description + " (batch-expanded)",
                )
            )

        return outputs

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Process image latent inputs
        for input_param in self._image_latent_inputs:
            image_latent_input_name = input_param.name
            image_latent_tensor = getattr(block_state, image_latent_input_name)
            if image_latent_tensor is None:
                continue

            # 1. Calculate height/width from latents and update if not provided
            height, width = calculate_dimension_from_latents(image_latent_tensor, components.vae_scale_factor)
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

            if not hasattr(block_state, "image_height"):
                block_state.image_height = height
            if not hasattr(block_state, "image_width"):
                block_state.image_width = width

            # 2. Patchify
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
        for input_param in self._additional_batch_inputs:
            input_name = input_param.name
            input_tensor = getattr(block_state, input_name)
            if input_tensor is None:
                continue

            input_tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=input_tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_name, input_tensor)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageEditPlusAdditionalInputsStep(ModularPipelineBlocks):
    """
    Input processing step for Edit Plus that:
        1. For image latent inputs (list): Collects heights/widths, patchifies each, concatenates, expands batch
        2. For additional batch inputs: Expands batch dimensions to match final batch size
        Height/width defaults to last image in the list.

      Configured inputs:
        - Image latent inputs: ['image_latents']

      This block should be placed after the encoder steps and the text input step.

      Components:
          pachifier (`QwenImagePachifier`)

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
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.

      Outputs:
          image_height (`list`):
              The image heights calculated from the image latents dimension
          image_width (`list`):
              The image widths calculated from the image latents dimension
          height (`int`):
              if not provided, updated to image height
          width (`int`):
              if not provided, updated to image width
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (patchified,
              concatenated, and batch-expanded)
    """

    model_name = "qwenimage-edit-plus"

    def __init__(
        self,
        image_latent_inputs: list[InputParam] | None = None,
        additional_batch_inputs: list[InputParam] | None = None,
    ):
        if image_latent_inputs is None:
            image_latent_inputs = [InputParam.template("image_latents")]
        if additional_batch_inputs is None:
            additional_batch_inputs = []

        if not isinstance(image_latent_inputs, list):
            raise ValueError(f"image_latent_inputs must be a list, but got {type(image_latent_inputs)}")
        else:
            for input_param in image_latent_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(f"image_latent_inputs must be a list of InputParam, but got {type(input_param)}")

        if not isinstance(additional_batch_inputs, list):
            raise ValueError(f"additional_batch_inputs must be a list, but got {type(additional_batch_inputs)}")
        else:
            for input_param in additional_batch_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(
                        f"additional_batch_inputs must be a list of InputParam, but got {type(input_param)}"
                    )

        self._image_latent_inputs = image_latent_inputs
        self._additional_batch_inputs = additional_batch_inputs
        super().__init__()

    @property
    def description(self) -> str:
        summary_section = (
            "Input processing step for Edit Plus that:\n"
            "  1. For image latent inputs (list): Collects heights/widths, patchifies each, concatenates, expands batch\n"
            "  2. For additional batch inputs: Expands batch dimensions to match final batch size\n"
            "  Height/width defaults to last image in the list."
        )

        inputs_info = ""
        if self._image_latent_inputs or self._additional_batch_inputs:
            inputs_info = "\n\nConfigured inputs:"
            if self._image_latent_inputs:
                inputs_info += f"\n  - Image latent inputs: {[p.name for p in self._image_latent_inputs]}"
            if self._additional_batch_inputs:
                inputs_info += f"\n  - Additional batch inputs: {[p.name for p in self._additional_batch_inputs]}"

        placement_section = "\n\nThis block should be placed after the encoder steps and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("batch_size"),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

        # default is `image_latents`
        inputs += self._image_latent_inputs + self._additional_batch_inputs

        return inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        outputs = [
            OutputParam(
                name="image_height",
                type_hint=list[int],
                description="The image heights calculated from the image latents dimension",
            ),
            OutputParam(
                name="image_width",
                type_hint=list[int],
                description="The image widths calculated from the image latents dimension",
            ),
        ]

        # `height`/`width` are updated if any image latent inputs are provided
        if len(self._image_latent_inputs) > 0:
            outputs.append(
                OutputParam(name="height", type_hint=int, description="if not provided, updated to image height")
            )
            outputs.append(
                OutputParam(name="width", type_hint=int, description="if not provided, updated to image width")
            )

        # image latent inputs are modified in place (patchified, concatenated, and batch-expanded)
        for input_param in self._image_latent_inputs:
            outputs.append(
                OutputParam(
                    name=input_param.name,
                    type_hint=input_param.type_hint,
                    description=input_param.description + " (patchified, concatenated, and batch-expanded)",
                )
            )

        # additional batch inputs (batch-expanded only)
        for input_param in self._additional_batch_inputs:
            outputs.append(
                OutputParam(
                    name=input_param.name,
                    type_hint=input_param.type_hint,
                    description=input_param.description + " (batch-expanded)",
                )
            )

        return outputs

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Process image latent inputs
        for input_param in self._image_latent_inputs:
            image_latent_input_name = input_param.name
            image_latent_tensor = getattr(block_state, image_latent_input_name)
            if image_latent_tensor is None:
                continue

            is_list = isinstance(image_latent_tensor, list)
            if not is_list:
                image_latent_tensor = [image_latent_tensor]

            image_heights = []
            image_widths = []
            packed_image_latent_tensors = []

            for i, img_latent_tensor in enumerate(image_latent_tensor):
                # 1. Calculate height/width from latents
                height, width = calculate_dimension_from_latents(img_latent_tensor, components.vae_scale_factor)
                image_heights.append(height)
                image_widths.append(width)

                # 2. Patchify
                img_latent_tensor = components.pachifier.pack_latents(img_latent_tensor)

                # 3. Expand batch size
                img_latent_tensor = repeat_tensor_to_batch_size(
                    input_name=f"{image_latent_input_name}[{i}]",
                    input_tensor=img_latent_tensor,
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    batch_size=block_state.batch_size,
                )
                packed_image_latent_tensors.append(img_latent_tensor)

            # Concatenate all packed latents along dim=1
            packed_image_latent_tensors = torch.cat(packed_image_latent_tensors, dim=1)

            # Output lists of heights/widths
            block_state.image_height = image_heights
            block_state.image_width = image_widths

            # Default height/width from last image
            block_state.height = block_state.height or image_heights[-1]
            block_state.width = block_state.width or image_widths[-1]

            setattr(block_state, image_latent_input_name, packed_image_latent_tensors)

        # Process additional batch inputs (only batch expansion)
        for input_param in self._additional_batch_inputs:
            input_name = input_param.name
            input_tensor = getattr(block_state, input_name)
            if input_tensor is None:
                continue

            input_tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=input_tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_name, input_tensor)

        self.set_block_state(state, block_state)
        return components, state


# same as QwenImageAdditionalInputsStep, but with layered pachifier.


# auto_docstring
class QwenImageLayeredAdditionalInputsStep(ModularPipelineBlocks):
    """
    Input processing step for Layered that:
        1. For image latent inputs: Updates height/width if None, patchifies with layered pachifier, and expands batch
           size
        2. For additional batch inputs: Expands batch dimensions to match final batch size

      Configured inputs:
        - Image latent inputs: ['image_latents']

      This block should be placed after the encoder steps and the text input step.

      Components:
          pachifier (`QwenImageLayeredPachifier`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.

      Outputs:
          image_height (`int`):
              The image height calculated from the image latents dimension
          image_width (`int`):
              The image width calculated from the image latents dimension
          height (`int`):
              if not provided, updated to image height
          width (`int`):
              if not provided, updated to image width
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (patchified
              with layered pachifier and batch-expanded)
    """

    model_name = "qwenimage-layered"

    def __init__(
        self,
        image_latent_inputs: list[InputParam] | None = None,
        additional_batch_inputs: list[InputParam] | None = None,
    ):
        if image_latent_inputs is None:
            image_latent_inputs = [InputParam.template("image_latents")]
        if additional_batch_inputs is None:
            additional_batch_inputs = []

        if not isinstance(image_latent_inputs, list):
            raise ValueError(f"image_latent_inputs must be a list, but got {type(image_latent_inputs)}")
        else:
            for input_param in image_latent_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(f"image_latent_inputs must be a list of InputParam, but got {type(input_param)}")

        if not isinstance(additional_batch_inputs, list):
            raise ValueError(f"additional_batch_inputs must be a list, but got {type(additional_batch_inputs)}")
        else:
            for input_param in additional_batch_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(
                        f"additional_batch_inputs must be a list of InputParam, but got {type(input_param)}"
                    )

        self._image_latent_inputs = image_latent_inputs
        self._additional_batch_inputs = additional_batch_inputs
        super().__init__()

    @property
    def description(self) -> str:
        summary_section = (
            "Input processing step for Layered that:\n"
            "  1. For image latent inputs: Updates height/width if None, patchifies with layered pachifier, and expands batch size\n"
            "  2. For additional batch inputs: Expands batch dimensions to match final batch size"
        )

        inputs_info = ""
        if self._image_latent_inputs or self._additional_batch_inputs:
            inputs_info = "\n\nConfigured inputs:"
            if self._image_latent_inputs:
                inputs_info += f"\n  - Image latent inputs: {[p.name for p in self._image_latent_inputs]}"
            if self._additional_batch_inputs:
                inputs_info += f"\n  - Additional batch inputs: {[p.name for p in self._additional_batch_inputs]}"

        placement_section = "\n\nThis block should be placed after the encoder steps and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImageLayeredPachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("batch_size"),
        ]
        # default is `image_latents`

        inputs += self._image_latent_inputs + self._additional_batch_inputs

        return inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        outputs = [
            OutputParam(
                name="image_height",
                type_hint=int,
                description="The image height calculated from the image latents dimension",
            ),
            OutputParam(
                name="image_width",
                type_hint=int,
                description="The image width calculated from the image latents dimension",
            ),
        ]

        if len(self._image_latent_inputs) > 0:
            outputs.append(
                OutputParam(name="height", type_hint=int, description="if not provided, updated to image height")
            )
            outputs.append(
                OutputParam(name="width", type_hint=int, description="if not provided, updated to image width")
            )

        # Add outputs for image latent inputs (patchified with layered pachifier and batch-expanded)
        for input_param in self._image_latent_inputs:
            outputs.append(
                OutputParam(
                    name=input_param.name,
                    type_hint=input_param.type_hint,
                    description=input_param.description + " (patchified with layered pachifier and batch-expanded)",
                )
            )

        # Add outputs for additional batch inputs (batch-expanded only)
        for input_param in self._additional_batch_inputs:
            outputs.append(
                OutputParam(
                    name=input_param.name,
                    type_hint=input_param.type_hint,
                    description=input_param.description + " (batch-expanded)",
                )
            )

        return outputs

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Process image latent inputs
        for input_param in self._image_latent_inputs:
            image_latent_input_name = input_param.name
            image_latent_tensor = getattr(block_state, image_latent_input_name)
            if image_latent_tensor is None:
                continue

            # 1. Calculate height/width from latents and update if not provided
            # Layered latents are (B, layers, C, H, W)
            height = image_latent_tensor.shape[3] * components.vae_scale_factor
            width = image_latent_tensor.shape[4] * components.vae_scale_factor
            block_state.height = height
            block_state.width = width

            if not hasattr(block_state, "image_height"):
                block_state.image_height = height
            if not hasattr(block_state, "image_width"):
                block_state.image_width = width

            # 2. Patchify with layered pachifier
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
        for input_param in self._additional_batch_inputs:
            input_name = input_param.name
            input_tensor = getattr(block_state, input_name)
            if input_tensor is None:
                continue

            input_tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=input_tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_name, input_tensor)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageControlNetInputsStep(ModularPipelineBlocks):
    """
    prepare the `control_image_latents` for controlnet. Insert after all the other inputs steps.

      Inputs:
          control_image_latents (`Tensor`):
              The control image latents to use for the denoising process. Can be generated in controlnet vae encoder
              step.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.

      Outputs:
          control_image_latents (`Tensor`):
              The control image latents (patchified and batch-expanded).
          height (`int`):
              if not provided, updated to control image height
          width (`int`):
              if not provided, updated to control image width
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "prepare the `control_image_latents` for controlnet. Insert after all the other inputs steps."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="control_image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The control image latents to use for the denoising process. Can be generated in controlnet vae encoder step.",
            ),
            InputParam.template("batch_size"),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="control_image_latents",
                type_hint=torch.Tensor,
                description="The control image latents (patchified and batch-expanded).",
            ),
            OutputParam(name="height", type_hint=int, description="if not provided, updated to control image height"),
            OutputParam(name="width", type_hint=int, description="if not provided, updated to control image width"),
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
