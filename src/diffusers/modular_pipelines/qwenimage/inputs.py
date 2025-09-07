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

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam, ComponentSpec
from .modular_pipeline import QwenImageModularPipeline, QwenImagePachifier
from ...models import QwenImageMultiControlNetModel

from ...utils.torch_utils import unwrap_module


def repeat_tensor_to_final_batch_size(
    input_name: str, 
    input_tensor: torch.Tensor, 
    batch_size: int,
    num_images_per_prompt: int = 1, 
) -> torch.Tensor:
    
    # make sure input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError(f"`{input_name}` must be a tensor")
    
    # make sure input tensor e.g. image_latents has batch size 1 or batch_size same as prompts
    if input_tensor.shape[0] == 1:
        repeat_by = batch_size * num_images_per_prompt
    elif input_tensor.shape[0] == batch_size:
        repeat_by =  num_images_per_prompt
    else:
        raise ValueError(f"`{input_name}` must have have batch size 1 or {batch_size}, but got {input_tensor.shape[0]}")
    
    # expand the tensor to match the batch_size * num_images_per_prompt
    input_tensor = input_tensor.repeat_interleave(repeat_by, dim=0)
    
    return input_tensor


def calculate_image_dimension_from_latents(latents: torch.Tensor, vae_scale_factor: int) -> Tuple[int, int]:

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


# YiYi TODO: combine this and the image input step
class QwenImageBatchInputsDynamicStep(ModularPipelineBlocks):

    model_name = "qwenimage"

    def __init__(
        self, 
        batch_inputs: List[str] = [], 
        ):
        """Initialize a configurable step that expands batch dimensions for additional conditional inputs.

        This step adjusts batch dimensions for additional conditional inputs to match the final batch size
        (batch_size * num_images_per_prompt). It should be placed after the default text input processing step
        when you have additional inputs that need batch size alignment.

        This is a dynamic block that allows you to configure which additional inputs to process.

        Args:
            batch_inputs (List[str], optional): Names of additional conditional input tensors to adjust batch size.
                These tensors will have their batch dimensions adjusted to match the final batch size. Can be a single
                string or list of strings. Defaults to []. Examples: ["image_latents"], ["control_image_latents",
                "reference_image"]

        Examples:
            # Configure to expand batch dimension for `image_latents`
            QwenImageBatchInputsDynamicStep(batch_inputs=["image_latents"])
            
            # Configure to expand batch dimension for multiple inputs
            QwenImageBatchInputsDynamicStep(batch_inputs=["image_latents", "control_image_latents"])
        """
        if not isinstance(batch_inputs, list):
            batch_inputs = [batch_inputs]

        self._batch_inputs = batch_inputs
        super().__init__()

    @property
    def description(self) -> str:
        # Functionality section
        summary_section = (
            "Input processing step that expands batch dimensions for additional conditional inputs.\n"
            "This step ensures all specified inputs have consistent batch sizes for the rest of the pipeline."
        )

        # Batch alignment inputs info
        inputs_info = ""
        if self._batch_inputs:
            inputs_info = f"\n\nInputs to process: {self._batch_inputs}"

        # Placement guidance
        placement_section = "\n\nThis block should be placed after all the encoders and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def inputs(self) -> List[InputParam]:
        inputs = [
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="batch_size", required=True),
        ]
        for input_name in self._batch_inputs:
            inputs.append(InputParam(name=input_name))

        return inputs

    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)


        # optionally, expand additional inputs to match the batch size of prompts

        for input_name in self._batch_inputs:
            input_tensor = getattr(block_state, input_name)

            if input_tensor is None:
                continue

            input_tensor = repeat_tensor_to_final_batch_size(
                input_name=input_name,
                input_tensor=input_tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_name, input_tensor)

        self.set_block_state(state, block_state)

        return components, state


class QwenImageImageInputsDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    def __init__(self, image_latent_inputs: List[str] = ["image_latents"]):
        """Initialize a configurable step that processes image latent inputs.

        This step handles two main tasks:
        1. Updates height/width if they are None (calculated from specified image latent dimensions)
        2. Patchifies the specified image latents for the transformer model

        This is a dynamic block that allows you to configure which image latent inputs to process. By default, it will process `image_latents`.

        Args:
            image_latent_inputs (List[str], optional): Names of image latent tensors to process.
                These tensors will be used to determine height/width if not provided, and will be patchified.
                Can be a single string or list of strings. Defaults to ["image_latents"]. 
                Examples: ["image_latents"], ["control_image_latents"]

        Examples:
            # Configure to process `image_latents`
            QwenImageImageInputsDynamicStep(image_latent_inputs=["image_latents"])
            
            # Configure to process multiple inputs
            QwenImageImageInputsDynamicStep(image_latent_inputs=["image_latents", "control_image_latents"])
        """
        if not isinstance(image_latent_inputs, list):
            image_latent_inputs = [image_latent_inputs]

        self._image_latent_inputs = image_latent_inputs
        super().__init__()

    @property
    def description(self) -> str:
        # Functionality section
        summary_section = (
            "Image latent processing step that:\n"
            "  1. Updates `height` and `width` if they are None (calculated from image latent dimensions)\n"
            "  2. Patchifies the image latents for the transformer model"
        )

        # Image latent inputs info
        inputs_info = ""
        if self._image_latent_inputs:
            inputs_info = f"\n\nConfigured to process image latents: {self._image_latent_inputs}"

        # Placement guidance
        placement_section = "\n\nThis block should be placed after the encoder steps and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def inputs(self) -> List[InputParam]:

        inputs = [InputParam(name="height"), InputParam(name="width")]

        for image_latent_input_name in self._image_latent_inputs:
            inputs.append(InputParam(name=image_latent_input_name))

        return inputs

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]


    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        for image_latent_input_name in self._image_latent_inputs:

            image_latent_tensor = getattr(block_state, image_latent_input_name)
            if image_latent_tensor is None:
                continue

            height, width = calculate_image_dimension_from_latents(image_latent_tensor, components.vae_scale_factor)

            # update height and width based on image latent dimensions if not provided
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

            # pack the image latent tensor
            image_latent_tensor = components.pachifier.pack_latents(image_latent_tensor)
            setattr(block_state, image_latent_input_name, image_latent_tensor)

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
                height, width = calculate_image_dimension_from_latents(control_image_latents_, components.vae_scale_factor)
                block_state.height = block_state.height or height
                block_state.width = block_state.width or width

                # 2. pack
                control_image_latents_ = components.pachifier.pack_latents(control_image_latents_)

                # 3. repeat to match the batch size
                control_image_latents_ = repeat_tensor_to_final_batch_size(
                    input_name=f"control_image_latents[{i}]",
                    input_tensor=control_image_latents_,
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    batch_size=block_state.batch_size,
                )

                control_image_latents.append(control_image_latents_)

            block_state.control_image_latents = control_image_latents   
        
        else:
            # 1. update height/width if not provided
            height, width = calculate_image_dimension_from_latents(block_state.control_image_latents, components.vae_scale_factor)
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

            # 2. pack
            block_state.control_image_latents = components.pachifier.pack_latents(block_state.control_image_latents)
            
            # 3. repeat to match the batch size
            block_state.control_image_latents = repeat_tensor_to_final_batch_size(
                input_name="control_image_latents",
                input_tensor=block_state.control_image_latents,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            block_state.control_image_latents = block_state.control_image_latents

        self.set_block_state(state, block_state)

        return components, state