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

import inspect
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor, InpaintProcessor
from ...models import QwenImageControlNetModel, QwenImageMultiControlNetModel
from ...pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor, unwrap_module
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline



class QwenImageResizeDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to target height and width"

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
        additional_inputs = []
        for input_name in self.image_input_names:
            additional_inputs.append(InputParam(name=input_name, description="The image to resize"))
        return [
            InputParam(name="height"),
            InputParam(name="width"),
        ] + additional_inputs

    def __init__(self, input_names: List[str] = ["image"]):
        if not isinstance(input_names, list):
            input_names = [input_names]
        self.image_input_names = input_names
        super().__init__()

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):
        if height is not None and height % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * 2} but is {height}")

        if width is not None and width % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * 2} but is {width}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        self.check_inputs(
            height=block_state.height,
            width=block_state.width,
            vae_scale_factor=components.vae_scale_factor,
        )
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        for input_name in self.image_input_names:
            image_input = getattr(block_state, input_name)
            if image_input is None:
                continue
            if not isinstance(image_input, list):
                image_input = [image_input]

            resized_images = [
                components.image_processor.resize(image, height=block_state.height, width=block_state.width) for image in image_input
            ]
            setattr(block_state, input_name, resized_images)

        self.set_block_state(state, block_state)
        return components, state


class QwenImageEditResizeStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to the target area while maintaining the aspect ratio."

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
            InputParam(name="image", required=True, type_hint=torch.Tensor, description="The image to resize"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = block_state.image
        if not isinstance(images, list):
            images = [images]

        image_width, image_height = images[0].size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_width / image_height)

        resized_images = [
            components.image_processor.resize(image, height=calculated_height, width=calculated_width)
            for image in images
        ]

        block_state.image = resized_images
        self.set_block_state(state, block_state)
        return components, state


class QwenImageInpaintProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Mask step that resize the image to the target area while maintaining the aspect ratio."


    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image", required=True),
            InputParam("mask_image", required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("padding_mask_crop"),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="original_image", type_hint=torch.Tensor, description="The original image"),
            OutputParam(name="original_mask", type_hint=torch.Tensor, description="The original mask"),
            OutputParam(name="crop_coords", type_hint=List[Tuple[int, int]], description="The crop coordinates to use for the preprocess/postprocess of the image and mask",),
        ]
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        block_state.image, block_state.mask_image, postprocessing_kwargs = components.image_processor.preprocess(
            image=block_state.image,
            mask=block_state.mask_image,
            height=block_state.height,
            width=block_state.width,
            padding_mask_crop=block_state.padding_mask_crop,
        )

        if postprocessing_kwargs:
            block_state.original_image = postprocessing_kwargs["original_image"]
            block_state.original_mask = postprocessing_kwargs["original_mask"]
            block_state.crop_coords = postprocessing_kwargs["crops_coords"]
        
        self.set_block_state(state, block_state)
        return components, state

    

class QwenImageExpandTextInputsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts input tensor shapes based on `batch_size` (number of prompts) and `num_images_per_prompt`\n\n"
            "All input tensors are expected to have either batch_size=1 or match the batch_size\n"
            "of prompt_embeds. The tensors will be duplicated across the batch dimension to\n"
            "have a final batch_size of batch_size * num_images_per_prompt."
        )

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
    def check_inputs(prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask):
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


class QwenImageExpandInputsDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  Adjusts input tensor shapes based on `batch_size` (number of prompts) and `num_images_per_prompt`\n\n"
            "All input tensors are expected to have either batch_size=1 or match the batch_size\n"
            "of prompt_embeds. The tensors will be duplicated across the batch dimension to\n"
            "have a final batch_size of batch_size * num_images_per_prompt."
        )

    @property
    def inputs(self) -> List[InputParam]:
        additional_inputs = []
        for input_name in self.tensor_input_names:
            additional_inputs.append(InputParam(name=input_name, required=True))

        return [
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(
                name="batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
        ] + additional_inputs

    @staticmethod
    def check_inputs(tensor_input, tensor_input_name, batch_size):
        if tensor_input is not None and tensor_input.shape[0] != 1 and tensor_input.shape[0] != batch_size:
            raise ValueError(
                f"`{tensor_input_name}` must have have batch size 1 or {batch_size}, but got {tensor_input.shape[0]}"
            )

    def __init__(self, input_names: List[str] = ["image_latents"]):
        if not isinstance(input_names, list):
            input_names = [input_names]
        self.tensor_input_names = input_names
        super().__init__()

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        final_batch_size = block_state.batch_size * block_state.num_images_per_prompt

        for input_name in self.tensor_input_names:
            tensor_input = getattr(block_state, input_name)
            # make sure input tensor e.g. image_latents has batch size 1 or batch_size same as prompts
            self.check_inputs(
                tensor_input=tensor_input,
                tensor_input_name=input_name,
                batch_size=block_state.batch_size,
            )
            tensor_input = tensor_input.repeat(final_batch_size // tensor_input.shape[0], 1, 1, 1, 1)
            setattr(block_state, input_name, tensor_input)

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
                "image_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("images", required=True, description= "the generated image from decoders step"),
            InputParam("original_image"),
            InputParam("original_mask"),
            InputParam("crop_coords"),
        ]
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        block_state.images = components.image_processor.postprocess(
            image=block_state.images,
            original_image=block_state.original_image,
            original_mask=block_state.original_mask,
            crops_coords=block_state.crop_coords,
        )

        self.set_block_state(state, block_state)
        return components, state