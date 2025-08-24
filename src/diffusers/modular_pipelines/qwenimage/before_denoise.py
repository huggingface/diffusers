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

from typing import List, Optional, Union, Tuple

import numpy as np
import torch
import inspect

from ...image_processor import VaeImageProcessor
from ...configuration_utils import FrozenDict

from ...pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline

from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor



# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


class QwenImageImageResizeStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to the target area while maintaining the aspect ratio"
    
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("image_processor", VaeImageProcessor, config=FrozenDict({"vae_scale_factor": 16}), default_creation_method="from_config"),
        ]
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="image", required=True, type_hint=torch.Tensor, description="The image to resize"),
        ]
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)


        if not isinstance(block_state.image, list):
            block_state.image = [block_state.image]
        
        image_width, image_height = block_state.image[0].size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_width / image_height)

        block_state.image = components.image_processor.resize(block_state.image, height=calculated_height, width=calculated_width)
        self.set_block_state(state, block_state)
        return components, state


class QwenImageInputStep(ModularPipelineBlocks):

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
            "  3. If `image_latents` is provided and `height` and `width` are not provided, it will update the `height` and `width` parameters."
        )
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="prompt_embeds", required=True, kwargs_type="guider_input_fields"),
            InputParam(name="prompt_embeds_mask", required=True, kwargs_type="guider_input_fields"),
            InputParam(name="negative_prompt_embeds", kwargs_type="guider_input_fields"),
            InputParam(name="negative_prompt_embeds_mask", kwargs_type="guider_input_fields"),
            InputParam(name="image_latents"),
            InputParam(name="height"),
            InputParam(name="width"),
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
    def check_inputs(prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, image_latents):
        
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("`negative_prompt_embeds_mask` is required when `negative_prompt_embeds` is not None")
        
        if negative_prompt_embeds is None and negative_prompt_embeds_mask is not None:
            raise ValueError("cannot pass `negative_prompt_embeds_mask` without `negative_prompt_embeds`")
        
        if prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`prompt_embeds_mask` must have the same batch size as `prompt_embeds`")
        
        elif negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`negative_prompt_embeds` must have the same batch size as `prompt_embeds`")
        
        elif negative_prompt_embeds_mask is not None and negative_prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`negative_prompt_embeds_mask` must have the same batch size as `prompt_embeds`")

        if image_latents is not None and image_latents.shape[0] != 1 and image_latents.shape[0] != prompt_embeds.shape[0]:
            raise ValueError(f"`image_latents` must have have batch size 1 or {prompt_embeds.shape[0]}, but got {image_latents.shape[0]}")
        

    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:

        block_state = self.get_block_state(state)

        self.check_inputs(
            prompt_embeds=block_state.prompt_embeds,
            prompt_embeds_mask=block_state.prompt_embeds_mask,
            negative_prompt_embeds=block_state.negative_prompt_embeds,
            negative_prompt_embeds_mask=block_state.negative_prompt_embeds_mask,
            image_latents=block_state.image_latents,
        )

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape

        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1)
        
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len)
        
        if block_state.negative_prompt_embeds is not None:
            _, seq_len, _ = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1)
            
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.repeat(1, block_state.num_images_per_prompt, 1)
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len)
        
        if block_state.image_latents is not None:
            final_batch_size = block_state.batch_size * block_state.num_images_per_prompt
            block_state.image_latents = block_state.image_latents.repeat(
                final_batch_size // block_state.image_latents.shape[0], 1, 1, 1, 1
            )

            height_image_latent, width_image_latent = block_state.image_latents.shape[3:]

            if block_state.height is None:
                block_state.height = height_image_latent * components.vae_scale_factor
            if block_state.width is None:
                block_state.width = width_image_latent * components.vae_scale_factor
           
        self.set_block_state(state, block_state)

        return components, state


class QwenImagePrepareLatentsStep(ModularPipelineBlocks):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-image generation process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="height"),
            InputParam(name="width"),
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="generator"),
            InputParam(
                name="batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam(name="dtype", required=True, type_hint=torch.dtype, description="The dtype of the model inputs, can be generated in input step."),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"),
        ]

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):

        if height is not None and height % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * 2} but is {height}")

        if width is not None and width % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * 2} but is {width}")
    
    @staticmethod
    def prepare_latents(
        batch_size,
        num_channels_latents,
        vae_scale_factor,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents


    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:

        block_state = self.get_block_state(state)

        self.check_inputs(
            height=block_state.height,
            width=block_state.width,
            vae_scale_factor=components.vae_scale_factor,
        )

        device = components._execution_device

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        final_batch_size = block_state.batch_size * block_state.num_images_per_prompt

        block_state.latents = self.prepare_latents(
            batch_size=final_batch_size,
            num_channels_latents=components.num_channels_latents,
            vae_scale_factor=components.vae_scale_factor,
            height=height,
            width=width,
            dtype=block_state.dtype,
            device=device,
            generator=block_state.generator)

        
        self.set_block_state(state, block_state)

        return components, state


class QwenImagePrepareImageLatentsStep(ModularPipelineBlocks):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-image generation process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="image_latents", required=True, type_hint=torch.Tensor, description="The latents representing the reference image, can be generated in vae encoder step"),
        ]


    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:

        block_state = self.get_block_state(state)

        height_image_latent, width_image_latent = block_state.image_latents.shape[3:]

        block_state.image_latents = pack_latents(
            latents=block_state.image_latents,
            batch_size=block_state.image_latents.shape[0],
            num_channels_latents=components.num_channels_latents,
            height=height_image_latent,
            width=width_image_latent,
        )

        
        self.set_block_state(state, block_state)

        return components, state




class QwenImageSetTimestepsStep(ModularPipelineBlocks):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that sets the the scheduler's timesteps for inference"
    
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="num_inference_steps", default=50),
            InputParam(name="sigmas"),
            InputParam(name="latents", required=True, type_hint=torch.Tensor, description="The latents to use for the denoising process"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="timesteps", type_hint=torch.Tensor, description="The timesteps to use for the denoising process"),
            OutputParam(name="num_inference_steps", type_hint=int, description="The number of inference steps to use for the denoising process"),
        ]
    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        sigmas = np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps) if block_state.sigmas is None else block_state.sigmas
        
        mu = calculate_shift(
            image_seq_len=block_state.latents.shape[1],
            base_seq_len= components.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len= components.scheduler.config.get("max_image_seq_len", 4096),
            base_shift= components.scheduler.config.get("base_shift", 0.5),
            max_shift= components.scheduler.config.get("max_shift", 1.15),
        )
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            scheduler=components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )

        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)

        return components, state
        

class QwenImagePrepareAdditionalInputsStep(ModularPipelineBlocks):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that prepares the additional inputs for the text-to-image generation process"
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="batch_size", required=True),
            InputParam(name="height"),
            InputParam(name="width"),
            InputParam(name="prompt_embeds_mask"),
            InputParam(name="negative_prompt_embeds_mask"),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="img_shapes", type_hint=List[List[Tuple[int, int, int]]], description="The shapes of the images latents, used for RoPE calculation"),
            OutputParam(name="txt_seq_lens", kwargs_type="guider_input_fields", type_hint=List[int], description="The sequence lengths of the prompt embeds, used for RoPE calculation"),
            OutputParam(name="negative_txt_seq_lens", kwargs_type="guider_input_fields", type_hint=List[int], description="The sequence lengths of the negative prompt embeds, used for RoPE calculation"),
        ]
    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:

        block_state = self.get_block_state(state)

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        block_state.img_shapes = [[(1, height // components.vae_scale_factor // 2, width // components.vae_scale_factor // 2)] * block_state.batch_size]
        block_state.txt_seq_lens = block_state.prompt_embeds_mask.sum(dim=1).tolist() if block_state.prompt_embeds_mask is not None else None
        block_state.negative_txt_seq_lens = (
            block_state.negative_prompt_embeds_mask.sum(dim=1).tolist() if block_state.negative_prompt_embeds_mask is not None else None
        )


        self.set_block_state(state, block_state)

        return components, state


class QwenImageEditPrepareAdditionalInputsStep(ModularPipelineBlocks):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that prepares the additional inputs for the text-to-image generation process"
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="batch_size", required=True),
            InputParam(name="image", required=True, type_hint=torch.Tensor, description="The resized image input"),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="prompt_embeds_mask"),
            InputParam(name="negative_prompt_embeds_mask"),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="img_shapes", type_hint=List[List[Tuple[int, int, int]]], description="The shapes of the images latents, used for RoPE calculation"),
            OutputParam(name="txt_seq_lens", kwargs_type="guider_input_fields", type_hint=List[int], description="The sequence lengths of the prompt embeds, used for RoPE calculation"),
            OutputParam(name="negative_txt_seq_lens", kwargs_type="guider_input_fields", type_hint=List[int], description="The sequence lengths of the negative prompt embeds, used for RoPE calculation"),
        ]
    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:

        block_state = self.get_block_state(state)

        image = block_state.image[0] if isinstance(block_state.image, list) else block_state.image
        image_width, image_height = image.size

        block_state.img_shapes = [
            [
                (1, block_state.height // components.vae_scale_factor // 2, block_state.width // components.vae_scale_factor // 2),
                (1, image_height // components.vae_scale_factor // 2, image_width // components.vae_scale_factor // 2),
            ]
        ] * block_state.batch_size

        block_state.txt_seq_lens = block_state.prompt_embeds_mask.sum(dim=1).tolist() if block_state.prompt_embeds_mask is not None else None
        block_state.negative_txt_seq_lens = (
            block_state.negative_prompt_embeds_mask.sum(dim=1).tolist() if block_state.negative_prompt_embeds_mask is not None else None
        )


        self.set_block_state(state, block_state)

        return components, state