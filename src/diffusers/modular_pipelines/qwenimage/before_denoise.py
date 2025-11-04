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

from ...models import QwenImageControlNetModel, QwenImageMultiControlNetModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor, unwrap_module
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline, QwenImagePachifier


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


# modified from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(num_inference_steps * strength, num_inference_steps)

    t_start = int(max(num_inference_steps - init_timestep, 0))
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(t_start * scheduler.order)

    return timesteps, num_inference_steps - t_start


# Prepare Latents steps


class QwenImagePrepareLatentsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Prepare initial random noise for the generation process"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

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
            InputParam(
                name="dtype",
                required=True,
                type_hint=torch.dtype,
                description="The dtype of the model inputs, can be generated in input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process",
            ),
        ]

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):
        if height is not None and height % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * 2} but is {height}")

        if width is not None and width % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * 2} but is {width}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(
            height=block_state.height,
            width=block_state.width,
            vae_scale_factor=components.vae_scale_factor,
        )

        device = components._execution_device
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        # we can update the height and width here since it's used to generate the initial
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        latent_height = 2 * (int(block_state.height) // (components.vae_scale_factor * 2))
        latent_width = 2 * (int(block_state.width) // (components.vae_scale_factor * 2))

        shape = (batch_size, components.num_channels_latents, 1, latent_height, latent_width)
        if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(block_state.generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        block_state.latents = randn_tensor(
            shape, generator=block_state.generator, device=device, dtype=block_state.dtype
        )
        block_state.latents = components.pachifier.pack_latents(block_state.latents)

        self.set_block_state(state, block_state)
        return components, state


class QwenImagePrepareLatentsWithStrengthStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that adds noise to image latents for image-to-image/inpainting. Should be run after set_timesteps, prepare_latents. Both noise and image latents should alreadybe patchified."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial random noised, can be generated in prepare latent step.",
            ),
            InputParam(
                name="image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The image latents to use for the denoising process. Can be generated in vae encoder and packed in input step.",
            ),
            InputParam(
                name="timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="initial_noise",
                type_hint=torch.Tensor,
                description="The initial random noised used for inpainting denoising.",
            ),
        ]

    @staticmethod
    def check_inputs(image_latents, latents):
        if image_latents.shape[0] != latents.shape[0]:
            raise ValueError(
                f"`image_latents` must have have same batch size as `latents`, but got {image_latents.shape[0]} and {latents.shape[0]}"
            )

        if image_latents.ndim != 3:
            raise ValueError(f"`image_latents` must have 3 dimensions (patchified), but got {image_latents.ndim}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(
            image_latents=block_state.image_latents,
            latents=block_state.latents,
        )

        # prepare latent timestep
        latent_timestep = block_state.timesteps[:1].repeat(block_state.latents.shape[0])

        # make copy of initial_noise
        block_state.initial_noise = block_state.latents

        # scale noise
        block_state.latents = components.scheduler.scale_noise(
            block_state.image_latents, latent_timestep, block_state.latents
        )

        self.set_block_state(state, block_state)

        return components, state


class QwenImageCreateMaskLatentsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that creates mask latents from preprocessed mask_image by interpolating to latent space."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="processed_mask_image",
                required=True,
                type_hint=torch.Tensor,
                description="The processed mask to use for the inpainting process.",
            ),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="dtype", required=True),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="mask", type_hint=torch.Tensor, description="The mask to use for the inpainting process."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.

        height_latents = 2 * (int(block_state.height) // (components.vae_scale_factor * 2))
        width_latents = 2 * (int(block_state.width) // (components.vae_scale_factor * 2))

        block_state.mask = torch.nn.functional.interpolate(
            block_state.processed_mask_image,
            size=(height_latents, width_latents),
        )

        block_state.mask = block_state.mask.unsqueeze(2)
        block_state.mask = block_state.mask.repeat(1, components.num_channels_latents, 1, 1, 1)
        block_state.mask = block_state.mask.to(device=device, dtype=block_state.dtype)

        block_state.mask = components.pachifier.pack_latents(block_state.mask)

        self.set_block_state(state, block_state)

        return components, state


# Set Timesteps steps


class QwenImageSetTimestepsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that sets the the scheduler's timesteps for text-to-image generation. Should be run after prepare latents step."

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
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to use for the denoising process, used to calculate the image sequence length.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="timesteps", type_hint=torch.Tensor, description="The timesteps to use for the denoising process"
            ),
        ]

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        sigmas = (
            np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)
            if block_state.sigmas is None
            else block_state.sigmas
        )

        mu = calculate_shift(
            image_seq_len=block_state.latents.shape[1],
            base_seq_len=components.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=components.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=components.scheduler.config.get("base_shift", 0.5),
            max_shift=components.scheduler.config.get("max_shift", 1.15),
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


class QwenImageSetTimestepsWithStrengthStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that sets the the scheduler's timesteps for image-to-image generation, and inpainting. Should be run after prepare latents step."

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
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to use for the denoising process, used to calculate the image sequence length.",
            ),
            InputParam(name="strength", default=0.9),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="timesteps",
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        sigmas = (
            np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)
            if block_state.sigmas is None
            else block_state.sigmas
        )

        mu = calculate_shift(
            image_seq_len=block_state.latents.shape[1],
            base_seq_len=components.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=components.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=components.scheduler.config.get("base_shift", 0.5),
            max_shift=components.scheduler.config.get("max_shift", 1.15),
        )
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            scheduler=components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )

        block_state.timesteps, block_state.num_inference_steps = get_timesteps(
            scheduler=components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            strength=block_state.strength,
        )

        self.set_block_state(state, block_state)

        return components, state


# other inputs for denoiser

## RoPE inputs for denoiser


class QwenImageRoPEInputsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the RoPE inputs for the denoising process. Should be place after prepare_latents step"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="batch_size", required=True),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="prompt_embeds_mask"),
            InputParam(name="negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                type_hint=List[List[Tuple[int, int, int]]],
                description="The shapes of the images latents, used for RoPE calculation",
            ),
            OutputParam(
                name="txt_seq_lens",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the prompt embeds, used for RoPE calculation",
            ),
            OutputParam(
                name="negative_txt_seq_lens",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the negative prompt embeds, used for RoPE calculation",
            ),
        ]

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.img_shapes = [
            [
                (
                    1,
                    block_state.height // components.vae_scale_factor // 2,
                    block_state.width // components.vae_scale_factor // 2,
                )
            ]
            * block_state.batch_size
        ]
        block_state.txt_seq_lens = (
            block_state.prompt_embeds_mask.sum(dim=1).tolist() if block_state.prompt_embeds_mask is not None else None
        )
        block_state.negative_txt_seq_lens = (
            block_state.negative_prompt_embeds_mask.sum(dim=1).tolist()
            if block_state.negative_prompt_embeds_mask is not None
            else None
        )

        self.set_block_state(state, block_state)

        return components, state


class QwenImageEditRoPEInputsStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that prepares the RoPE inputs for denoising process. This is used in QwenImage Edit. Should be placed after prepare_latents step"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="batch_size", required=True),
            InputParam(name="image_height", required=True),
            InputParam(name="image_width", required=True),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="prompt_embeds_mask"),
            InputParam(name="negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                type_hint=List[List[Tuple[int, int, int]]],
                description="The shapes of the images latents, used for RoPE calculation",
            ),
            OutputParam(
                name="txt_seq_lens",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the prompt embeds, used for RoPE calculation",
            ),
            OutputParam(
                name="negative_txt_seq_lens",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the negative prompt embeds, used for RoPE calculation",
            ),
        ]

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # for edit, image size can be different from the target size (height/width)

        block_state.img_shapes = [
            [
                (
                    1,
                    block_state.height // components.vae_scale_factor // 2,
                    block_state.width // components.vae_scale_factor // 2,
                ),
                (
                    1,
                    block_state.image_height // components.vae_scale_factor // 2,
                    block_state.image_width // components.vae_scale_factor // 2,
                ),
            ]
        ] * block_state.batch_size

        block_state.txt_seq_lens = (
            block_state.prompt_embeds_mask.sum(dim=1).tolist() if block_state.prompt_embeds_mask is not None else None
        )
        block_state.negative_txt_seq_lens = (
            block_state.negative_prompt_embeds_mask.sum(dim=1).tolist()
            if block_state.negative_prompt_embeds_mask is not None
            else None
        )

        self.set_block_state(state, block_state)

        return components, state


## ControlNet inputs for denoiser
class QwenImageControlNetBeforeDenoiserStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("controlnet", QwenImageControlNetModel),
        ]

    @property
    def description(self) -> str:
        return "step that prepare inputs for controlnet. Insert before the Denoise Step, after set_timesteps step."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("control_image_latents", required=True),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("controlnet_keep", type_hint=List[float], description="The controlnet keep values"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        controlnet = unwrap_module(components.controlnet)

        # control_guidance_start/control_guidance_end (align format)
        if not isinstance(block_state.control_guidance_start, list) and isinstance(
            block_state.control_guidance_end, list
        ):
            block_state.control_guidance_start = len(block_state.control_guidance_end) * [
                block_state.control_guidance_start
            ]
        elif not isinstance(block_state.control_guidance_end, list) and isinstance(
            block_state.control_guidance_start, list
        ):
            block_state.control_guidance_end = len(block_state.control_guidance_start) * [
                block_state.control_guidance_end
            ]
        elif not isinstance(block_state.control_guidance_start, list) and not isinstance(
            block_state.control_guidance_end, list
        ):
            mult = (
                len(block_state.control_image_latents) if isinstance(controlnet, QwenImageMultiControlNetModel) else 1
            )
            block_state.control_guidance_start, block_state.control_guidance_end = (
                mult * [block_state.control_guidance_start],
                mult * [block_state.control_guidance_end],
            )

        # controlnet_conditioning_scale (align format)
        if isinstance(controlnet, QwenImageMultiControlNetModel) and isinstance(
            block_state.controlnet_conditioning_scale, float
        ):
            block_state.controlnet_conditioning_scale = [block_state.controlnet_conditioning_scale] * mult

        # controlnet_keep
        block_state.controlnet_keep = []
        for i in range(len(block_state.timesteps)):
            keeps = [
                1.0 - float(i / len(block_state.timesteps) < s or (i + 1) / len(block_state.timesteps) > e)
                for s, e in zip(block_state.control_guidance_start, block_state.control_guidance_end)
            ]
            block_state.controlnet_keep.append(keeps[0] if isinstance(controlnet, QwenImageControlNetModel) else keeps)

        self.set_block_state(state, block_state)

        return components, state
