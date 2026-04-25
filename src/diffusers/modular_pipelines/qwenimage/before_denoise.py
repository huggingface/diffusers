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

import numpy as np
import torch

from ...models import QwenImageControlNetModel, QwenImageMultiControlNetModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor, unwrap_module
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageLayeredPachifier, QwenImageModularPipeline, QwenImagePachifier


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
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
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
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
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


# ====================
# 1. PREPARE LATENTS
# ====================


# auto_docstring
class QwenImagePrepareLatentsStep(ModularPipelineBlocks):
    """
    Prepare initial random noise for the generation process

      Components:
          pachifier (`QwenImagePachifier`)

      Inputs:
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          height (`int`):
              if not set, updated to default value
          width (`int`):
              if not set, updated to default value
          latents (`Tensor`):
              The initial latents to use for the denoising process
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Prepare initial random noise for the generation process"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam.template("batch_size"),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="height", type_hint=int, description="if not set, updated to default value"),
            OutputParam(name="width", type_hint=int, description="if not set, updated to default value"),
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
        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=block_state.dtype
            )
            block_state.latents = components.pachifier.pack_latents(block_state.latents)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageLayeredPrepareLatentsStep(ModularPipelineBlocks):
    """
    Prepare initial random noise (B, layers+1, C, H, W) for the generation process

      Components:
          pachifier (`QwenImageLayeredPachifier`)

      Inputs:
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          layers (`int`, *optional*, defaults to 4):
              Number of layers to extract from the image
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          height (`int`):
              if not set, updated to default value
          width (`int`):
              if not set, updated to default value
          latents (`Tensor`):
              The initial latents to use for the denoising process
    """

    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Prepare initial random noise (B, layers+1, C, H, W) for the generation process"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImageLayeredPachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("layers"),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam.template("batch_size"),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="height", type_hint=int, description="if not set, updated to default value"),
            OutputParam(name="width", type_hint=int, description="if not set, updated to default value"),
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

        shape = (batch_size, block_state.layers + 1, components.num_channels_latents, latent_height, latent_width)
        if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(block_state.generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=block_state.dtype
            )
            block_state.latents = components.pachifier.pack_latents(block_state.latents)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImagePrepareLatentsWithStrengthStep(ModularPipelineBlocks):
    """
    Step that adds noise to image latents for image-to-image/inpainting. Should be run after set_timesteps,
    prepare_latents. Both noise and image latents should alreadybe patchified.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          latents (`Tensor`):
              The initial random noised, can be generated in prepare latent step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (Can be
              generated from vae encoder and updated in input step.)
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.

      Outputs:
          initial_noise (`Tensor`):
              The initial random noised used for inpainting denoising.
          latents (`Tensor`):
              The scaled noisy latents to use for inpainting/image-to-image denoising.
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that adds noise to image latents for image-to-image/inpainting. Should be run after set_timesteps, prepare_latents. Both noise and image latents should alreadybe patchified."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial random noised, can be generated in prepare latent step.",
            ),
            InputParam.template("image_latents", note="Can be generated from vae encoder and updated in input step."),
            InputParam(
                name="timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="initial_noise",
                type_hint=torch.Tensor,
                description="The initial random noised used for inpainting denoising.",
            ),
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The scaled noisy latents to use for inpainting/image-to-image denoising.",
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


# auto_docstring
class QwenImageCreateMaskLatentsStep(ModularPipelineBlocks):
    """
    Step that creates mask latents from preprocessed mask_image by interpolating to latent space.

      Components:
          pachifier (`QwenImagePachifier`)

      Inputs:
          processed_mask_image (`Tensor`):
              The processed mask to use for the inpainting process.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          mask (`Tensor`):
              The mask to use for the inpainting process.
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that creates mask latents from preprocessed mask_image by interpolating to latent space."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pachifier", QwenImagePachifier, default_creation_method="from_config"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="processed_mask_image",
                required=True,
                type_hint=torch.Tensor,
                description="The processed mask to use for the inpainting process.",
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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


# ====================
# 2. SET TIMESTEPS
# ====================


# auto_docstring
class QwenImageSetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the scheduler's timesteps for text-to-image generation. Should be run after prepare latents step.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`):
              The initial random noised latents for the denoising process. Can be generated in prepare latents step.

      Outputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for text-to-image generation. Should be run after prepare latents step."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps"),
            InputParam.template("sigmas"),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial random noised latents for the denoising process. Can be generated in prepare latents step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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


# auto_docstring
class QwenImageLayeredSetTimestepsStep(ModularPipelineBlocks):
    """
    Set timesteps step for QwenImage Layered with custom mu calculation based on image_latents.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.

      Outputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process.
    """

    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Set timesteps step for QwenImage Layered with custom mu calculation based on image_latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps"),
            InputParam.template("sigmas"),
            InputParam.template("image_latents"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="timesteps", type_hint=torch.Tensor, description="The timesteps to use for the denoising process."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        # Layered-specific mu calculation
        base_seqlen = 256 * 256 / 16 / 16  # = 256
        mu = (block_state.image_latents.shape[1] / base_seqlen) ** 0.5

        # Default sigmas if not provided
        sigmas = (
            np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)
            if block_state.sigmas is None
            else block_state.sigmas
        )

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageSetTimestepsWithStrengthStep(ModularPipelineBlocks):
    """
    Step that sets the scheduler's timesteps for image-to-image generation, and inpainting. Should be run after prepare
    latents step.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`):
              The latents to use for the denoising process. Can be generated in prepare latents step.
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.

      Outputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process.
          num_inference_steps (`int`):
              The number of denoising steps to perform at inference time. Updated based on strength.
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for image-to-image generation, and inpainting. Should be run after prepare latents step."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps"),
            InputParam.template("sigmas"),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to use for the denoising process. Can be generated in prepare latents step.",
            ),
            InputParam.template("strength", default=0.9),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="timesteps",
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process.",
            ),
            OutputParam(
                name="num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time. Updated based on strength.",
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


# ====================
# 3. OTHER INPUTS FOR DENOISER
# ====================

## RoPE inputs for denoiser


# auto_docstring
class QwenImageRoPEInputsStep(ModularPipelineBlocks):
    """
    Step that prepares the RoPE inputs for the denoising process. Should be place after prepare_latents step

      Inputs:
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.

      Outputs:
          img_shapes (`list`):
              The shapes of the images latents, used for RoPE calculation
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the RoPE inputs for the denoising process. Should be place after prepare_latents step"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("batch_size"),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                kwargs_type="denoiser_input_fields",
                type_hint=list[list[tuple[int, int, int]]],
                description="The shapes of the images latents, used for RoPE calculation",
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
        ] * block_state.batch_size

        self.set_block_state(state, block_state)

        return components, state


# auto_docstring
class QwenImageEditRoPEInputsStep(ModularPipelineBlocks):
    """
    Step that prepares the RoPE inputs for denoising process. This is used in QwenImage Edit. Should be placed after
    prepare_latents step

      Inputs:
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          image_height (`int`):
              The height of the reference image. Can be generated in input step.
          image_width (`int`):
              The width of the reference image. Can be generated in input step.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.

      Outputs:
          img_shapes (`list`):
              The shapes of the images latents, used for RoPE calculation
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that prepares the RoPE inputs for denoising process. This is used in QwenImage Edit. Should be placed after prepare_latents step"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("batch_size"),
            InputParam(
                name="image_height",
                required=True,
                type_hint=int,
                description="The height of the reference image. Can be generated in input step.",
            ),
            InputParam(
                name="image_width",
                required=True,
                type_hint=int,
                description="The width of the reference image. Can be generated in input step.",
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                kwargs_type="denoiser_input_fields",
                type_hint=list[list[tuple[int, int, int]]],
                description="The shapes of the images latents, used for RoPE calculation",
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

        self.set_block_state(state, block_state)

        return components, state


# auto_docstring
class QwenImageEditPlusRoPEInputsStep(ModularPipelineBlocks):
    """
    Step that prepares the RoPE inputs for denoising process. This is used in QwenImage Edit Plus.
      Unlike Edit, Edit Plus handles lists of image_height/image_width for multiple reference images. Should be placed
      after prepare_latents step.

      Inputs:
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          image_height (`list`):
              The heights of the reference images. Can be generated in input step.
          image_width (`list`):
              The widths of the reference images. Can be generated in input step.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.

      Outputs:
          img_shapes (`list`):
              The shapes of the image latents, used for RoPE calculation
          txt_seq_lens (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation
          negative_txt_seq_lens (`list`):
              The sequence lengths of the negative prompt embeds, used for RoPE calculation
    """

    model_name = "qwenimage-edit-plus"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the RoPE inputs for denoising process. This is used in QwenImage Edit Plus.\n"
            "Unlike Edit, Edit Plus handles lists of image_height/image_width for multiple reference images.\n"
            "Should be placed after prepare_latents step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("batch_size"),
            InputParam(
                name="image_height",
                required=True,
                type_hint=list[int],
                description="The heights of the reference images. Can be generated in input step.",
            ),
            InputParam(
                name="image_width",
                required=True,
                type_hint=list[int],
                description="The widths of the reference images. Can be generated in input step.",
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                kwargs_type="denoiser_input_fields",
                type_hint=list[list[tuple[int, int, int]]],
                description="The shapes of the image latents, used for RoPE calculation",
            ),
            OutputParam(
                name="txt_seq_lens",
                kwargs_type="denoiser_input_fields",
                type_hint=list[int],
                description="The sequence lengths of the prompt embeds, used for RoPE calculation",
            ),
            OutputParam(
                name="negative_txt_seq_lens",
                kwargs_type="denoiser_input_fields",
                type_hint=list[int],
                description="The sequence lengths of the negative prompt embeds, used for RoPE calculation",
            ),
        ]

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        vae_scale_factor = components.vae_scale_factor

        # Edit Plus: image_height and image_width are lists
        block_state.img_shapes = [
            [
                (1, block_state.height // vae_scale_factor // 2, block_state.width // vae_scale_factor // 2),
                *[
                    (1, img_height // vae_scale_factor // 2, img_width // vae_scale_factor // 2)
                    for img_height, img_width in zip(block_state.image_height, block_state.image_width)
                ],
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


# auto_docstring
class QwenImageLayeredRoPEInputsStep(ModularPipelineBlocks):
    """
    Step that prepares the RoPE inputs for the denoising process. Should be place after prepare_latents step

      Inputs:
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          layers (`int`, *optional*, defaults to 4):
              Number of layers to extract from the image
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.

      Outputs:
          img_shapes (`list`):
              The shapes of the image latents, used for RoPE calculation
          txt_seq_lens (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation
          negative_txt_seq_lens (`list`):
              The sequence lengths of the negative prompt embeds, used for RoPE calculation
          additional_t_cond (`Tensor`):
              The additional t cond, used for RoPE calculation
    """

    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the RoPE inputs for the denoising process. Should be place after prepare_latents step"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("batch_size"),
            InputParam.template("layers"),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                type_hint=list[list[tuple[int, int, int]]],
                kwargs_type="denoiser_input_fields",
                description="The shapes of the image latents, used for RoPE calculation",
            ),
            OutputParam(
                name="txt_seq_lens",
                type_hint=list[int],
                kwargs_type="denoiser_input_fields",
                description="The sequence lengths of the prompt embeds, used for RoPE calculation",
            ),
            OutputParam(
                name="negative_txt_seq_lens",
                type_hint=list[int],
                kwargs_type="denoiser_input_fields",
                description="The sequence lengths of the negative prompt embeds, used for RoPE calculation",
            ),
            OutputParam(
                name="additional_t_cond",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The additional t cond, used for RoPE calculation",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        # All shapes are the same for Layered
        shape = (
            1,
            block_state.height // components.vae_scale_factor // 2,
            block_state.width // components.vae_scale_factor // 2,
        )

        # layers+1 output shapes + 1 condition shape (all same)
        block_state.img_shapes = [[shape] * (block_state.layers + 2)] * block_state.batch_size

        # txt_seq_lens
        block_state.txt_seq_lens = (
            block_state.prompt_embeds_mask.sum(dim=1).tolist() if block_state.prompt_embeds_mask is not None else None
        )
        block_state.negative_txt_seq_lens = (
            block_state.negative_prompt_embeds_mask.sum(dim=1).tolist()
            if block_state.negative_prompt_embeds_mask is not None
            else None
        )

        block_state.additional_t_cond = torch.tensor([0] * block_state.batch_size).to(device=device, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


## ControlNet inputs for denoiser


# auto_docstring
class QwenImageControlNetBeforeDenoiserStep(ModularPipelineBlocks):
    """
    step that prepare inputs for controlnet. Insert before the Denoise Step, after set_timesteps step.

      Components:
          controlnet (`QwenImageControlNetModel`)

      Inputs:
          control_guidance_start (`float`, *optional*, defaults to 0.0):
              When to start applying ControlNet.
          control_guidance_end (`float`, *optional*, defaults to 1.0):
              When to stop applying ControlNet.
          controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
              Scale for ControlNet conditioning.
          control_image_latents (`Tensor`):
              The control image latents to use for the denoising process. Can be generated in controlnet vae encoder
              step.
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.

      Outputs:
          controlnet_keep (`list`):
              The controlnet keep values
    """

    model_name = "qwenimage"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("controlnet", QwenImageControlNetModel),
        ]

    @property
    def description(self) -> str:
        return "step that prepare inputs for controlnet. Insert before the Denoise Step, after set_timesteps step."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("control_guidance_start"),
            InputParam.template("control_guidance_end"),
            InputParam.template("controlnet_conditioning_scale"),
            InputParam(
                name="control_image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The control image latents to use for the denoising process. Can be generated in controlnet vae encoder step.",
            ),
            InputParam(
                name="timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("controlnet_keep", type_hint=list[float], description="The controlnet keep values"),
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
