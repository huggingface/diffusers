# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Union

import numpy as np
import torch

from ...pipelines import FluxPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import FluxModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def _get_initial_timesteps_and_optionals(
    transformer,
    scheduler,
    batch_size,
    height,
    width,
    vae_scale_factor,
    num_inference_steps,
    guidance_scale,
    sigmas,
    device,
):
    image_seq_len = (int(height) // vae_scale_factor // 2) * (int(width) // vae_scale_factor // 2)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    if transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(batch_size)
    else:
        guidance = None

    return timesteps, num_inference_steps, sigmas, guidance


class FluxSetTimestepsStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("timesteps"),
            InputParam("sigmas"),
            InputParam("guidance_scale", default=3.5),
            InputParam("latents", type_hint=torch.Tensor),
            InputParam("num_images_per_prompt", default=1),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`. Can be generated in input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time",
            ),
            OutputParam("guidance", type_hint=torch.Tensor, description="Optional guidance to be used."),
        ]

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        scheduler = components.scheduler
        transformer = components.transformer

        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        timesteps, num_inference_steps, sigmas, guidance = _get_initial_timesteps_and_optionals(
            transformer,
            scheduler,
            batch_size,
            block_state.height,
            block_state.width,
            components.vae_scale_factor,
            block_state.num_inference_steps,
            block_state.guidance_scale,
            block_state.sigmas,
            block_state.device,
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.sigmas = sigmas
        block_state.guidance = guidance

        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


class FluxImg2ImgSetTimestepsStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("timesteps"),
            InputParam("sigmas"),
            InputParam("strength", default=0.6),
            InputParam("guidance_scale", default=3.5),
            InputParam("num_images_per_prompt", default=1),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`. Can be generated in input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time",
            ),
            OutputParam("guidance", type_hint=torch.Tensor, description="Optional guidance to be used."),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps with self.scheduler->scheduler
    def get_timesteps(scheduler, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start * scheduler.order)

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        scheduler = components.scheduler
        transformer = components.transformer
        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        timesteps, num_inference_steps, sigmas, guidance = _get_initial_timesteps_and_optionals(
            transformer,
            scheduler,
            batch_size,
            block_state.height,
            block_state.width,
            components.vae_scale_factor,
            block_state.num_inference_steps,
            block_state.guidance_scale,
            block_state.sigmas,
            block_state.device,
        )
        timesteps, num_inference_steps = self.get_timesteps(
            scheduler, num_inference_steps, block_state.strength, block_state.device
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.sigmas = sigmas
        block_state.guidance = guidance

        self.set_block_state(state, block_state)
        return components, state


class FluxPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-image generation process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("latents", type_hint=Optional[torch.Tensor]),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`. Can be generated in input step.",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"
            ),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        if (block_state.height is not None and block_state.height % (components.vae_scale_factor * 2) != 0) or (
            block_state.width is not None and block_state.width % (components.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {components.vae_scale_factor} but are {block_state.height} and {block_state.width}."
            )

    @staticmethod
    def prepare_latents(
        comp,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (comp.vae_scale_factor * 2))
        width = 2 * (int(width) // (comp.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # TODO: move packing latents code to a patchifier similar to Qwen
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        block_state.device = components._execution_device
        block_state.num_channels_latents = components.num_channels_latents

        self.check_inputs(components, block_state)
        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        block_state.latents = self.prepare_latents(
            components,
            batch_size,
            block_state.num_channels_latents,
            block_state.height,
            block_state.width,
            block_state.dtype,
            block_state.device,
            block_state.generator,
            block_state.latents,
        )

        self.set_block_state(state, block_state)

        return components, state


class FluxImg2ImgPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def description(self) -> str:
        return "Step that adds noise to image latents for image-to-image. Should be run after `set_timesteps`,"
        " `prepare_latents`. Both noise and image latents should already be patchified."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

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
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(image_latents=block_state.image_latents, latents=block_state.latents)

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


class FluxRoPEInputsStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def description(self) -> str:
        return "Step that prepares the RoPE inputs for the denoising process. Should be placed after text encoder and latent preparation steps."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="prompt_embeds"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="txt_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the prompt embeds, used for RoPE calculation.",
            ),
            OutputParam(
                name="img_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the image latents, used for RoPE calculation.",
            ),
        ]

    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_embeds = block_state.prompt_embeds
        device, dtype = prompt_embeds.device, prompt_embeds.dtype
        block_state.txt_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=prompt_embeds.device, dtype=prompt_embeds.dtype
        )

        height = 2 * (int(block_state.height) // (components.vae_scale_factor * 2))
        width = 2 * (int(block_state.width) // (components.vae_scale_factor * 2))
        block_state.img_ids = FluxPipeline._prepare_latent_image_ids(None, height // 2, width // 2, device, dtype)

        self.set_block_state(state, block_state)

        return components, state


class FluxKontextRoPEInputsStep(ModularPipelineBlocks):
    model_name = "flux-kontext"

    @property
    def description(self) -> str:
        return "Step that prepares the RoPE inputs for the denoising process of Flux Kontext. Should be placed after text encoder and latent preparation steps."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="image_height"),
            InputParam(name="image_width"),
            InputParam(name="height"),
            InputParam(name="width"),
            InputParam(name="prompt_embeds"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="txt_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the prompt embeds, used for RoPE calculation.",
            ),
            OutputParam(
                name="img_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=List[int],
                description="The sequence lengths of the image latents, used for RoPE calculation.",
            ),
        ]

    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_embeds = block_state.prompt_embeds
        device, dtype = prompt_embeds.device, prompt_embeds.dtype
        block_state.txt_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=prompt_embeds.device, dtype=prompt_embeds.dtype
        )

        img_ids = None
        if (
            getattr(block_state, "image_height", None) is not None
            and getattr(block_state, "image_width", None) is not None
        ):
            image_latent_height = 2 * (int(block_state.image_height) // (components.vae_scale_factor * 2))
            image_latent_width = 2 * (int(block_state.image_width) // (components.vae_scale_factor * 2))
            img_ids = FluxPipeline._prepare_latent_image_ids(
                None, image_latent_height // 2, image_latent_width // 2, device, dtype
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            img_ids[..., 0] = 1

        height = 2 * (int(block_state.height) // (components.vae_scale_factor * 2))
        width = 2 * (int(block_state.width) // (components.vae_scale_factor * 2))
        latent_ids = FluxPipeline._prepare_latent_image_ids(None, height // 2, width // 2, device, dtype)

        if img_ids is not None:
            latent_ids = torch.cat([latent_ids, img_ids], dim=0)

        block_state.img_ids = latent_ids

        self.set_block_state(state, block_state)

        return components, state
