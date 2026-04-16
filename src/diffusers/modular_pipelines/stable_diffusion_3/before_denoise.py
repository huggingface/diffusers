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

import inspect

import torch

from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import StableDiffusion3ModularPipeline


logger = logging.get_logger(__name__)


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


def _get_initial_timesteps_and_optionals(
    transformer,
    scheduler,
    height,
    width,
    patch_size,
    vae_scale_factor,
    num_inference_steps,
    sigmas,
    device,
    mu=None,
):
    scheduler_kwargs = {}
    if scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        image_seq_len = (height // vae_scale_factor // patch_size) * (width // vae_scale_factor // patch_size)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
    )
    return timesteps, num_inference_steps


class StableDiffusion3SetTimestepsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_inference_steps",
                default=50,
                description="The number of denoising steps.",
            ),
            InputParam(
                "timesteps",
                description="Custom timesteps to use for the denoising process.",
            ),
            InputParam("sigmas", description="Custom sigmas to use for the denoising process."),
            InputParam(
                "height",
                type_hint=int,
                description="The height in pixels of the generated image.",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="The width in pixels of the generated image.",
            ),
            InputParam(
                "mu",
                type_hint=float,
                description="The mu value used for dynamic shifting. If not provided, it is dynamically calculated.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "timesteps",
                type_hint=torch.Tensor,
                description="The timesteps schedule for the denoising process.",
            ),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The final number of inference steps.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        timesteps, num_inference_steps = _get_initial_timesteps_and_optionals(
            components.transformer,
            components.scheduler,
            block_state.height,
            block_state.width,
            components.patch_size,
            components.vae_scale_factor,
            block_state.num_inference_steps,
            block_state.sigmas,
            block_state.device,
            getattr(block_state, "mu", None),
        )

        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusion3Img2ImgSetTimestepsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for img2img inference"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_inference_steps",
                default=50,
                description="The number of denoising steps.",
            ),
            InputParam(
                "timesteps",
                description="Custom timesteps to use for the denoising process.",
            ),
            InputParam("sigmas", description="Custom sigmas to use for the denoising process."),
            InputParam(
                "strength",
                default=0.6,
                description="Indicates extent to transform the reference image.",
            ),
            InputParam(
                "height",
                type_hint=int,
                description="The height in pixels of the generated image.",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="The width in pixels of the generated image.",
            ),
            InputParam(
                "mu",
                type_hint=float,
                description="The mu value used for dynamic shifting. If not provided, it is dynamically calculated.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "timesteps",
                type_hint=torch.Tensor,
                description="The timesteps schedule for the denoising process.",
            ),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The final number of inference steps.",
            ),
        ]

    @staticmethod
    def get_timesteps(scheduler, num_inference_steps, strength):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start * scheduler.order)
        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        timesteps, num_inference_steps = _get_initial_timesteps_and_optionals(
            components.transformer,
            components.scheduler,
            block_state.height,
            block_state.width,
            components.patch_size,
            components.vae_scale_factor,
            block_state.num_inference_steps,
            block_state.sigmas,
            block_state.device,
            getattr(block_state, "mu", None),
        )

        timesteps, num_inference_steps = self.get_timesteps(
            components.scheduler, num_inference_steps, block_state.strength
        )

        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusion3PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def description(self) -> str:
        return "Prepare latents step for Text-to-Image"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "height",
                type_hint=int,
                description="The height in pixels of the generated image.",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="The width in pixels of the generated image.",
            ),
            InputParam(
                "latents",
                type_hint=torch.Tensor | None,
                description="Pre-generated noisy latents to be used as inputs for image generation.",
            ),
            InputParam(
                "num_images_per_prompt",
                type_hint=int,
                default=1,
                description="The number of images to generate per prompt.",
            ),
            InputParam(
                "generator",
                description="One or a list of torch generator(s) to make generation deterministic.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="The batch size for latent generation.",
            ),
            InputParam(
                "dtype",
                type_hint=torch.dtype,
                description="The data type for the latents.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The prepared latent tensors to be denoised.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=block_state.device, dtype=block_state.dtype)
        else:
            shape = (
                batch_size,
                components.num_channels_latents,
                int(block_state.height) // components.vae_scale_factor,
                int(block_state.width) // components.vae_scale_factor,
            )
            block_state.latents = randn_tensor(
                shape,
                generator=block_state.generator,
                device=block_state.device,
                dtype=block_state.dtype,
            )

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusion3Img2ImgPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to be scaled by the scheduler.",
            ),
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The image latents encoded by the VAE.",
            ),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps schedule.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "initial_noise",
                type_hint=torch.Tensor,
                description="The initial noise applied to the image latents.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        latent_timestep = block_state.timesteps[:1].repeat(block_state.latents.shape[0])
        block_state.initial_noise = block_state.latents
        block_state.latents = components.scheduler.scale_noise(
            block_state.image_latents, latent_timestep, block_state.latents
        )
        self.set_block_state(state, block_state)
        return components, state
