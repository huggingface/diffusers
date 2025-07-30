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

from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import PipelineBlock, PipelineState
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


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


class FluxInputStep(PipelineBlock):
    model_name = "flux"

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
            InputParam("num_images_per_prompt", default=1),
        ]

    @property
    def intermediate_inputs(self) -> List[str]:
        return [
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="Pre-generated pooled text embeddings. Can be generated from text_encoder step.",
            ),
            # TODO: support negative embeddings?
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
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="pooled text embeddings used to guide the image generation",
            ),
            # TODO: support negative embeddings?
        ]

    def check_inputs(self, components, block_state):
        if block_state.prompt_embeds is not None and block_state.pooled_prompt_embeds is not None:
            if block_state.prompt_embeds.shape[0] != block_state.pooled_prompt_embeds.shape[0]:
                raise ValueError(
                    "`prompt_embeds` and `pooled_prompt_embeds` must have the same batch size when passed directly, but"
                    f" got: `prompt_embeds` {block_state.prompt_embeds.shape} != `pooled_prompt_embeds`"
                    f" {block_state.pooled_prompt_embeds.shape}."
                )

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        # TODO: consider adding negative embeddings?
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
        )
        self.set_block_state(state, block_state)

        return components, state


class FluxSetTimestepsStep(PipelineBlock):
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
        ]

    @property
    def intermediate_inputs(self) -> List[str]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            )
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

        latents = block_state.latents
        image_seq_len = latents.shape[1]

        num_inference_steps = block_state.num_inference_steps
        sigmas = block_state.sigmas
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None
        block_state.sigmas = sigmas
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            scheduler, block_state.num_inference_steps, block_state.device, sigmas=block_state.sigmas, mu=mu
        )
        if components.transformer.config.guidance_embeds:
            guidance = torch.full([1], block_state.guidance_scale, device=block_state.device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        block_state.guidance = guidance

        self.set_block_state(state, block_state)
        return components, state


class FluxPrepareLatentsStep(PipelineBlock):
    model_name = "flux"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-video generation process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("latents", type_hint=Optional[torch.Tensor]),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
        ]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
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
            OutputParam(
                "latent_image_ids",
                type_hint=torch.Tensor,
                description="IDs computed from the image sequence needed for RoPE",
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
        # Couldn't use the `prepare_latents` method directly from Flux because I decided to copy over
        # the packing methods here. So, for example, `comp._pack_latents()` won't work if we were
        # to go with the "# Copied from ..." approach. Or maybe there's a way?

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (comp.vae_scale_factor * 2))
        width = 2 * (int(width) // (comp.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        block_state.device = components._execution_device
        block_state.dtype = torch.bfloat16  # TODO: okay to hardcode this?
        block_state.num_channels_latents = components.num_channels_latents

        self.check_inputs(components, block_state)

        block_state.latents, block_state.latent_image_ids = self.prepare_latents(
            components,
            block_state.batch_size * block_state.num_images_per_prompt,
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
