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

import numpy as np
import torch

from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ChromaModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


def _prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


class ChromaPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "chroma"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-image generation process"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("latents"),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`. Can be generated in input step.",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The initial noisy latents (patchified) to use for the denoising process",
            ),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        if (block_state.height is not None and block_state.height % (components.vae_scale_factor * 2) != 0) or (
            block_state.width is not None and block_state.width % (components.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {components.vae_scale_factor * 2} but are {block_state.height} and {block_state.width}."
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
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
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

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents

    @torch.no_grad()
    def __call__(self, components: ChromaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        device = components._execution_device

        self.check_inputs(components, block_state)
        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        block_state.latents = self.prepare_latents(
            components,
            batch_size,
            components.num_channels_latents,
            block_state.height,
            block_state.width,
            block_state.dtype,
            device,
            block_state.generator,
            block_state.latents,
        )

        self.set_block_state(state, block_state)

        return components, state


class ChromaSetTimestepsStep(ModularPipelineBlocks):
    model_name = "chroma"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference. Should be run after the prepare latents step."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True, note="patchified, can be generated in prepare_latents step"),
            InputParam.template("num_inference_steps", default=35),
            InputParam.template("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: ChromaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        num_inference_steps = block_state.num_inference_steps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if block_state.sigmas is None
            else block_state.sigmas
        )
        image_seq_len = block_state.latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            components.scheduler.config.get("base_image_seq_len", 256),
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_shift", 0.5),
            components.scheduler.config.get("max_shift", 1.15),
        )
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        self.set_block_state(state, block_state)
        return components, state


class ChromaPrepareAttentionMaskStep(ModularPipelineBlocks):
    model_name = "chroma"

    @property
    def description(self) -> str:
        return (
            "Step that extends the prompt attention masks to cover the image tokens in the final sequence. "
            "Should be run after the text input and prepare latents steps."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True, note="patchified, can be generated in prepare_latents step"),
            InputParam(
                "prompt_attention_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Attention mask for the prompt embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask for the negative prompt embeddings. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask covering the text and image tokens of the final sequence.",
            ),
            OutputParam(
                "negative_attention_mask",
                type_hint=torch.Tensor,
                description="Negative attention mask covering the text and image tokens of the final sequence.",
            ),
        ]

    @staticmethod
    def prepare_attention_mask(batch_size, sequence_length, attention_mask):
        if attention_mask is None:
            return attention_mask

        # Extend the prompt attention mask to account for image tokens in the final sequence
        attention_mask = torch.cat(
            [attention_mask, torch.ones(batch_size, sequence_length, device=attention_mask.device, dtype=torch.bool)],
            dim=1,
        )

        return attention_mask

    @torch.no_grad()
    def __call__(self, components: ChromaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        batch_size, image_seq_len = block_state.latents.shape[0], block_state.latents.shape[1]

        block_state.attention_mask = self.prepare_attention_mask(
            batch_size, image_seq_len, block_state.prompt_attention_mask
        )
        block_state.negative_attention_mask = self.prepare_attention_mask(
            batch_size, image_seq_len, block_state.negative_prompt_attention_mask
        )

        self.set_block_state(state, block_state)
        return components, state


class ChromaRoPEInputsStep(ModularPipelineBlocks):
    model_name = "chroma"

    @property
    def description(self) -> str:
        return "Step that prepares the RoPE inputs for the denoising process. Should be placed after text encoder and latent preparation steps."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("prompt_embeds"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="txt_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="IDs computed from the text sequence, used for RoPE calculation.",
            ),
            OutputParam(
                name="img_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="IDs computed from the latent sequence, used for RoPE calculation.",
            ),
        ]

    def __call__(self, components: ChromaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_embeds = block_state.prompt_embeds
        device, dtype = prompt_embeds.device, prompt_embeds.dtype
        block_state.txt_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        height = 2 * (int(block_state.height) // (components.vae_scale_factor * 2))
        width = 2 * (int(block_state.width) // (components.vae_scale_factor * 2))
        block_state.img_ids = _prepare_latent_image_ids(height // 2, width // 2, device, dtype)

        self.set_block_state(state, block_state)

        return components, state
