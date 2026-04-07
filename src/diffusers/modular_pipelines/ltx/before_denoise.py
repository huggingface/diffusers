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

import numpy as np
import torch

from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import LTXModularPipeline


logger = logging.get_logger(__name__)


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
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape
    # [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length,
    # dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


# Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


class LTXTextInputStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts input tensor shapes based on `batch_size` and `num_videos_per_prompt`"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("prompt_embeds", required=True),
            InputParam.template("prompt_embeds_mask", name="prompt_attention_mask"),
            InputParam.template("negative_prompt_embeds"),
            InputParam.template("negative_prompt_embeds_mask", name="negative_prompt_attention_mask"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype
        num_videos = block_state.num_videos_per_prompt

        # Repeat prompt_embeds for num_videos_per_prompt
        _, seq_len, _ = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, num_videos, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(block_state.batch_size * num_videos, seq_len, -1)

        if block_state.prompt_attention_mask is not None:
            block_state.prompt_attention_mask = block_state.prompt_attention_mask.repeat(num_videos, 1)

        if block_state.negative_prompt_embeds is not None:
            _, seq_len, _ = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(1, num_videos, 1)
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                block_state.batch_size * num_videos, seq_len, -1
            )

        if block_state.negative_prompt_attention_mask is not None:
            block_state.negative_prompt_attention_mask = block_state.negative_prompt_attention_mask.repeat(
                num_videos, 1
            )

        self.set_block_state(state, block_state)
        return components, state


class LTXSetTimestepsStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps"),
            InputParam.template("timesteps"),
            InputParam.template("sigmas"),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=161),
            InputParam("frame_rate", type_hint=int, default=25),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
            OutputParam("rope_interpolation_scale", type_hint=tuple),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        height = block_state.height
        width = block_state.width
        num_frames = block_state.num_frames
        frame_rate = block_state.frame_rate

        latent_num_frames = (num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = height // components.vae_spatial_compression_ratio
        latent_width = width // components.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width

        custom_timesteps = block_state.timesteps
        sigmas = block_state.sigmas

        if custom_timesteps is not None:
            # User provided custom timesteps, don't compute sigmas
            block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
                components.scheduler,
                block_state.num_inference_steps,
                device,
                custom_timesteps,
            )
        else:
            if sigmas is None:
                sigmas = np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)

            mu = calculate_shift(
                video_sequence_length,
                components.scheduler.config.get("base_image_seq_len", 256),
                components.scheduler.config.get("max_image_seq_len", 4096),
                components.scheduler.config.get("base_shift", 0.5),
                components.scheduler.config.get("max_shift", 1.15),
            )

            block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
                components.scheduler,
                block_state.num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )

        block_state.rope_interpolation_scale = (
            components.vae_temporal_compression_ratio / frame_rate,
            components.vae_spatial_compression_ratio,
            components.vae_spatial_compression_ratio,
        )

        self.set_block_state(state, block_state)
        return components, state


class LTXPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-video generation process"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=161),
            InputParam.template("latents"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("generator"),
            InputParam.template("batch_size", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_channels_latents = components.transformer.config.in_channels

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)
        else:
            height = block_state.height // components.vae_spatial_compression_ratio
            width = block_state.width // components.vae_spatial_compression_ratio
            num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

            shape = (batch_size, num_channels_latents, num_frames, height, width)
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=torch.float32
            )
            block_state.latents = _pack_latents(
                block_state.latents,
                components.transformer_spatial_patch_size,
                components.transformer_temporal_patch_size,
            )

        self.set_block_state(state, block_state)
        return components, state


class LTXImage2VideoPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return (
            "Prepare latents step for image-to-video: takes pre-encoded image latents and creates a conditioning mask"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image_latents", type_hint=torch.Tensor, required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=161),
            InputParam.template("latents"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("generator"),
            InputParam.template("batch_size", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("conditioning_mask", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt

        height = block_state.height // components.vae_spatial_compression_ratio
        width = block_state.width // components.vae_spatial_compression_ratio
        num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

        mask_shape = (batch_size, 1, num_frames, height, width)

        if block_state.latents is not None:
            conditioning_mask = block_state.latents.new_zeros(mask_shape)
            conditioning_mask[:, :, 0] = 1.0
            conditioning_mask = _pack_latents(
                conditioning_mask,
                components.transformer_spatial_patch_size,
                components.transformer_temporal_patch_size,
            ).squeeze(-1)
            block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)
            block_state.conditioning_mask = conditioning_mask
            self.set_block_state(state, block_state)
            return components, state

        init_latents = block_state.image_latents.to(device=device, dtype=torch.float32)
        if init_latents.shape[0] < batch_size:
            init_latents = init_latents.repeat_interleave(batch_size // init_latents.shape[0], dim=0)
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)

        actual_mask_shape = (
            init_latents.shape[0],
            1,
            init_latents.shape[2],
            init_latents.shape[3],
            init_latents.shape[4],
        )
        conditioning_mask = torch.zeros(actual_mask_shape, device=device, dtype=torch.float32)
        conditioning_mask[:, :, 0] = 1.0

        noise = randn_tensor(init_latents.shape, generator=block_state.generator, device=device, dtype=torch.float32)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = _pack_latents(
            conditioning_mask,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        ).squeeze(-1)
        latents = _pack_latents(
            latents,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        )

        block_state.latents = latents
        block_state.conditioning_mask = conditioning_mask

        self.set_block_state(state, block_state)
        return components, state
