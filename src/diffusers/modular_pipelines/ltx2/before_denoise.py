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

import copy
import inspect

import numpy as np
import torch

from ...models import AutoencoderKLLTX2Audio, LTX2VideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


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
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
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


# The pack/normalize helpers below mirror the static methods on `diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline`.
# Redefined here (not imported) because modular blocks must not import from `diffusers.pipelines.*` (gotcha #1).
def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
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


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_audio_latents(
    latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
) -> torch.Tensor:
    # Audio latents of shape [B, C, L, M] (L = latent audio length, M = mel bins). With no patch sizes this packs to
    # [B, L, C * M] (implicit mel patch_size of M, temporal patch_size of 1).
    if patch_size is not None and patch_size_t is not None:
        batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
        post_patch_latent_length = latent_length / patch_size_t
        post_patch_mel_bins = latent_mel_bins / patch_size
        latents = latents.reshape(
            batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
    else:
        latents = latents.transpose(1, 2).flatten(2, 3)  # [B, C, L, M] -> [B, L, C * M]
    return latents


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


def _normalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


def _create_noised_state(
    latents: torch.Tensor, noise_scale: float | torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    return noise_scale * noise + (1 - noise_scale) * latents


class LTX2TextInputStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` from `prompt_embeds`\n"
            "  2. Expands the text embeddings/masks by `num_videos_per_prompt`"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_attention_mask", type_hint=torch.Tensor, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype
        num_videos = block_state.num_videos_per_prompt

        def _expand(embeds, mask):
            _, seq_len, _ = embeds.shape
            embeds = embeds.repeat(1, num_videos, 1).view(block_state.batch_size * num_videos, seq_len, -1)
            mask = mask.repeat(num_videos, 1)
            return embeds, mask

        block_state.prompt_embeds, block_state.prompt_attention_mask = _expand(
            block_state.prompt_embeds, block_state.prompt_attention_mask
        )
        block_state.negative_prompt_embeds, block_state.negative_prompt_attention_mask = _expand(
            block_state.negative_prompt_embeds, block_state.negative_prompt_attention_mask
        )

        self.set_block_state(state, block_state)
        return components, state


class LTX2SetTimestepsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Sets the flow-match timesteps for the video `scheduler` and produces a deep-copied `audio_scheduler` "
            "(with the same schedule) so the audio latents are denoised on an independent scheduler state."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=40),
            InputParam.template("timesteps"),
            InputParam.template("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
            OutputParam(
                "audio_scheduler",
                description="Independent deep copy of `scheduler` used to update the audio latents in the loop.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        num_inference_steps = block_state.num_inference_steps
        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        # LTX-2 evaluates the shift at `max_image_seq_len`, which makes `mu` collapse to `max_shift` regardless of the
        # actual video sequence length (i.e. a resolution-independent constant shift). Ported as-is for parity.
        mu = calculate_shift(
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_image_seq_len", 1024),
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_shift", 0.95),
            components.scheduler.config.get("max_shift", 2.05),
        )

        block_state.audio_scheduler = copy.deepcopy(components.scheduler)
        retrieve_timesteps(
            block_state.audio_scheduler, num_inference_steps, device, block_state.timesteps, sigmas=sigmas, mu=mu
        )
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler, num_inference_steps, device, block_state.timesteps, sigmas=sigmas, mu=mu
        )

        # Set begin index to skip the nonzero().item() call in scheduler init, which triggers a GPU sync.
        components.scheduler.set_begin_index(0)
        block_state.audio_scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


class LTX2PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Prepares the packed video noise latents for text-to-video generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam.template("latents"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam("noise_scale", type_hint=float, default=0.0),
            InputParam.template("generator"),
            InputParam("batch_size", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="Packed noisy video latents.")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_channels_latents = components.transformer.config.in_channels
        spatial_patch = components.transformer_spatial_patch_size
        temporal_patch = components.transformer_temporal_patch_size

        if block_state.latents is not None:
            latents = block_state.latents
            if latents.ndim == 5:
                latents = _normalize_latents(
                    latents,
                    components.vae.latents_mean,
                    components.vae.latents_std,
                    components.vae.config.scaling_factor,
                )
                latents = _pack_latents(latents, spatial_patch, temporal_patch)
            latents = _create_noised_state(latents, block_state.noise_scale, block_state.generator)
            block_state.latents = latents.to(device=device, dtype=torch.float32)
        else:
            latent_height = block_state.height // components.vae_spatial_compression_ratio
            latent_width = block_state.width // components.vae_spatial_compression_ratio
            latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

            shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
            latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=torch.float32)
            block_state.latents = _pack_latents(latents, spatial_patch, temporal_patch)

        self.set_block_state(state, block_state)
        return components, state


class LTX2Image2VideoPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Prepares image-to-video latents: blends the pre-encoded `image_latents` (kept clean on the first latent "
            "frame) with the noise `latents` via a conditioning mask. Expects pure-noise `latents` from "
            "`LTX2PrepareLatentsStep`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image_latents", type_hint=torch.Tensor, required=True),
            InputParam.template("latents", required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam("batch_size", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="Packed noisy latents with image conditioning."
            ),
            OutputParam(
                "conditioning_mask",
                type_hint=torch.Tensor,
                description="Packed per-token mask marking the clean (image-conditioned) first latent frame.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        spatial_patch = components.transformer_spatial_patch_size
        temporal_patch = components.transformer_temporal_patch_size

        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

        init_latents = block_state.image_latents.to(device=device, dtype=torch.float32)
        if init_latents.shape[0] < batch_size:
            init_latents = init_latents.repeat_interleave(batch_size // init_latents.shape[0], dim=0)
        init_latents = init_latents.repeat(1, 1, latent_num_frames, 1, 1)

        conditioning_mask = torch.zeros(
            batch_size, 1, latent_num_frames, latent_height, latent_width, device=device, dtype=torch.float32
        )
        conditioning_mask[:, :, 0] = 1.0

        noise = _unpack_latents(
            block_state.latents, latent_num_frames, latent_height, latent_width, spatial_patch, temporal_patch
        )
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        block_state.conditioning_mask = _pack_latents(conditioning_mask, spatial_patch, temporal_patch).squeeze(-1)
        block_state.latents = _pack_latents(latents, spatial_patch, temporal_patch)

        self.set_block_state(state, block_state)
        return components, state


class LTX2PrepareAudioLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Prepares the packed audio noise latents and derives the audio latent frame count."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("audio_vae", AutoencoderKLLTX2Audio)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_frames", type_hint=int, default=121),
            InputParam("frame_rate", type_hint=float, default=24.0),
            InputParam("audio_latents", type_hint=torch.Tensor, default=None),
            InputParam("noise_scale", type_hint=float, default=0.0),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("generator"),
            InputParam("batch_size", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Packed noisy audio latents."),
            OutputParam("audio_num_frames", type_hint=int, description="Number of audio latent frames."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_channels_latents = components.audio_vae.config.latent_channels
        num_mel_bins = components.audio_vae.config.mel_bins

        duration_s = block_state.num_frames / block_state.frame_rate
        audio_latents_per_second = (
            components.audio_sampling_rate
            / components.audio_hop_length
            / float(components.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)

        if block_state.audio_latents is not None:
            audio_latents = block_state.audio_latents
            if audio_latents.ndim == 4:
                audio_num_frames = audio_latents.shape[2]
                audio_latents = _pack_audio_latents(audio_latents)
            audio_latents = _normalize_audio_latents(
                audio_latents, components.audio_vae.latents_mean, components.audio_vae.latents_std
            )
            audio_latents = _create_noised_state(audio_latents, block_state.noise_scale, block_state.generator)
            block_state.audio_latents = audio_latents.to(device=device, dtype=torch.float32)
        else:
            latent_mel_bins = num_mel_bins // components.audio_vae_mel_compression_ratio
            shape = (batch_size, num_channels_latents, audio_num_frames, latent_mel_bins)
            audio_latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=torch.float32)
            block_state.audio_latents = _pack_audio_latents(audio_latents)

        block_state.audio_num_frames = audio_num_frames

        self.set_block_state(state, block_state)
        return components, state


class LTX2PrepareCoordsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Pre-computes the video and audio RoPE positional coordinates (constant across denoising steps). CFG/STG "
            "batch duplication is left to the denoise step."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam("frame_rate", type_hint=float, default=24.0),
            InputParam("audio_num_frames", type_hint=int, required=True),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam("batch_size", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("video_coords", type_hint=torch.Tensor, description="Video RoPE patch coordinates."),
            OutputParam("audio_coords", type_hint=torch.Tensor, description="Audio RoPE patch coordinates."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

        block_state.video_coords = components.transformer.rope.prepare_video_coords(
            batch_size, latent_num_frames, latent_height, latent_width, device, fps=block_state.frame_rate
        )
        block_state.audio_coords = components.transformer.audio_rope.prepare_audio_coords(
            batch_size, block_state.audio_num_frames, device
        )

        self.set_block_state(state, block_state)
        return components, state
