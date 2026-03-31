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

import copy
import inspect

import numpy as np
import torch

from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...models.transformers import LTX2VideoTransformer3DModel
from ...pipelines.ltx2.connectors import LTX2TextConnectors
from ...pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
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
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size, -1, post_patch_num_frames, patch_size_t, post_patch_height, patch_size, post_patch_width, patch_size
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


def _pack_audio_latents(
    latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
        post_patch_latent_length = latent_length / patch_size_t
        post_patch_mel_bins = latent_mel_bins / patch_size
        latents = latents.reshape(
            batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
    else:
        latents = latents.transpose(1, 2).flatten(2, 3)
    return latents


def _normalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


class LTX2InputStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Input processing step that determines batch_size and dtype, "
            "and expands embeddings for num_videos_per_prompt"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_videos_per_prompt", default=1),
            InputParam("connector_prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("connector_audio_prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("connector_attention_mask", required=True, type_hint=torch.Tensor),
            InputParam("connector_negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("connector_audio_negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("connector_negative_attention_mask", type_hint=torch.Tensor),
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

        block_state.batch_size = block_state.connector_prompt_embeds.shape[0]
        block_state.dtype = components.transformer.dtype

        self.set_block_state(state, block_state)
        return components, state


class LTX2SetTimestepsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Step that sets up the scheduler timesteps for both video and audio denoising"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("vae", AutoencoderKLLTX2Video),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_inference_steps", default=40),
            InputParam("timesteps_input"),
            InputParam("sigmas"),
            InputParam("height", default=512, type_hint=int),
            InputParam("width", default=768, type_hint=int),
            InputParam("num_frames", default=121, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
            OutputParam("audio_scheduler"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        num_inference_steps = block_state.num_inference_steps
        sigmas = block_state.sigmas
        timesteps_input = block_state.timesteps_input

        vae_spatial_compression_ratio = components.vae.spatial_compression_ratio
        vae_temporal_compression_ratio = components.vae.temporal_compression_ratio
        height = block_state.height
        width = block_state.width
        num_frames = block_state.num_frames
        latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1
        latent_height = height // vae_spatial_compression_ratio
        latent_width = width // vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width

        if sigmas is None:
            # Use torch.linspace (float32) to match reference scheduler precision.
            # np.linspace computes in float64 then casts to float32, which produces
            # values that differ by 1 ULP from torch's native float32 computation.
            sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1].numpy()

        mu = calculate_shift(
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_image_seq_len", 1024),
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_shift", 0.95),
            components.scheduler.config.get("max_shift", 2.05),
        )

        audio_scheduler = copy.deepcopy(components.scheduler)
        _, _ = retrieve_timesteps(
            audio_scheduler, num_inference_steps, device, timesteps_input, sigmas=sigmas, mu=mu
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler, num_inference_steps, device, timesteps_input, sigmas=sigmas, mu=mu
        )

        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.audio_scheduler = audio_scheduler

        self.set_block_state(state, block_state)
        return components, state


class LTX2PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Prepare video latents, optionally applying conditioning mask for I2V/conditional generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("height", default=512, type_hint=int),
            InputParam("width", default=768, type_hint=int),
            InputParam("num_frames", default=121, type_hint=int),
            InputParam("noise_scale", default=1.0, type_hint=float),
            InputParam("latents", type_hint=torch.Tensor),
            InputParam("generator"),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("num_videos_per_prompt", default=1, type_hint=int),
            InputParam("condition_latents", type_hint=list),
            InputParam("condition_strengths", type_hint=list),
            InputParam("condition_indices", type_hint=list),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("conditioning_mask", type_hint=torch.Tensor),
            OutputParam("clean_latents", type_hint=torch.Tensor),
            OutputParam("latent_num_frames", type_hint=int),
            OutputParam("latent_height", type_hint=int),
            OutputParam("latent_width", type_hint=int),
            OutputParam("video_sequence_length", type_hint=int),
            OutputParam("transformer_spatial_patch_size", type_hint=int),
            OutputParam("transformer_temporal_patch_size", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        height = block_state.height
        width = block_state.width
        num_frames = block_state.num_frames
        noise_scale = block_state.noise_scale
        generator = block_state.generator
        batch_size = block_state.batch_size * block_state.num_videos_per_prompt

        vae_spatial_compression_ratio = components.vae.spatial_compression_ratio
        vae_temporal_compression_ratio = components.vae.temporal_compression_ratio
        transformer_spatial_patch_size = components.transformer.config.patch_size
        transformer_temporal_patch_size = components.transformer.config.patch_size_t
        num_channels_latents = components.transformer.config.in_channels

        latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1
        latent_height = height // vae_spatial_compression_ratio
        latent_width = width // vae_spatial_compression_ratio

        condition_latents = block_state.condition_latents or []
        condition_strengths = block_state.condition_strengths or []
        condition_indices = block_state.condition_indices or []
        has_conditions = len(condition_latents) > 0

        if block_state.latents is not None:
            latents = block_state.latents
            if latents.ndim == 5:
                latents = _normalize_latents(
                    latents, components.vae.latents_mean, components.vae.latents_std, components.vae.config.scaling_factor
                )
                _, _, latent_num_frames, latent_height, latent_width = latents.shape
                latents = _pack_latents(latents, transformer_spatial_patch_size, transformer_temporal_patch_size)
        else:
            # Reference: create zeros in [B,C,F,H,W] in model dtype, pack to [B,S,D],
            # then generate noise in packed shape with same dtype
            latent_dtype = components.transformer.dtype
            shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
            latents = torch.zeros(shape, device=device, dtype=latent_dtype)
            latents = _pack_latents(latents, transformer_spatial_patch_size, transformer_temporal_patch_size)

        conditioning_mask = None
        clean_latents = None

        if has_conditions:
            mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)
            conditioning_mask = torch.zeros(mask_shape, device=device, dtype=torch.float32)
            conditioning_mask = _pack_latents(
                conditioning_mask, transformer_spatial_patch_size, transformer_temporal_patch_size
            )

            clean_latents = torch.zeros_like(latents)
            for cond, strength, latent_idx in zip(condition_latents, condition_strengths, condition_indices):
                num_cond_tokens = cond.size(1)
                start_token_idx = latent_idx * latent_height * latent_width
                end_token_idx = start_token_idx + num_cond_tokens

                latents[:, start_token_idx:end_token_idx] = cond
                conditioning_mask[:, start_token_idx:end_token_idx] = strength
                clean_latents[:, start_token_idx:end_token_idx] = cond

            if isinstance(generator, list):
                generator = generator[0]

            # Noise in packed [B,S,D] shape and same dtype as latent (matches reference GaussianNoiser)
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            scaled_mask = (1.0 - conditioning_mask) * noise_scale
            latents = noise * scaled_mask + latents * (1 - scaled_mask)
        else:
            # T2V: noise in packed shape, same dtype as latent
            if isinstance(generator, list):
                generator = generator[0]
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            scaled_mask = noise_scale
            latents = noise * scaled_mask + latents * (1 - scaled_mask)

        block_state.latents = latents
        block_state.conditioning_mask = conditioning_mask
        block_state.clean_latents = clean_latents
        block_state.latent_num_frames = latent_num_frames
        block_state.latent_height = latent_height
        block_state.latent_width = latent_width
        block_state.video_sequence_length = latent_num_frames * latent_height * latent_width
        block_state.transformer_spatial_patch_size = transformer_spatial_patch_size
        block_state.transformer_temporal_patch_size = transformer_temporal_patch_size

        self.set_block_state(state, block_state)
        return components, state


class LTX2PrepareAudioLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Prepare audio latents for the denoising process"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_frames", default=121, type_hint=int),
            InputParam("frame_rate", default=24.0, type_hint=float),
            InputParam("noise_scale", default=1.0, type_hint=float),
            InputParam("audio_latents", type_hint=torch.Tensor),
            InputParam("generator"),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("num_videos_per_prompt", default=1, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("audio_latents", type_hint=torch.Tensor),
            OutputParam("audio_num_frames", type_hint=int),
            OutputParam("latent_mel_bins", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        num_frames = block_state.num_frames
        frame_rate = block_state.frame_rate
        noise_scale = block_state.noise_scale
        generator = block_state.generator
        batch_size = block_state.batch_size * block_state.num_videos_per_prompt

        audio_sampling_rate = components.audio_vae.config.sample_rate
        audio_hop_length = components.audio_vae.config.mel_hop_length
        audio_vae_temporal_compression_ratio = components.audio_vae.temporal_compression_ratio
        audio_vae_mel_compression_ratio = components.audio_vae.mel_compression_ratio

        duration_s = num_frames / frame_rate
        audio_latents_per_second = audio_sampling_rate / audio_hop_length / float(audio_vae_temporal_compression_ratio)
        audio_num_frames = round(duration_s * audio_latents_per_second)

        num_mel_bins = components.audio_vae.config.mel_bins
        latent_mel_bins = num_mel_bins // audio_vae_mel_compression_ratio
        num_channels_latents_audio = components.audio_vae.config.latent_channels

        if block_state.audio_latents is not None:
            audio_latents = block_state.audio_latents
            if audio_latents.ndim == 4:
                _, _, audio_num_frames, _ = audio_latents.shape
                audio_latents = _pack_audio_latents(audio_latents)
                audio_latents = _normalize_audio_latents(
                    audio_latents, components.audio_vae.latents_mean, components.audio_vae.latents_std
                )
                if noise_scale > 0.0:
                    noise = randn_tensor(
                        audio_latents.shape, generator=generator, device=audio_latents.device, dtype=audio_latents.dtype
                    )
                    audio_latents = noise_scale * noise + (1 - noise_scale) * audio_latents
            elif audio_latents.ndim == 3 and noise_scale > 0.0:
                noise = randn_tensor(
                    audio_latents.shape, generator=generator, device=audio_latents.device, dtype=audio_latents.dtype
                )
                audio_latents = noise_scale * noise + (1 - noise_scale) * audio_latents
            audio_latents = audio_latents.to(device=device, dtype=torch.float32)
        else:
            # Reference: create zeros in [B,C,T,M] in model dtype, pack, then noise in packed shape
            latent_dtype = components.audio_vae.dtype
            shape = (batch_size, num_channels_latents_audio, audio_num_frames, latent_mel_bins)
            audio_latents = torch.zeros(shape, device=device, dtype=latent_dtype)
            audio_latents = _pack_audio_latents(audio_latents)
            if isinstance(generator, list):
                generator = generator[0]
            noise = randn_tensor(audio_latents.shape, generator=generator, device=device, dtype=latent_dtype)
            audio_latents = noise * noise_scale + audio_latents * (1 - noise_scale)

        block_state.audio_latents = audio_latents
        block_state.audio_num_frames = audio_num_frames
        block_state.latent_mel_bins = latent_mel_bins

        self.set_block_state(state, block_state)
        return components, state


class LTX2PrepareCoordinatesStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Prepare video and audio RoPE coordinates for positional encoding"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("audio_latents", required=True, type_hint=torch.Tensor),
            InputParam("latent_num_frames", required=True, type_hint=int),
            InputParam("latent_height", required=True, type_hint=int),
            InputParam("latent_width", required=True, type_hint=int),
            InputParam("audio_num_frames", required=True, type_hint=int),
            InputParam("frame_rate", default=24.0, type_hint=float),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("video_coords", type_hint=torch.Tensor),
            OutputParam("audio_coords", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents
        audio_latents = block_state.audio_latents
        frame_rate = block_state.frame_rate

        video_coords = components.transformer.rope.prepare_video_coords(
            latents.shape[0],
            block_state.latent_num_frames,
            block_state.latent_height,
            block_state.latent_width,
            latents.device,
            fps=frame_rate,
        )
        # Cast to latent dtype to match reference (positions stored in model dtype)
        video_coords = video_coords.to(latents.dtype)
        audio_coords = components.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], block_state.audio_num_frames, audio_latents.device
        )
        # Note: audio_coords already match reference dtype, no cast needed

        block_state.video_coords = video_coords
        block_state.audio_coords = audio_coords

        self.set_block_state(state, block_state)
        return components, state


class LTX2Stage2SetTimestepsStep(LTX2SetTimestepsStep):
    """SetTimesteps for Stage 2: fixed distilled sigmas, no dynamic shifting."""

    @property
    def description(self) -> str:
        return "Stage 2 timestep setup: uses fixed distilled sigmas with dynamic shifting disabled"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_inference_steps", default=3),
            InputParam("timesteps_input"),
            InputParam("sigmas", default=list(STAGE_2_DISTILLED_SIGMA_VALUES)),
            InputParam("height", default=512, type_hint=int),
            InputParam("width", default=768, type_hint=int),
            InputParam("num_frames", default=121, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        components.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            components.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )
        return super().__call__(components, state)


class LTX2Stage2PrepareLatentsStep(LTX2PrepareLatentsStep):
    """PrepareLatents for Stage 2: noise_scale defaults to first distilled sigma value."""

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("height", default=512, type_hint=int),
            InputParam("width", default=768, type_hint=int),
            InputParam("num_frames", default=121, type_hint=int),
            InputParam("noise_scale", default=STAGE_2_DISTILLED_SIGMA_VALUES[0], type_hint=float),
            InputParam("latents", type_hint=torch.Tensor),
            InputParam("generator"),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("num_videos_per_prompt", default=1, type_hint=int),
            InputParam("condition_latents", type_hint=list),
            InputParam("condition_strengths", type_hint=list),
            InputParam("condition_indices", type_hint=list),
        ]


class LTX2DisableAdapterStep(ModularPipelineBlocks):
    """Disables LoRA adapters on transformer and connectors. No-op if no adapters are loaded."""

    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Disable LoRA adapters before stage 1 (no-op if no adapters loaded)"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("connectors", LTX2TextConnectors),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        for model in [components.transformer, components.connectors]:
            if getattr(model, "_hf_peft_config_loaded", False):
                model.disable_adapters()
        self.set_block_state(state, block_state)
        return components, state


class LTX2EnableAdapterStep(ModularPipelineBlocks):
    """Enables LoRA adapters by name before stage 2. No-op if stage_2_adapter is not provided."""

    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Enable LoRA adapters before stage 2 (no-op if stage_2_adapter not provided)"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("connectors", LTX2TextConnectors),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("stage_2_adapter", type_hint=str, description="Name of the LoRA adapter to enable for stage 2"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        adapter_name = block_state.stage_2_adapter
        if adapter_name is not None:
            for model in [components.transformer, components.connectors]:
                if getattr(model, "_hf_peft_config_loaded", False):
                    model.enable_adapters()
        self.set_block_state(state, block_state)
        return components, state
