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

from ...models import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel

# NOTE (modular.md gotcha #1): `LTX2ReferenceCondition` is a plain dataclass under `diffusers.pipelines.ltx2.*`, and
# modular blocks must not import from `diffusers.pipelines.*`. It belongs in the same neutral-module relocation as
# the other shared LTX-2 data/utilities enumerated in `encoders.py`. Imported from the pipelines path here only so
# the draft is runnable.
from ...pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
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


def _prepare_keyframe_coords(
    keyframe_latent_num_frames: int,
    keyframe_latent_height: int,
    keyframe_latent_width: int,
    pixel_frame_idx: int,
    num_pixel_frames: int,
    fps: float,
    patch_size: int,
    patch_size_t: int,
    scale_factors: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Positional coordinates for a keyframe condition appended as extra tokens, mirroring
    `VideoConditionByKeyframeIndex.apply_to` in the reference implementation:
      - latent coords scaled to pixel space *without* the causal fix (non-zero-index keyframes don't need the
        first-frame causal adjustment),
      - temporal axis offset by `pixel_frame_idx` (the pixel-space index the keyframe appears at),
      - for single-pixel-frame keyframes the per-patch temporal extent is clamped to `[idx, idx + 1)` so the keyframe
        occupies one pixel timestep rather than the VAE-scaled range,
      - temporal coords divided by `fps` to give seconds.
    """
    grid_f = torch.arange(
        start=0, end=keyframe_latent_num_frames, step=patch_size_t, dtype=torch.float32, device=device
    )
    grid_h = torch.arange(start=0, end=keyframe_latent_height, step=patch_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(start=0, end=keyframe_latent_width, step=patch_size, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), dim=0)

    patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
    patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

    latent_coords = torch.stack([grid, patch_ends], dim=-1)  # [3, N_F, N_H, N_W, 2]
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0)  # [1, 3, num_patches, 2]

    scale_tensor = torch.tensor(scale_factors, device=device, dtype=latent_coords.dtype)
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] + pixel_frame_idx
    if num_pixel_frames == 1:
        pixel_coords[:, 0, :, 1:] = pixel_coords[:, 0, :, :1] + 1
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    return pixel_coords


class LTX2TextInputStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Input processing step that expands the connector text conditioning (cond and uncond) by "
            "`num_videos_per_prompt`, so it matches the `batch_size * num_videos_per_prompt` batch of the video and "
            "audio latents. Runs at the head of the denoise stage, which keeps the text-conditioning stage's outputs "
            "reusable across denoise runs with different `num_videos_per_prompt`."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam(
                "connector_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Video-branch text conditioning (cond).",
            ),
            InputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Audio-branch text conditioning (cond).",
            ),
            InputParam(
                "connector_attention_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Binary text attention mask (cond).",
            ),
            InputParam(
                "negative_connector_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Video-branch text conditioning (uncond).",
            ),
            InputParam(
                "negative_connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Audio-branch text conditioning (uncond).",
            ),
            InputParam(
                "negative_connector_attention_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Binary text attention mask (uncond).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "connector_prompt_embeds",
                type_hint=torch.Tensor,
                description="Video-branch text conditioning (cond), expanded per prompt.",
            ),
            OutputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                description="Audio-branch text conditioning (cond), expanded per prompt.",
            ),
            OutputParam(
                "connector_attention_mask",
                type_hint=torch.Tensor,
                description="Binary text attention mask (cond), expanded per prompt.",
            ),
            OutputParam(
                "negative_connector_prompt_embeds",
                type_hint=torch.Tensor,
                description="Video-branch text conditioning (uncond), expanded per prompt.",
            ),
            OutputParam(
                "negative_connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                description="Audio-branch text conditioning (uncond), expanded per prompt.",
            ),
            OutputParam(
                "negative_connector_attention_mask",
                type_hint=torch.Tensor,
                description="Binary text attention mask (uncond), expanded per prompt.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # `repeat_interleave` keeps each prompt's copies contiguous, matching how the latents are laid out
        # (`batch_size * num_videos_per_prompt`, prompt-major) and how `image_latents` are expanded downstream.
        num_videos = block_state.num_videos_per_prompt
        for name in self.intermediate_output_names:
            setattr(block_state, name, getattr(block_state, name).repeat_interleave(num_videos, dim=0))

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
            InputParam.template("num_inference_steps", default=30),
            InputParam.template("timesteps"),
            InputParam.template("sigmas"),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
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

        # Resolution-aware (dynamic) shift: `mu` is computed from the actual packed video sequence length, matching
        # `latents.shape[1]` in the standard pipeline. The packed length divides the latent grid by the transformer
        # patch sizes (see `_pack_latents`). `num_frames` must already be a concrete int here -- when auto-duration is
        # used, the duration step resolves it upstream, before this block runs.
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        video_seq_len = (
            (latent_num_frames // components.transformer_temporal_patch_size)
            * (latent_height // components.transformer_spatial_patch_size)
            * (latent_width // components.transformer_spatial_patch_size)
        )
        mu = calculate_shift(
            video_seq_len,
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
        return (
            "Prepares the packed video noise latents for text-to-video generation. `noise_scale` is declared with a "
            "`None` default and resolved to 0.0 here rather than declared as 0.0: a blockset keeps the first "
            "non-`None` default across its blocks, so a literal 0.0 would shadow the condition workflow's "
            "`None -> sigmas[0] or 1.0` resolution wherever the two share a blockset (`LTX2AutoBlocks`). The "
            "resolved value is written back to state for `LTX2PrepareAudioLatentsStep`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam.template("latents"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=None,
                description=(
                    "Interpolation factor between random noise and any provided latents. `None` (default) resolves "
                    "to 0.0, which keeps the provided latents."
                ),
            ),
            InputParam.template("generator"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Packed noisy video latents."),
            OutputParam(
                "noise_scale",
                type_hint=float,
                description="The resolved interpolation factor, forwarded to the audio latents step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_channels_latents = components.transformer.config.in_channels
        spatial_patch = components.transformer_spatial_patch_size
        temporal_patch = components.transformer_temporal_patch_size

        if block_state.noise_scale is None:
            block_state.noise_scale = 0.0

        if block_state.latents is not None:
            latents = block_state.latents
            if latents.ndim == 5:
                latents = _normalize_latents(
                    latents,
                    components.latents_mean,
                    components.latents_std,
                    components.vae_scaling_factor,
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
            InputParam(
                "image_latents",
                type_hint=torch.Tensor,
                required=True,
                description="VAE-encoded reference-image latents used for image-to-video conditioning.",
            ),
            InputParam.template("latents", required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
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
        return (
            "Prepares the packed audio noise latents and derives the audio latent frame count. `noise_scale` is "
            "declared with a `None` default for the reason given on `LTX2PrepareLatentsStep`, which runs first and "
            "writes the resolved value back to state."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("audio_vae", AutoencoderKLLTX2Audio)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                default=None,
                description="Optional pre-encoded audio latents; random noise is used when not provided.",
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=None,
                description=(
                    "Interpolation factor between random noise and any provided `audio_latents`. Resolved upstream "
                    "by `LTX2PrepareLatentsStep` (0.0, which keeps the provided latents)."
                ),
            ),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("generator"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Packed noisy audio latents."),
            OutputParam(
                "audio_num_frames",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Number of audio latent frames.",
            ),
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
                audio_latents, components.audio_latents_mean, components.audio_latents_std
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
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam("audio_num_frames", type_hint=int, required=True),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "video_coords",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Video RoPE patch coordinates.",
            ),
            OutputParam(
                "audio_coords",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Audio RoPE patch coordinates.",
            ),
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


class LTX2ConditionPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Prepares the packed video latents for condition-based generation. Conditions at latent index 0 "
            "overwrite the tokens at the first-frame positions (`VideoConditionByLatentIndex` semantics); conditions "
            "at other latent indices are appended to the sequence as keyframe tokens carrying their own RoPE "
            "coordinates (`VideoConditionByKeyframeIndex` semantics). Emits the per-token `conditioning_mask` and "
            "`clean_latents` that drive the loop's masked timestep and x0 blend. Unlike `LTX2PrepareLatentsStep` "
            "this starts from zeros and samples the noise last, once the conditioning mask is known."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("vae", AutoencoderKLLTX2Video),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "condition_latents",
                type_hint=list,
                required=True,
                description="Per-condition normalized VAE latents of shape [1, C, F, H, W].",
            ),
            InputParam(
                "condition_strengths",
                type_hint=list,
                required=True,
                description="Per-condition conditioning strengths.",
            ),
            InputParam(
                "condition_indices",
                type_hint=list,
                required=True,
                description="Per-condition latent frame index at which the condition is applied.",
            ),
            InputParam(
                "condition_pixel_frames",
                type_hint=list,
                required=True,
                description="Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.",
            ),
            InputParam.template("latents"),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=None,
                description=(
                    "Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` "
                    "when custom `sigmas` are supplied, else 1.0."
                ),
            ),
            InputParam.template("sigmas"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Packed noisy video latents, with any keyframe condition tokens appended.",
            ),
            OutputParam(
                "conditioning_mask",
                type_hint=torch.Tensor,
                description=(
                    "Packed per-token conditioning strengths of shape [B, S, 1] in [0, 1]: 1 at fully-conditioned "
                    "positions, 0 at free positions."
                ),
            ),
            OutputParam(
                "clean_latents",
                type_hint=torch.Tensor,
                description="Clean condition latents at conditioned positions, zeros elsewhere; same shape as `latents`.",
            ),
            OutputParam(
                "appended_coords",
                type_hint=torch.Tensor,
                description=(
                    "RoPE coordinates of shape [B, 3, num_keyframe_tokens, 2] for the appended keyframe tokens, "
                    "zero-width when there are none."
                ),
            ),
            OutputParam(
                "base_token_count",
                type_hint=int,
                description="Number of generated-video tokens, i.e. the sequence length before appended tokens.",
            ),
            OutputParam(
                "noise_scale",
                type_hint=float,
                description="The resolved initial noise level, forwarded to the audio latents step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        spatial_patch = components.transformer_spatial_patch_size
        temporal_patch = components.transformer_temporal_patch_size
        frame_scale_factor = components.vae_temporal_compression_ratio

        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        latent_num_frames = (block_state.num_frames - 1) // frame_scale_factor + 1

        # Noise level the un-conditioned tokens start at: the first (largest) sigma when custom sigmas are supplied,
        # else 1.0. Matches `LTX2ConditionPipeline.__call__`.
        noise_scale = block_state.noise_scale
        if noise_scale is None:
            noise_scale = block_state.sigmas[0] if block_state.sigmas is not None else 1.0

        if isinstance(block_state.generator, list):
            logger.warning(
                f"{self.__class__.__name__} does not support using a list of generators. The first generator in the"
                f" list will be used for all (pseudo-)random operations."
            )

        if block_state.latents is not None:
            latents = _normalize_latents(
                block_state.latents,
                components.latents_mean,
                components.latents_std,
                components.vae_scaling_factor,
            )
        else:
            # Zeros rather than a Gaussian sample: the noise is mixed in at the end, once the mask is known.
            shape = (
                batch_size,
                components.transformer.config.in_channels,
                latent_num_frames,
                latent_height,
                latent_width,
            )
            latents = torch.zeros(shape, device=device, dtype=torch.float32)

        conditioning_mask = latents.new_zeros((batch_size, 1, latent_num_frames, latent_height, latent_width))
        latents = _pack_latents(latents, spatial_patch, temporal_patch)
        conditioning_mask = _pack_latents(conditioning_mask, spatial_patch, temporal_patch)  # [B, S, 1]

        if latents.ndim != 3 or latents.shape[:2] != conditioning_mask.shape[:2]:
            raise ValueError(
                f"Provided `latents` tensor packs to shape {latents.shape}, but the expected packed shape is "
                f"{conditioning_mask.shape[:2] + (components.transformer.config.in_channels,)}."
            )

        base_token_count = latents.shape[1]
        condition_latents_packed = [
            _pack_latents(cond, spatial_patch, temporal_patch) for cond in block_state.condition_latents
        ]

        # First-frame conditions (latent index 0): overwrite the tokens at the first-frame positions. Condition
        # tensors carry batch 1 and broadcast across the generation batch.
        clean_latents = torch.zeros_like(latents)
        for cond, strength, latent_idx in zip(
            condition_latents_packed, block_state.condition_strengths, block_state.condition_indices
        ):
            if latent_idx != 0:
                continue
            num_cond_tokens = cond.size(1)
            latents[:, :num_cond_tokens] = cond
            conditioning_mask[:, :num_cond_tokens] = strength
            clean_latents[:, :num_cond_tokens] = cond

        # Non-first-frame ("keyframe") conditions (latent index > 0): append as extra tokens with an all-`strength`
        # conditioning mask and their own coords. At denoising step i they see an effective noise level of
        # (1 - strength) * sigma_i.
        scale_factors = (
            frame_scale_factor,
            components.vae_spatial_compression_ratio,
            components.vae_spatial_compression_ratio,
        )
        keyframe_tokens, keyframe_masks, keyframe_coords = [], [], []
        for cond_5d, cond_packed, strength, latent_idx, num_pixel_frames in zip(
            block_state.condition_latents,
            condition_latents_packed,
            block_state.condition_strengths,
            block_state.condition_indices,
            block_state.condition_pixel_frames,
        ):
            if latent_idx == 0:
                continue

            _, _, kf_latent_frames, kf_latent_height, kf_latent_width = cond_5d.shape
            coords = _prepare_keyframe_coords(
                keyframe_latent_num_frames=kf_latent_frames,
                keyframe_latent_height=kf_latent_height,
                keyframe_latent_width=kf_latent_width,
                pixel_frame_idx=(latent_idx - 1) * frame_scale_factor + 1,
                num_pixel_frames=num_pixel_frames,
                fps=block_state.frame_rate,
                patch_size=spatial_patch,
                patch_size_t=temporal_patch,
                scale_factors=scale_factors,
                device=device,
            )

            keyframe_tokens.append(cond_packed.expand(batch_size, -1, -1))
            keyframe_masks.append(
                torch.full(
                    (batch_size, cond_packed.shape[1], 1),
                    float(strength),
                    device=device,
                    dtype=conditioning_mask.dtype,
                )
            )
            keyframe_coords.append(coords.expand(batch_size, -1, -1, -1))

        if keyframe_tokens:
            keyframe_tokens = torch.cat(keyframe_tokens, dim=1)
            latents = torch.cat([latents, keyframe_tokens], dim=1)
            clean_latents = torch.cat([clean_latents, keyframe_tokens], dim=1)
            conditioning_mask = torch.cat([conditioning_mask, torch.cat(keyframe_masks, dim=1)], dim=1)
            appended_coords = torch.cat(keyframe_coords, dim=2)
        else:
            appended_coords = torch.zeros((batch_size, 3, 0, 2), device=device, dtype=torch.float32)

        # Mask semantics: 0 -> fully noised, 1 -> kept clean, in between -> noise level (1 - mask) * noise_scale.
        noise = randn_tensor(
            latents.shape, generator=block_state.generator, device=latents.device, dtype=latents.dtype
        )
        scaled_mask = (1.0 - conditioning_mask) * noise_scale
        block_state.latents = noise * scaled_mask + latents * (1 - scaled_mask)

        block_state.conditioning_mask = conditioning_mask
        block_state.clean_latents = clean_latents
        block_state.appended_coords = appended_coords
        block_state.base_token_count = base_token_count
        block_state.noise_scale = noise_scale

        self.set_block_state(state, block_state)
        return components, state


class LTX2InContextPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Prepares the packed video latents for in-context (IC-LoRA) generation. Same frame-condition handling as "
            "`LTX2ConditionPrepareLatentsStep` (first-frame overwrite, keyframe token append), then appends the "
            "encoded reference tokens after the keyframes with a per-token `conditioning_mask` of their own "
            "strength, giving a single `[base | keyframe | reference]` sequence. Mirrors "
            "`LTX2InContextPipeline.prepare_latents`, which likewise re-implements the condition version rather "
            "than extending it -- the two are kept side by side so each reads top to bottom."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("vae", AutoencoderKLLTX2Video),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "condition_latents",
                type_hint=list,
                required=True,
                description="Per-condition normalized VAE latents of shape [1, C, F, H, W].",
            ),
            InputParam(
                "condition_strengths",
                type_hint=list,
                required=True,
                description="Per-condition conditioning strengths.",
            ),
            InputParam(
                "condition_indices",
                type_hint=list,
                required=True,
                description="Per-condition latent frame index at which the condition is applied.",
            ),
            InputParam(
                "condition_pixel_frames",
                type_hint=list,
                required=True,
                description="Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.",
            ),
            InputParam(
                "reference_conditions",
                type_hint=list,
                default=None,
                description=(
                    "`LTX2ReferenceCondition` (or list of them); only their `strength` is read here. Omit for "
                    "IC-LoRAs that carry their behavior in the adapter weights and take no reference video."
                ),
            ),
            InputParam(
                "reference_latents",
                type_hint=torch.Tensor,
                default=None,
                description=(
                    "Packed reference tokens of shape [1, total_reference_tokens, C], or `None` when no reference "
                    "conditions were supplied (`LTX2AutoReferenceEncoderStep` is skipped)."
                ),
            ),
            InputParam(
                "reference_coords",
                type_hint=torch.Tensor,
                default=None,
                description="RoPE coordinates for the reference tokens.",
            ),
            InputParam(
                "reference_token_counts",
                type_hint=list,
                default=None,
                description="Per-reference token counts, in `reference_conditions` order.",
            ),
            InputParam.template("latents"),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=None,
                description=(
                    "Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` "
                    "when custom `sigmas` are supplied, else 1.0."
                ),
            ),
            InputParam.template("sigmas"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Packed noisy video latents, with keyframe and reference tokens appended.",
            ),
            OutputParam(
                "conditioning_mask",
                type_hint=torch.Tensor,
                description=(
                    "Packed per-token conditioning strengths of shape [B, S, 1] in [0, 1]: 1 at fully-conditioned "
                    "positions, 0 at free positions."
                ),
            ),
            OutputParam(
                "clean_latents",
                type_hint=torch.Tensor,
                description="Clean condition latents at conditioned positions, zeros elsewhere; same shape as `latents`.",
            ),
            OutputParam(
                "appended_coords",
                type_hint=torch.Tensor,
                description=(
                    "RoPE coordinates of shape [B, 3, num_keyframe_tokens + num_reference_tokens, 2] for the "
                    "appended tokens, zero-width when there are none."
                ),
            ),
            OutputParam(
                "base_token_count",
                type_hint=int,
                description="Number of generated-video tokens, i.e. the sequence length before appended tokens.",
            ),
            OutputParam(
                "num_ref_tokens",
                type_hint=int,
                description="Number of reference tokens, which sit at the very end of the sequence.",
            ),
            OutputParam(
                "noise_scale",
                type_hint=float,
                description="The resolved initial noise level, forwarded to the audio latents step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        spatial_patch = components.transformer_spatial_patch_size
        temporal_patch = components.transformer_temporal_patch_size
        frame_scale_factor = components.vae_temporal_compression_ratio

        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        latent_num_frames = (block_state.num_frames - 1) // frame_scale_factor + 1

        noise_scale = block_state.noise_scale
        if noise_scale is None:
            noise_scale = block_state.sigmas[0] if block_state.sigmas is not None else 1.0

        if isinstance(block_state.generator, list):
            logger.warning(
                f"{self.__class__.__name__} does not support using a list of generators. The first generator in the"
                f" list will be used for all (pseudo-)random operations."
            )

        if block_state.latents is not None:
            latents = _normalize_latents(
                block_state.latents,
                components.latents_mean,
                components.latents_std,
                components.vae_scaling_factor,
            )
        else:
            shape = (
                batch_size,
                components.transformer.config.in_channels,
                latent_num_frames,
                latent_height,
                latent_width,
            )
            latents = torch.zeros(shape, device=device, dtype=torch.float32)

        conditioning_mask = latents.new_zeros((batch_size, 1, latent_num_frames, latent_height, latent_width))
        latents = _pack_latents(latents, spatial_patch, temporal_patch)
        conditioning_mask = _pack_latents(conditioning_mask, spatial_patch, temporal_patch)  # [B, S, 1]

        if latents.ndim != 3 or latents.shape[:2] != conditioning_mask.shape[:2]:
            raise ValueError(
                f"Provided `latents` tensor packs to shape {latents.shape}, but the expected packed shape is "
                f"{conditioning_mask.shape[:2] + (components.transformer.config.in_channels,)}."
            )

        base_token_count = latents.shape[1]
        condition_latents_packed = [
            _pack_latents(cond, spatial_patch, temporal_patch) for cond in block_state.condition_latents
        ]

        # First-frame conditions (latent index 0): overwrite the tokens at the first-frame positions.
        clean_latents = torch.zeros_like(latents)
        for cond, strength, latent_idx in zip(
            condition_latents_packed, block_state.condition_strengths, block_state.condition_indices
        ):
            if latent_idx != 0:
                continue
            num_cond_tokens = cond.size(1)
            latents[:, :num_cond_tokens] = cond
            conditioning_mask[:, :num_cond_tokens] = strength
            clean_latents[:, :num_cond_tokens] = cond

        # Non-first-frame ("keyframe") conditions (latent index > 0): append as extra tokens with their own coords.
        scale_factors = (
            frame_scale_factor,
            components.vae_spatial_compression_ratio,
            components.vae_spatial_compression_ratio,
        )
        keyframe_tokens, keyframe_masks, keyframe_coords = [], [], []
        for cond_5d, cond_packed, strength, latent_idx, num_pixel_frames in zip(
            block_state.condition_latents,
            condition_latents_packed,
            block_state.condition_strengths,
            block_state.condition_indices,
            block_state.condition_pixel_frames,
        ):
            if latent_idx == 0:
                continue

            _, _, kf_latent_frames, kf_latent_height, kf_latent_width = cond_5d.shape
            coords = _prepare_keyframe_coords(
                keyframe_latent_num_frames=kf_latent_frames,
                keyframe_latent_height=kf_latent_height,
                keyframe_latent_width=kf_latent_width,
                pixel_frame_idx=(latent_idx - 1) * frame_scale_factor + 1,
                num_pixel_frames=num_pixel_frames,
                fps=block_state.frame_rate,
                patch_size=spatial_patch,
                patch_size_t=temporal_patch,
                scale_factors=scale_factors,
                device=device,
            )

            keyframe_tokens.append(cond_packed.expand(batch_size, -1, -1))
            keyframe_masks.append(
                torch.full(
                    (batch_size, cond_packed.shape[1], 1),
                    float(strength),
                    device=device,
                    dtype=conditioning_mask.dtype,
                )
            )
            keyframe_coords.append(coords.expand(batch_size, -1, -1, -1))

        # Seeded with a zero-width block so the concat below is unconditional whether or not anything is appended.
        appended_coords = [torch.zeros((batch_size, 3, 0, 2), device=device, dtype=torch.float32)]
        if keyframe_tokens:
            keyframe_tokens = torch.cat(keyframe_tokens, dim=1)
            latents = torch.cat([latents, keyframe_tokens], dim=1)
            clean_latents = torch.cat([clean_latents, keyframe_tokens], dim=1)
            conditioning_mask = torch.cat([conditioning_mask, torch.cat(keyframe_masks, dim=1)], dim=1)
            appended_coords.append(torch.cat(keyframe_coords, dim=2))

        # Reference (IC-LoRA) tokens, appended last so they sit at the very end of the sequence -- the video
        # self-attention mask is built off that placement. Same mechanism as the keyframes above: a per-token
        # `conditioning_mask` of the reference's strength, matching `VideoConditionByReferenceLatent` in the
        # reference implementation. Absent for IC-LoRAs that take no reference video (camera control, style, ...),
        # which `LTX2InContextPipeline` supports too -- `LTX2AutoReferenceEncoderStep` is then skipped.
        num_ref_tokens = 0
        if block_state.reference_latents is not None:
            reference_conditions = block_state.reference_conditions
            if isinstance(reference_conditions, LTX2ReferenceCondition):
                reference_conditions = [reference_conditions]
            reference_tokens = block_state.reference_latents.expand(batch_size, -1, -1)
            num_ref_tokens = reference_tokens.shape[1]
            # Per-reference token counts come from the encoder rather than an equal split of the total, so
            # references of differing lengths (a reference video shorter than `num_frames`) get the right strengths.
            reference_masks = [
                torch.full(
                    (batch_size, num_tokens, 1),
                    float(ref_cond.strength),
                    device=device,
                    dtype=conditioning_mask.dtype,
                )
                for ref_cond, num_tokens in zip(reference_conditions, block_state.reference_token_counts)
            ]
            latents = torch.cat([latents, reference_tokens], dim=1)
            clean_latents = torch.cat([clean_latents, reference_tokens], dim=1)
            conditioning_mask = torch.cat([conditioning_mask, torch.cat(reference_masks, dim=1)], dim=1)
            appended_coords.append(block_state.reference_coords.expand(batch_size, -1, -1, -1))

        # Mask semantics: 0 -> fully noised, 1 -> kept clean, in between -> noise level (1 - mask) * noise_scale.
        noise = randn_tensor(
            latents.shape, generator=block_state.generator, device=latents.device, dtype=latents.dtype
        )
        scaled_mask = (1.0 - conditioning_mask) * noise_scale
        block_state.latents = noise * scaled_mask + latents * (1 - scaled_mask)

        block_state.conditioning_mask = conditioning_mask
        block_state.clean_latents = clean_latents
        block_state.appended_coords = torch.cat(appended_coords, dim=2)
        block_state.base_token_count = base_token_count
        block_state.num_ref_tokens = num_ref_tokens
        block_state.noise_scale = noise_scale

        self.set_block_state(state, block_state)
        return components, state


class LTX2BuildVideoSelfAttentionMaskStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Builds the video self-attention mask over the `[base | keyframe | reference]` token sequence for "
            "in-context generation, mirroring `build_attention_mask` in the reference implementation. Each "
            "`LTX2ReferenceCondition` is its own attention group:\n"
            "  - base <-> base, base <-> keyframe, keyframe <-> keyframe: 1.0 (full attention)\n"
            "  - base <-> reference group: that group's slice of `reference_cross_mask`, broadcast symmetrically "
            "across the base-token axis\n"
            "  - reference group <-> itself: 1.0 (a group fully attends to itself)\n"
            "  - reference group <-> any other appended group (keyframes, other references): 0.0\n"
            "Note the cross blocks span only the *base* tokens: keyframe tokens are appended conditioning like the "
            "references are, and the two are masked off from each other. Emitted tagged `denoiser_input_fields`, so "
            "it reaches the transformer's `video_self_attention_mask` argument without any change to "
            "`LTX2LoopDenoiser`. Only run when the references carry a per-token strength (see "
            "`LTX2AutoBuildVideoSelfAttentionMaskStep`); attention is otherwise left unmasked."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam(
                "base_token_count",
                type_hint=int,
                required=True,
                description="Number of generated-video tokens, i.e. the sequence length before appended tokens.",
            ),
            InputParam(
                "num_ref_tokens",
                type_hint=int,
                required=True,
                description="Number of reference tokens, which sit at the very end of the sequence.",
            ),
            InputParam(
                "reference_cross_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].",
            ),
            InputParam(
                "reference_token_counts",
                type_hint=list,
                required=True,
                description="Per-reference token counts, used to split `reference_cross_mask` into attention groups.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "video_self_attention_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Multiplicative self-attention mask of shape [B, S, S] with values in [0, 1].",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size, total_tokens, _ = block_state.latents.shape
        # Cross-attention partners are the base tokens only. The prefix (base + keyframe tokens) attends fully to
        # itself, but keyframe tokens are appended conditioning just like the references, so the two groups are
        # masked off from each other.
        num_base_tokens = block_state.base_token_count
        num_prefix_tokens = total_tokens - block_state.num_ref_tokens
        cross = block_state.reference_cross_mask.to(device=device, dtype=torch.float32)

        # Start from zeros so the keyframe<->reference and reference<->reference blocks stay masked without explicit
        # assignment. Each guidance pass is its own single-batch forward, so this is built at the generation batch
        # size -- the standard pipeline's expand to 2B for the CFG batch has no counterpart here.
        attn_mask = torch.zeros((batch_size, total_tokens, total_tokens), device=device, dtype=torch.float32)
        attn_mask[:, :num_prefix_tokens, :num_prefix_tokens] = 1.0

        # One attention group per reference condition, in the order the encoder emitted their tokens.
        offset = num_prefix_tokens
        for group_cross in torch.split(cross, block_state.reference_token_counts, dim=1):
            n = group_cross.shape[1]
            attn_mask[:, :num_base_tokens, offset : offset + n] = group_cross.unsqueeze(1)
            attn_mask[:, offset : offset + n, :num_base_tokens] = group_cross.unsqueeze(2)
            attn_mask[:, offset : offset + n, offset : offset + n] = 1.0
            offset += n

        block_state.video_self_attention_mask = attn_mask

        self.set_block_state(state, block_state)
        return components, state


class LTX2ConditionSetTimestepsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Sets the flow-match timesteps for the video `scheduler` and produces a deep-copied `audio_scheduler` "
            "(with the same schedule) so the audio latents are denoised on an independent scheduler state. Unlike "
            "`LTX2SetTimestepsStep`, the resolution-aware shift `mu` is read off the packed `latents` sequence "
            "length, which for condition workflows includes the appended keyframe tokens -- so this block runs "
            "*after* `LTX2ConditionPrepareLatentsStep`, matching `LTX2ConditionPipeline`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=30),
            InputParam.template("timesteps"),
            InputParam.template("sigmas"),
            InputParam.template("latents", required=True),
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

        mu = calculate_shift(
            block_state.latents.shape[1],
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


class LTX2ConditionPrepareAudioLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Prepares the packed audio noise latents and derives the audio latent frame count, for condition-based "
            "generation. Two deliberate differences from `LTX2PrepareAudioLatentsStep`:\n"
            "  1. the noise is sampled directly in the packed shape [B, L, C * M], matching "
            "`LTX2ConditionPipeline.prepare_audio_latents`. The text-to-video/image-to-video pipelines sample "
            "unpacked [B, C, L, M] and pack afterwards; both draw the same number of values from the generator but "
            "lay them out differently, so sampling the wrong way silently desynchronizes the audio noise (and, "
            "through the joint attention, the video too).\n"
            "  2. `noise_scale` is declared with a `None` default: a blockset keeps the first non-`None` default "
            "across its blocks, so the text-to-video default of 0.0 would otherwise shadow the condition workflow's "
            "`None -> sigmas[0] or 1.0` resolution. `LTX2ConditionPrepareLatentsStep` runs first and writes the "
            "resolved value back to state."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("audio_vae", AutoencoderKLLTX2Audio)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                default=None,
                description="Optional pre-encoded audio latents; random noise is used when not provided.",
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=None,
                description=(
                    "Initial noise level applied to any provided `audio_latents`. Resolved upstream by "
                    "`LTX2ConditionPrepareLatentsStep` (`sigmas[0]` when custom sigmas are supplied, else 1.0)."
                ),
            ),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("generator"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Packed noisy audio latents."),
            OutputParam(
                "audio_num_frames",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Number of audio latent frames.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_channels_latents = components.audio_vae.config.latent_channels
        latent_mel_bins = components.audio_vae.config.mel_bins // components.audio_vae_mel_compression_ratio

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
                audio_latents, components.audio_latents_mean, components.audio_latents_std
            )
            audio_latents = _create_noised_state(audio_latents, block_state.noise_scale, block_state.generator)
            block_state.audio_latents = audio_latents.to(device=device, dtype=torch.float32)
        else:
            # Sample directly in packed shape, following `LTX2ConditionPipeline.prepare_audio_latents` -- see the
            # block description for why the unpacked-then-pack order used by text-to-video is not interchangeable.
            packed_shape = (batch_size, audio_num_frames, num_channels_latents * latent_mel_bins)
            block_state.audio_latents = randn_tensor(
                packed_shape, generator=block_state.generator, device=device, dtype=torch.float32
            )

        block_state.audio_num_frames = audio_num_frames

        self.set_block_state(state, block_state)
        return components, state


class LTX2ConditionPrepareCoordsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Pre-computes the video and audio RoPE positional coordinates (constant across denoising steps) and "
            "appends the keyframe-condition coordinates onto the video coordinates, matching the order the "
            "keyframe tokens were appended to the latent sequence. CFG/STG batch duplication is left to the "
            "denoise step."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam("audio_num_frames", type_hint=int, required=True),
            InputParam(
                "appended_coords",
                type_hint=torch.Tensor,
                required=True,
                description="RoPE coordinates for the appended keyframe tokens, zero-width when there are none.",
            ),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "video_coords",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Video RoPE patch coordinates, with the keyframe-condition coordinates appended.",
            ),
            OutputParam(
                "audio_coords",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Audio RoPE patch coordinates.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

        video_coords = components.transformer.rope.prepare_video_coords(
            batch_size, latent_num_frames, latent_height, latent_width, device, fps=block_state.frame_rate
        )
        block_state.video_coords = torch.cat([video_coords, block_state.appended_coords.to(video_coords.dtype)], dim=2)
        block_state.audio_coords = components.transformer.audio_rope.prepare_audio_coords(
            batch_size, block_state.audio_num_frames, device
        )

        self.set_block_state(state, block_state)
        return components, state
