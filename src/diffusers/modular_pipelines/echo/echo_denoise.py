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

from __future__ import annotations

import torch

from ...models import LTX2VideoTransformer3DModel
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam


DEFAULT_ECHO_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)


class EchoLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "echo"

    @property
    def description(self) -> str:
        return "Builds Echo's per-token video/audio timesteps and prepends clean memory tokens for one DMD step."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam(
                "audio_latents", type_hint=torch.Tensor, required=True, description="Packed noisy target audio tokens."
            ),
            InputParam(
                "first_frame_token_count",
                type_hint=int,
                required=True,
                description="Number of clean first-frame tokens.",
            ),
            InputParam(
                "memory_video_tokens",
                type_hint=torch.Tensor,
                required=False,
                description="Packed clean image-memory tokens.",
            ),
            InputParam(
                "memory_video_coords",
                type_hint=torch.Tensor,
                required=False,
                description="RoPE coordinates for image-memory tokens.",
            ),
            InputParam(
                "memory_audio_tokens",
                type_hint=torch.Tensor,
                required=False,
                description="Packed clean audio-memory tokens.",
            ),
            InputParam(
                "memory_audio_coords",
                type_hint=torch.Tensor,
                required=False,
                description="RoPE coordinates for audio-memory tokens.",
            ),
            InputParam(
                "video_coords",
                type_hint=torch.Tensor,
                required=True,
                description="RoPE coordinates for target video tokens.",
            ),
            InputParam(
                "audio_coords",
                type_hint=torch.Tensor,
                required=True,
                description="RoPE coordinates for target audio tokens.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, sigma: float):
        batch_size, video_token_count = block_state.latents.shape[:2]
        audio_token_count = block_state.audio_latents.shape[1]
        scaled_sigma = float(sigma) * float(components.transformer.config.timestep_scale_multiplier)

        video_timestep = torch.full(
            (batch_size, video_token_count),
            scaled_sigma,
            device=block_state.latents.device,
            dtype=torch.float32,
        )
        if block_state.first_frame_token_count:
            video_timestep[:, : block_state.first_frame_token_count] = 0
        audio_timestep = torch.full(
            (batch_size, audio_token_count),
            scaled_sigma,
            device=block_state.audio_latents.device,
            dtype=torch.float32,
        )

        block_state.latent_model_input = block_state.latents
        block_state.model_video_timestep = video_timestep
        block_state.model_video_coords = block_state.video_coords
        if block_state.memory_video_tokens is not None:
            memory_timestep = torch.zeros(
                (batch_size, block_state.memory_video_tokens.shape[1]),
                device=video_timestep.device,
                dtype=video_timestep.dtype,
            )
            block_state.latent_model_input = torch.cat([block_state.memory_video_tokens, block_state.latents], dim=1)
            block_state.model_video_timestep = torch.cat([memory_timestep, video_timestep], dim=1)
            block_state.model_video_coords = torch.cat(
                [block_state.memory_video_coords, block_state.video_coords], dim=2
            )

        block_state.audio_latent_model_input = block_state.audio_latents
        block_state.model_audio_timestep = audio_timestep
        block_state.model_audio_coords = block_state.audio_coords
        if block_state.memory_audio_tokens is not None:
            memory_audio_timestep = torch.zeros(
                (batch_size, block_state.memory_audio_tokens.shape[1]),
                device=audio_timestep.device,
                dtype=audio_timestep.dtype,
            )
            block_state.audio_latent_model_input = torch.cat(
                [block_state.memory_audio_tokens, block_state.audio_latents], dim=1
            )
            block_state.model_audio_timestep = torch.cat([memory_audio_timestep, audio_timestep], dim=1)
            block_state.model_audio_coords = torch.cat(
                [block_state.memory_audio_coords, block_state.audio_coords], dim=2
            )

        block_state.video_timestep = video_timestep
        block_state.audio_timestep = audio_timestep
        return components, block_state


class EchoLoopDenoiser(ModularPipelineBlocks):
    model_name = "echo"

    @property
    def description(self) -> str:
        return (
            "Runs the Echo transformer once without CFG, using clean ref/memory token timesteps and cross-modal "
            "timestep exchange, then converts velocity predictions to x0."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "connector_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Positive video-branch text conditioning.",
            ),
            InputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Positive audio-branch text conditioning.",
            ),
            InputParam(
                "connector_attention_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Binary attention mask for text conditioning.",
            ),
            InputParam.template("latents", required=True),
            InputParam(
                "audio_latents", type_hint=torch.Tensor, required=True, description="Packed noisy target audio tokens."
            ),
            InputParam(
                "video_coords",
                type_hint=torch.Tensor,
                required=True,
                description="RoPE coordinates for target video tokens.",
            ),
            InputParam(
                "audio_coords",
                type_hint=torch.Tensor,
                required=True,
                description="RoPE coordinates for target audio tokens.",
            ),
            InputParam(
                "latent_num_frames", type_hint=int, required=True, description="Number of target video latent frames."
            ),
            InputParam("latent_height", type_hint=int, required=True, description="Target video latent height."),
            InputParam("latent_width", type_hint=int, required=True, description="Target video latent width."),
            InputParam(
                "audio_num_frames", type_hint=int, required=True, description="Number of target audio latent frames."
            ),
            InputParam(
                "memory_video_token_count",
                type_hint=int,
                required=True,
                description="Number of prepended image-memory tokens.",
            ),
            InputParam(
                "memory_audio_token_count",
                type_hint=int,
                required=True,
                description="Number of prepended audio-memory tokens.",
            ),
            InputParam("frame_rate", type_hint=float, default=25.0, description="Frame rate of the generated video."),
            InputParam.template("attention_kwargs"),
        ]

    @staticmethod
    def _expand_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] != 1:
            raise ValueError(f"Cannot expand text conditioning batch {value.shape[0]} to {batch_size}.")
        return value.repeat_interleave(batch_size, dim=0)

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, sigma: float):
        batch_size = block_state.latents.shape[0]
        transformer_dtype = components.transformer.dtype
        video_context = self._expand_batch(block_state.connector_prompt_embeds, batch_size).to(transformer_dtype)
        audio_context = self._expand_batch(block_state.connector_audio_prompt_embeds, batch_size).to(transformer_dtype)
        context_mask = self._expand_batch(block_state.connector_attention_mask, batch_size)

        velocity_video, velocity_audio = components.transformer(
            hidden_states=block_state.latent_model_input.to(transformer_dtype),
            audio_hidden_states=block_state.audio_latent_model_input.to(transformer_dtype),
            encoder_hidden_states=video_context,
            audio_encoder_hidden_states=audio_context,
            timestep=block_state.model_video_timestep,
            audio_timestep=block_state.model_audio_timestep,
            # Echo deliberately uses the first target token as its global video sigma. When a clean first frame is
            # present this is zero, while the audio branch keeps the current DMD sigma. `use_cross_timestep=True`
            # exchanges these values in the cross-modal modulation blocks, matching the released wrapper.
            sigma=block_state.video_timestep[:, 0],
            audio_sigma=block_state.audio_timestep[:, 0],
            encoder_attention_mask=context_mask,
            audio_encoder_attention_mask=context_mask,
            num_frames=block_state.latent_num_frames,
            height=block_state.latent_height,
            width=block_state.latent_width,
            fps=block_state.frame_rate,
            audio_num_frames=block_state.audio_num_frames,
            video_coords=block_state.model_video_coords,
            audio_coords=block_state.model_audio_coords,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=True,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
        )
        velocity_video = velocity_video[:, block_state.memory_video_token_count :]
        velocity_audio = velocity_audio[:, block_state.memory_audio_token_count :]
        timestep_scale = float(components.transformer.config.timestep_scale_multiplier)
        block_state.predicted_video_x0 = (
            block_state.latents.float()
            - velocity_video.float() * block_state.video_timestep.unsqueeze(-1) / timestep_scale
        ).to(transformer_dtype)
        block_state.predicted_audio_x0 = (
            block_state.audio_latents.float()
            - velocity_audio.float() * block_state.audio_timestep.unsqueeze(-1) / timestep_scale
        ).to(transformer_dtype)
        if block_state.first_frame_tokens is not None:
            block_state.predicted_video_x0[:, : block_state.first_frame_token_count] = block_state.first_frame_tokens
        return components, block_state


class EchoLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "echo"

    @property
    def description(self) -> str:
        return "Re-noises each predicted x0 with fresh Gaussian noise at the next Echo DMD sigma."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam(
                "audio_latents", type_hint=torch.Tensor, required=True, description="Packed target audio tokens."
            ),
            InputParam(
                "sigmas",
                type_hint=list | tuple,
                required=True,
                description="DMD sigma schedule including the terminal zero.",
            ),
            InputParam.template("generator"),
            InputParam(
                "first_frame_tokens",
                type_hint=torch.Tensor,
                required=False,
                description="Packed clean first-frame tokens.",
            ),
            InputParam(
                "first_frame_token_count",
                type_hint=int,
                required=True,
                description="Number of clean first-frame tokens.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, sigma: float):
        next_sigma = float(block_state.sigmas[i + 1])
        if next_sigma > 0:
            video_noise = randn_tensor(
                block_state.predicted_video_x0.shape,
                generator=block_state.generator,
                device=block_state.predicted_video_x0.device,
                dtype=block_state.predicted_video_x0.dtype,
            )
            audio_noise = randn_tensor(
                block_state.predicted_audio_x0.shape,
                generator=block_state.generator,
                device=block_state.predicted_audio_x0.device,
                dtype=block_state.predicted_audio_x0.dtype,
            )
            block_state.latents = block_state.predicted_video_x0 * (1.0 - next_sigma) + video_noise * next_sigma
            block_state.audio_latents = block_state.predicted_audio_x0 * (1.0 - next_sigma) + audio_noise * next_sigma
            if block_state.first_frame_tokens is not None:
                block_state.latents[:, : block_state.first_frame_token_count] = block_state.first_frame_tokens
        else:
            block_state.latents = block_state.predicted_video_x0
            block_state.audio_latents = block_state.predicted_audio_x0
        return components, block_state


# auto_docstring
class EchoDenoiseStep(LoopSequentialPipelineBlocks):
    """
    Iteratively predicts clean video/audio latents and re-noises them with fresh Gaussian noise according to Echo's DMD
    sigma schedule.

      Components:
          transformer (`LTX2VideoTransformer3DModel`)

      Inputs:
          sigmas (`list | tuple`):
              DMD sigma schedule, including the terminal zero.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          audio_latents (`Tensor`):
              Packed noisy target audio tokens.
          first_frame_token_count (`int`):
              Number of clean first-frame tokens.
          memory_video_tokens (`Tensor`, *optional*):
              Packed clean image-memory tokens.
          memory_video_coords (`Tensor`, *optional*):
              RoPE coordinates for image-memory tokens.
          memory_audio_tokens (`Tensor`, *optional*):
              Packed clean audio-memory tokens.
          memory_audio_coords (`Tensor`, *optional*):
              RoPE coordinates for audio-memory tokens.
          video_coords (`Tensor`):
              RoPE coordinates for target video tokens.
          audio_coords (`Tensor`):
              RoPE coordinates for target audio tokens.
          connector_prompt_embeds (`Tensor`):
              Positive video-branch text conditioning.
          connector_audio_prompt_embeds (`Tensor`):
              Positive audio-branch text conditioning.
          connector_attention_mask (`Tensor`):
              Binary attention mask for text conditioning.
          latent_num_frames (`int`):
              Number of target video latent frames.
          latent_height (`int`):
              Target video latent height.
          latent_width (`int`):
              Target video latent width.
          audio_num_frames (`int`):
              Number of target audio latent frames.
          memory_video_token_count (`int`):
              Number of prepended image-memory tokens.
          memory_audio_token_count (`int`):
              Number of prepended audio-memory tokens.
          frame_rate (`float`, *optional*, defaults to 25.0):
              Frame rate of the generated video.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          audio_latents (`Tensor`):
              Packed target audio tokens.
          sigmas (`list | tuple`):
              DMD sigma schedule including the terminal zero.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          first_frame_tokens (`Tensor`, *optional*):
              Packed clean first-frame tokens.
    """

    model_name = "echo"
    block_classes = [EchoLoopBeforeDenoiser, EchoLoopDenoiser, EchoLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Iteratively predicts clean video/audio latents and re-noises them with fresh Gaussian noise according "
            "to Echo's DMD sigma schedule."
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "sigmas",
                type_hint=list | tuple,
                default=DEFAULT_ECHO_SIGMAS,
                description="DMD sigma schedule, including the terminal zero.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        sigmas = [float(value) for value in block_state.sigmas]
        if len(sigmas) < 2 or sigmas[-1] != 0.0:
            raise ValueError("Echo `sigmas` must contain at least two values and end at 0.")
        if any(left < right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("Echo `sigmas` must be monotonically non-increasing.")
        block_state.sigmas = sigmas

        with self.progress_bar(total=len(sigmas) - 1) as progress_bar:
            for i, sigma in enumerate(sigmas[:-1]):
                components, block_state = self.loop_step(components, block_state, i=i, sigma=sigma)
                progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state
