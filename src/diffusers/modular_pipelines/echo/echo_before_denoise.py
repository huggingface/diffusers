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

from ...models import AutoencoderKLLTX2Audio, LTX2VideoTransformer3DModel
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .echo_encoders import _pack_audio_latents, _pack_latents


class EchoPrepareLatentsStep(ModularPipelineBlocks):
    """Prepare the target video/audio sequences and their fixed Echo coordinates."""

    model_name = "echo"

    @property
    def description(self) -> str:
        return (
            "Creates packed video and audio noise, inserts the optional clean first frame, and prepares fixed target "
            "RoPE coordinates for Echo's stochastic DMD loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=241,
                description="Number of generated pixel frames; must be `1 + k * vae_temporal_compression_ratio`.",
            ),
            InputParam("frame_rate", type_hint=float, default=25.0, description="Frame rate of the generated video."),
            InputParam(
                "model_frame_rate",
                type_hint=float,
                default=24.0,
                description="Training-time frame rate used for video RoPE coordinates.",
            ),
            InputParam.template("latents"),
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                default=None,
                description="Optional packed initial audio noise latents.",
            ),
            InputParam.template("generator"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam("batch_size", type_hint=int, required=True),
            InputParam("first_frame_tokens", type_hint=torch.Tensor, required=False),
            InputParam("memory_video_tokens", type_hint=torch.Tensor, required=False),
            InputParam("memory_video_coords", type_hint=torch.Tensor, required=False),
            InputParam("memory_audio_tokens", type_hint=torch.Tensor, required=False),
            InputParam("memory_audio_coords", type_hint=torch.Tensor, required=False),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Packed initial noisy video latents."),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Packed initial noisy audio latents."),
            OutputParam(
                "video_coords", type_hint=torch.Tensor, description="RoPE coordinates for target video tokens."
            ),
            OutputParam(
                "audio_coords", type_hint=torch.Tensor, description="RoPE coordinates for target audio tokens."
            ),
            OutputParam("latent_num_frames", type_hint=int, description="Number of target video latent frames."),
            OutputParam("latent_height", type_hint=int, description="Target video latent height."),
            OutputParam("latent_width", type_hint=int, description="Target video latent width."),
            OutputParam("audio_num_frames", type_hint=int, description="Number of target audio latent frames."),
            OutputParam(
                "first_frame_token_count", type_hint=int, description="Number of clean first-frame video tokens."
            ),
            OutputParam(
                "memory_video_token_count", type_hint=int, description="Number of prepended image-memory tokens."
            ),
            OutputParam(
                "memory_audio_token_count", type_hint=int, description="Number of prepended audio-memory tokens."
            ),
            OutputParam("dtype", type_hint=torch.dtype, description="Transformer computation dtype."),
        ]

    @staticmethod
    def _expand_batch(value: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if value is None or value.shape[0] == batch_size:
            return value
        if value.shape[0] != 1:
            raise ValueError(f"Cannot expand a condition batch of {value.shape[0]} to {batch_size}.")
        return value.repeat_interleave(batch_size, dim=0)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype
        batch_size = block_state.batch_size * block_state.num_videos_per_prompt

        if block_state.num_frames < 1 or (block_state.num_frames - 1) % components.vae_temporal_compression_ratio:
            raise ValueError(
                "`num_frames` must be `1 + k * vae_temporal_compression_ratio` for Echo, got "
                f"{block_state.num_frames}."
            )
        if block_state.height % components.vae_spatial_compression_ratio or block_state.width % (
            components.vae_spatial_compression_ratio
        ):
            raise ValueError(
                "`height` and `width` must be divisible by the video VAE spatial compression ratio "
                f"({components.vae_spatial_compression_ratio})."
            )

        latent_num_frames = 1 + (block_state.num_frames - 1) // components.vae_temporal_compression_ratio
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        video_token_count = (
            (latent_num_frames // components.transformer_temporal_patch_size)
            * (latent_height // components.transformer_spatial_patch_size)
            * (latent_width // components.transformer_spatial_patch_size)
        )

        if block_state.latents is None:
            block_state.latents = randn_tensor(
                (batch_size, video_token_count, components.transformer.config.in_channels),
                generator=block_state.generator,
                device=device,
                dtype=dtype,
            )
        else:
            latents = block_state.latents.to(device=device, dtype=dtype)
            if latents.ndim == 5:
                latents = _pack_latents(
                    latents,
                    components.transformer_spatial_patch_size,
                    components.transformer_temporal_patch_size,
                )
            if latents.shape != (batch_size, video_token_count, components.transformer.config.in_channels):
                raise ValueError(
                    "Unexpected Echo video latent shape: expected "
                    f"{(batch_size, video_token_count, components.transformer.config.in_channels)}, got "
                    f"{tuple(latents.shape)}."
                )
            block_state.latents = latents

        duration = block_state.num_frames / block_state.frame_rate
        audio_frames_per_second = (
            components.audio_sampling_rate
            / components.audio_hop_length
            / components.audio_vae_temporal_compression_ratio
        )
        audio_num_frames = round(duration * audio_frames_per_second)
        if block_state.audio_latents is None:
            block_state.audio_latents = randn_tensor(
                (batch_size, audio_num_frames, components.transformer.config.audio_in_channels),
                generator=block_state.generator,
                device=device,
                dtype=dtype,
            )
        else:
            audio_latents = block_state.audio_latents.to(device=device, dtype=dtype)
            if audio_latents.ndim == 4:
                audio_latents = _pack_audio_latents(audio_latents)
            if audio_latents.shape != (
                batch_size,
                audio_num_frames,
                components.transformer.config.audio_in_channels,
            ):
                raise ValueError(
                    "Unexpected Echo audio latent shape: expected "
                    f"{(batch_size, audio_num_frames, components.transformer.config.audio_in_channels)}, got "
                    f"{tuple(audio_latents.shape)}."
                )
            block_state.audio_latents = audio_latents

        first_frame_tokens = self._expand_batch(block_state.first_frame_tokens, batch_size)
        first_frame_token_count = 0
        if first_frame_tokens is not None:
            first_frame_token_count = first_frame_tokens.shape[1]
            expected_first_frame_tokens = (latent_height // components.transformer_spatial_patch_size) * (
                latent_width // components.transformer_spatial_patch_size
            )
            if first_frame_token_count != expected_first_frame_tokens:
                raise ValueError(
                    "The encoded first frame has the wrong token count: expected "
                    f"{expected_first_frame_tokens}, got {first_frame_token_count}."
                )
            block_state.latents = block_state.latents.clone()
            block_state.latents[:, :first_frame_token_count] = first_frame_tokens

        block_state.first_frame_tokens = first_frame_tokens
        block_state.memory_video_tokens = self._expand_batch(block_state.memory_video_tokens, batch_size)
        block_state.memory_audio_tokens = self._expand_batch(block_state.memory_audio_tokens, batch_size)
        block_state.video_coords = components.transformer.rope.prepare_video_coords(
            batch_size,
            latent_num_frames,
            latent_height,
            latent_width,
            device,
            fps=block_state.model_frame_rate,
        )
        block_state.audio_coords = components.transformer.audio_rope.prepare_audio_coords(
            batch_size, audio_num_frames, device
        )
        block_state.memory_video_coords = self._expand_batch(block_state.memory_video_coords, batch_size)
        block_state.memory_audio_coords = self._expand_batch(block_state.memory_audio_coords, batch_size)
        block_state.latent_num_frames = latent_num_frames
        block_state.latent_height = latent_height
        block_state.latent_width = latent_width
        block_state.audio_num_frames = audio_num_frames
        block_state.first_frame_token_count = first_frame_token_count
        block_state.memory_video_token_count = (
            0 if block_state.memory_video_tokens is None else block_state.memory_video_tokens.shape[1]
        )
        block_state.memory_audio_token_count = (
            0 if block_state.memory_audio_tokens is None else block_state.memory_audio_tokens.shape[1]
        )
        block_state.dtype = dtype

        self.set_block_state(state, block_state)
        return components, state
