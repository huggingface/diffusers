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

import torch

from ...configuration_utils import FrozenDict
from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...pipelines.ltx2.vocoder import LTX2Vocoder
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def _unpack_audio_latents(
    latents: torch.Tensor,
    latent_length: int,
    num_mel_bins: int,
    patch_size: int | None = None,
    patch_size_t: int | None = None,
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
    else:
        latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
    return latents


def _denormalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


class LTX2VideoDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Step that decodes the denoised video latents into video frames"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="Denoised video latents"),
            InputParam("output_type", default="np", type_hint=str, description="Output format: pil, np, pt, latent"),
            InputParam("decode_timestep", default=0.0, description="Timestep for VAE decode conditioning"),
            InputParam("decode_noise_scale", default=None, description="Noise scale for decode conditioning"),
            InputParam("generator", description="Random generator for reproducibility"),
            InputParam("latent_num_frames", required=True, type_hint=int),
            InputParam("latent_height", required=True, type_hint=int),
            InputParam("latent_width", required=True, type_hint=int),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("dtype", required=True, type_hint=torch.dtype),
            InputParam("transformer_spatial_patch_size", default=1, type_hint=int),
            InputParam("transformer_temporal_patch_size", default=1, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("videos", description="The decoded video frames"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents

        # Unpack latents from [B, S, D] -> [B, C, F, H, W]
        # Uses the transformer's patchify sizes (not the VAE's internal patch_size)
        latents = _unpack_latents(
            latents,
            block_state.latent_num_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.transformer_spatial_patch_size,
            block_state.transformer_temporal_patch_size,
        )
        # Denormalize
        latents = _denormalize_latents(
            latents, components.vae.latents_mean, components.vae.latents_std, components.vae.config.scaling_factor
        )

        if block_state.output_type == "latent":
            block_state.videos = latents
        else:
            latents = latents.to(block_state.dtype)
            device = latents.device

            if not components.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(
                    latents.shape, generator=block_state.generator, device=device, dtype=latents.dtype
                )
                decode_timestep = block_state.decode_timestep
                decode_noise_scale = block_state.decode_noise_scale
                batch_size = block_state.batch_size

                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            latents = latents.to(components.vae.dtype)
            video = components.vae.decode(latents, timestep, return_dict=False)[0]
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )

        self.set_block_state(state, block_state)
        return components, state


class LTX2AudioDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Step that decodes the denoised audio latents into audio waveforms"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
            ComponentSpec("vocoder", LTX2Vocoder),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("audio_latents", required=True, type_hint=torch.Tensor, description="Denoised audio latents"),
            InputParam("output_type", default="np", type_hint=str),
            InputParam("audio_num_frames", required=True, type_hint=int),
            InputParam("latent_mel_bins", required=True, type_hint=int),
            InputParam("dtype", required=True, type_hint=torch.dtype),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("audio", description="The decoded audio waveforms"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        audio_latents = block_state.audio_latents

        # Denormalize audio latents
        audio_latents = _denormalize_audio_latents(
            audio_latents, components.audio_vae.latents_mean, components.audio_vae.latents_std
        )
        # Unpack audio latents
        audio_latents = _unpack_audio_latents(
            audio_latents, block_state.audio_num_frames, num_mel_bins=block_state.latent_mel_bins
        )

        if block_state.output_type == "latent":
            block_state.audio = audio_latents
        else:
            audio_latents = audio_latents.to(components.audio_vae.dtype)
            generated_mel_spectrograms = components.audio_vae.decode(audio_latents, return_dict=False)[0]
            # Squeeze batch dim and cast to float32 to match reference's decode_audio output format
            block_state.audio = components.vocoder(generated_mel_spectrograms).squeeze(0).float()

        self.set_block_state(state, block_state)
        return components, state
