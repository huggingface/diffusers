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

from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...pipelines.ltx2.vocoder import LTX2Vocoder
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


# Copied from diffusers.modular_pipelines.ltx2.decoders._denormalize_latents
def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Denormalize video latents across the channel dimension [B, C, F, H, W].
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def _denormalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    # Mirror the encoder's channel/mel statistics on unpacked VAE latents.
    latents_mean = latents_mean.view(1, latents.shape[1], 1, latents.shape[3]).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, latents.shape[1], 1, latents.shape[3]).to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


class EchoVaeDecoderStep(ModularPipelineBlocks):
    model_name = "echo"

    @property
    def description(self) -> str:
        return "Denormalizes and decodes Echo video VAE latents into videos or returns denormalized latents."

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
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                required=True,
                description="Normalized video VAE latents of shape (B, C, F, H, W).",
            ),
            InputParam.template("output_type", default="pil"),
            InputParam("decode_timestep", default=0.0, description="Timestep used to decode the final latents."),
            InputParam(
                "decode_noise_scale",
                default=None,
                description="Noise interpolation factor applied at the decode timestep.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        latents = block_state.latents

        if block_state.output_type == "latent":
            block_state.videos = _denormalize_latents(
                latents,
                components.latents_mean,
                components.latents_std,
                components.vae_scaling_factor,
            )
            self.set_block_state(state, block_state)
            return components, state

        if not vae.config.timestep_conditioning:
            timestep = None
        else:
            batch_size = latents.shape[0]
            decode_timestep = block_state.decode_timestep
            decode_noise_scale = block_state.decode_noise_scale
            noise = randn_tensor(
                latents.shape,
                generator=block_state.generator,
                device=latents.device,
                dtype=latents.dtype,
            )
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size

            timestep = torch.tensor(decode_timestep, device=latents.device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor(
                decode_noise_scale,
                device=latents.device,
                dtype=latents.dtype,
            )[:, None, None, None, None]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents = _denormalize_latents(
            latents,
            components.latents_mean,
            components.latents_std,
            components.vae_scaling_factor,
        ).to(vae.dtype)
        video = vae.decode(latents, timestep, return_dict=False)[0]
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class EchoAudioDecoderStep(ModularPipelineBlocks):
    """Decode Echo audio latents while preserving per-component mixed precision."""

    model_name = "echo"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
            ComponentSpec("vocoder", LTX2Vocoder),
        ]

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Normalized audio VAE latents of shape (B, C, L, M).",
            ),
            InputParam.template("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform.")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        audio_vae = components.audio_vae

        audio_latents = _denormalize_audio_latents(
            block_state.audio_latents,
            components.audio_latents_mean,
            components.audio_latents_std,
        )

        if block_state.output_type == "latent":
            block_state.audio = audio_latents
        else:
            audio_latents = audio_latents.to(audio_vae.dtype)
            generated_mel_spectrograms = audio_vae.decode(audio_latents, return_dict=False)[0]
            block_state.audio = components.vocoder(generated_mel_spectrograms.to(components.vocoder.dtype))

        self.set_block_state(state, block_state)
        return components, state
