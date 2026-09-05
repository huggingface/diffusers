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
from ..modular_pipeline import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
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


# Copied from diffusers.modular_pipelines.ltx2.decoders._unpack_latents
def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    # Packed video latents of shape [B, S, D] are unpacked into a video tensor of shape [B, C, F, H, W].
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


# Copied from diffusers.modular_pipelines.ltx2.decoders._denormalize_audio_latents
def _denormalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


# Copied from diffusers.modular_pipelines.ltx2.decoders._unpack_audio_latents
def _unpack_audio_latents(
    latents: torch.Tensor,
    latent_length: int,
    num_mel_bins: int,
    patch_size: int | None = None,
    patch_size_t: int | None = None,
) -> torch.Tensor:
    # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor [B, C, L, M].
    if patch_size is not None and patch_size_t is not None:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
    else:
        # Assume [B, S, D] = [B, L, C * M], i.e. a (mel) patch_size of M and a patch_size_t of 1.
        latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
    return latents


class EchoVaeDecoderStep(ModularPipelineBlocks):
    model_name = "echo"

    @property
    def description(self) -> str:
        return "Unpacks and decodes the denoised Echo video latents into videos or returns latents."

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
            InputParam.template("latents", required=True),
            InputParam.template("output_type", default="pil"),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=None, description="Number of generated video frames."),
            InputParam("decode_timestep", default=0.0, description="Timestep used to decode the final latents."),
            InputParam(
                "decode_noise_scale",
                default=None,
                description="Noise interpolation factor applied at the decode timestep.",
            ),
            InputParam.template("generator"),
            InputParam.template("batch_size"),
            InputParam.template("dtype", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        latents = _unpack_latents(
            block_state.latents,
            latent_num_frames,
            latent_height,
            latent_width,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        )

        if block_state.output_type == "latent":
            block_state.videos = _denormalize_latents(
                latents,
                components.latents_mean,
                components.latents_std,
                components.vae_scaling_factor,
            )
            self.set_block_state(state, block_state)
            return components, state

        latents = latents.to(block_state.dtype)
        if not vae.config.timestep_conditioning:
            timestep = None
        else:
            batch_size = block_state.batch_size
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
            InputParam("audio_latents", type_hint=torch.Tensor, required=True, description="Denoised audio latents."),
            InputParam(
                "audio_num_frames",
                type_hint=int,
                required=True,
                description="Number of audio latent frames used to unpack the audio latent sequence.",
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

        num_mel_bins = audio_vae.config.mel_bins
        latent_mel_bins = num_mel_bins // components.audio_vae_mel_compression_ratio
        audio_latents = _denormalize_audio_latents(
            block_state.audio_latents,
            components.audio_latents_mean,
            components.audio_latents_std,
        )
        audio_latents = _unpack_audio_latents(
            audio_latents,
            block_state.audio_num_frames,
            num_mel_bins=latent_mel_bins,
        )

        if block_state.output_type == "latent":
            block_state.audio = audio_latents
        else:
            audio_latents = audio_latents.to(audio_vae.dtype)
            generated_mel_spectrograms = audio_vae.decode(audio_latents, return_dict=False)[0]
            block_state.audio = components.vocoder(generated_mel_spectrograms.to(components.vocoder.dtype))

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class EchoDecoderStep(SequentialPipelineBlocks):
    """
    Decode Echo video and audio outputs with mixed-precision-safe audio vocoding.

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) audio_vae (`AutoencoderKLLTX2Audio`)
          vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts before per-prompt expansion.
          dtype (`dtype`):
              The dtype of the model inputs.
          audio_latents (`Tensor`):
              Denoised audio latents.
          audio_num_frames (`int`):
              Number of audio latent frames used to unpack the audio latent sequence.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "echo"
    block_classes = [EchoVaeDecoderStep, EchoAudioDecoderStep]
    block_names = ["video_decode", "audio_decode"]

    @property
    def description(self):
        return "Decode Echo video and audio outputs with mixed-precision-safe audio vocoding."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]
