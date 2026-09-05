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

import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, LTX2VideoDiffusionDecoderModel

# NOTE (modular.md gotcha #1): `LTX2Vocoder` / `LTX2VocoderWithBWE` currently live under
# `diffusers.pipelines.ltx2.vocoder`, and modular blocks must not import from `diffusers.pipelines.*`.
# They are already `ModelMixin` / `ConfigMixin` model classes, so the clean fix is to relocate them to
# `src/diffusers/models/` and re-export from `diffusers.models` before this lands. Imported from the
# pipelines path here only so the draft is runnable; switch to the models path once moved.
from ...pipelines.ltx2.vocoder import LTX2Vocoder
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


# The pack/unpack/denormalize helpers below mirror the static methods on
# `diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline`. They are redefined here (rather than imported)
# because modular blocks must not import from `diffusers.pipelines.*` (modular.md gotcha #1); this follows
# the same redefinition pattern as `modular_pipelines/ltx/decoders.py`.
def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Denormalize video latents across the channel dimension [B, C, F, H, W].
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    # Packed video latents of shape [B, S, D] are unpacked into a video tensor of shape [B, C, F, H, W].
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _denormalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    # Denormalizes audio latents of shape [B, C, L, M]. The statistics are stored per (channel, mel bin), flattened in
    # the order the packed `[B, L, C * M]` layout uses, so they broadcast as [1, C, 1, M] here.
    num_channels, num_mel_bins = latents.shape[1], latents.shape[3]
    latents_mean = latents_mean.view(1, num_channels, 1, num_mel_bins).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, num_channels, 1, num_mel_bins).to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


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


class LTX2UnpackLatentsStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Unpacks the denoised video and audio latents from the transformer's token layout back into the "
            "`[B, C, F, H, W]` / `[B, C, L, M]` form the VAE encoders emit (still normalized). Closes every core "
            "denoise group, so the decode and upsample blocks that follow take the same form the encoders produce "
            "and need no geometry inputs."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                required=True,
                description="Denoised video latents, packed and normalized, of shape [B, S, C].",
            ),
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Denoised audio latents, packed and normalized, of shape [B, L, C * M].",
            ),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                required=True,
                description="The number of frames in the generated video.",
            ),
            InputParam(
                "audio_num_frames",
                type_hint=int,
                required=True,
                description="Number of audio latent frames, used to unpack the audio latents.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Video latents of shape [B, C, F, H, W] (normalized, not packed).",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="Audio latents of shape [B, C, L, M] (normalized, not packed).",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

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
        block_state.latents = latents

        latent_mel_bins = components.audio_latent_mel_bins
        block_state.audio_latents = _unpack_audio_latents(
            block_state.audio_latents, block_state.audio_num_frames, num_mel_bins=latent_mel_bins
        )

        self.set_block_state(state, block_state)
        return components, state


class LTX2TrimConditionTokensStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Drops the appended keyframe-condition tokens from the denoised packed latents, leaving only the "
            "generated-video tokens. Runs ahead of `LTX2UnpackLatentsStep`, which needs the plain video token grid."
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
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Denoised latents for the generated video, with condition tokens removed.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.latents = block_state.latents[:, : block_state.base_token_count]
        self.set_block_state(state, block_state)
        return components, state


class LTX2DiffusionVaeDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Decodes the video latents with the LTX-2 diffusion decoder. Swap this in for `LTX2VaeDecoderStep` on "
            "checkpoints that ship the diffusion decoder, which from LTX-2.5 on is the native default. The decoder "
            "denoises rather than deterministically decoding, so it draws its own noise from `generator` and takes "
            "no decode timestep."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("diffusion_decoder", LTX2VideoDiffusionDecoderModel),
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
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                required=True,
                description="Video latents of shape [B, C, F, H, W] (normalized, not packed).",
            ),
            InputParam.template("output_type", default="pil"),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        decoder = components.diffusion_decoder

        # Denormalize in the loop's float32 before casting to the decoder dtype -- the same order as running the
        # pipeline with `output_type="latent"` and decoding with `LTX2VideoDiffusionDecodePipeline`.
        latents = _denormalize_latents(
            block_state.latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
        )
        latents = latents.to(decoder.dtype)
        # It samples the noise it denoises, so pass the generator to keep decoding reproducible.
        video = decoder.decode(latents, generator=block_state.generator, return_dict=False)[0]
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class LTX2VaeDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Decodes the video latents with the video VAE into the output video."

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
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                required=True,
                description="Video latents of shape [B, C, F, H, W] (normalized, not packed).",
            ),
            InputParam.template("output_type", default="pil"),
            InputParam(
                "decode_timestep", default=0.0, description="The timestep at which the VAE decodes the final latents."
            ),
            InputParam(
                "decode_noise_scale",
                default=None,
                description="Noise interpolation factor applied to the latents at the decode timestep.",
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

        # LTX-2 applies the optional decode-time noise on the *normalized* latents, then denormalizes
        # (the reverse of LTX-1's decoder order).
        latents = block_state.latents.to(block_state.dtype)
        if not vae.config.timestep_conditioning:
            timestep = None
        else:
            device = latents.device
            batch_size = block_state.batch_size
            decode_timestep = block_state.decode_timestep
            decode_noise_scale = block_state.decode_noise_scale

            noise = randn_tensor(latents.shape, generator=block_state.generator, device=device, dtype=latents.dtype)
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

        latents = _denormalize_latents(
            latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
        )
        latents = latents.to(vae.dtype)
        video = vae.decode(latents, timestep, return_dict=False)[0]
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class LTX2AudioDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Decodes the audio latents with the audio VAE into a mel spectrogram and vocodes it into a waveform."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        # The checkpoint may ship either `LTX2Vocoder` or `LTX2VocoderWithBWE`; the concrete class is resolved from
        # the vocoder subfolder's config at load time. `LTX2Vocoder` is declared here as the representative type.
        return [
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
            ComponentSpec("vocoder", LTX2Vocoder),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Audio latents of shape [B, C, L, M] (normalized, not packed).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        audio_vae = components.audio_vae

        audio_latents = _denormalize_audio_latents(
            block_state.audio_latents, components.audio_latents_mean, components.audio_latents_std
        )
        audio_latents = audio_latents.to(audio_vae.dtype)
        generated_mel_spectrograms = audio_vae.decode(audio_latents, return_dict=False)[0]
        block_state.audio = components.vocoder(generated_mel_spectrograms)

        self.set_block_state(state, block_state)
        return components, state
