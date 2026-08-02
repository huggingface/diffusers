# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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
from ...models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline
from .packing import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3VideoDecodeStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Unpacks the generated video rows back into latents, denormalizes them and decodes them into video. The "
            "spatial tiling of the video VAE covers the canvas exactly, so the decoded frames need no crop back, but "
            "the decode itself runs under float16 autocast even though the VAE weights are float32, and the VAE "
            "produces ImageNet-normalized RGB that is reverted here."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLMiniMaxH3),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                # The video VAE decodes into ImageNet-normalized RGB over a [0, 1] base range, which this block
                # reverts itself, so the processor must not denormalize a second time.
                config=FrozenDict({"vae_scale_factor": 16, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The denoised video rows of the packed sequence, conditioning rows first.",
            ),
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many leading video rows are conditioning rows and are dropped here.",
            ),
            InputParam(
                name="num_latent_frames", type_hint=int, required=True, description="Number of video latent frames."
            ),
            InputParam(name="latent_height", type_hint=int, required=True, description="Height of the video latents."),
            InputParam(name="latent_width", type_hint=int, required=True, description="Width of the video latents."),
            InputParam.template(
                "output_type", description="Output format: 'pil', 'np', 'pt' or 'latent' for the raw latents."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos", description="The generated video.")]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        latents = unpatchify_video_tokens(
            block_state.latents[block_state.num_condition_video_rows :],
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            components.vae_latent_channels,
            components.patch_size,
        )
        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = latents * latents_std + latents_mean

        if block_state.output_type == "latent":
            block_state.videos = latents
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                video = components.vae.decode(latents, return_dict=False)[0]
            pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3AudioDecodeStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Unpacks the generated audio rows back into latents, denormalizes them and decodes them into a stereo "
            "waveform. The audio VAE is mono and takes the two stereo channels as two batch items."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("audio_vae", AutoencoderKLMiniMaxH3Audio)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The denoised audio rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="num_condition_audio_rows",
                type_hint=int,
                default=0,
                description="How many leading audio rows are reference rows and are dropped here.",
            ),
            InputParam(
                name="num_audio_latents",
                type_hint=int,
                required=True,
                description="Number of audio latents per channel.",
            ),
            InputParam.template(
                "output_type", description="Output format: 'pil', 'np', 'pt' or 'latent' for the raw latents."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "audio",
                type_hint=torch.Tensor,
                description="The generated soundtrack, of shape `(1, 2, num_samples)`.",
            ),
            OutputParam(
                "sampling_rate",
                type_hint=int,
                description="Sample rate of the generated soundtrack in Hz.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        audio_latents = unpack_audio_tokens(
            block_state.audio_latents[block_state.num_condition_audio_rows :], block_state.num_audio_latents
        )
        audio_latents_mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_latents_std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        audio_latents = audio_latents * audio_latents_std + audio_latents_mean

        if block_state.output_type == "latent":
            block_state.audio = audio_latents
        else:
            audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
            block_state.audio = audio.float().permute(1, 0, 2)
        block_state.sampling_rate = components.audio_sampling_rate

        self.set_block_state(state, block_state)
        return components, state
