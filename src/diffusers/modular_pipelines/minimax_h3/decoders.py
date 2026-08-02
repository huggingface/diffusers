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
from .packing import MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3AfterDenoiseStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Turns the denoised rows of the packed sequence back into latents: drops the conditioning rows the loop "
            "never wrote, then unpacks the video rows into `(1, latent_channels, num_latent_frames, latent_height, "
            "latent_width)` and the channel-major audio rows into the `(2, audio_latent_channels, num_audio_latents)` "
            "the mono audio VAE consumes. The decoders then take latents from any source, and popping them leaves "
            "latents in hand."
        )

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
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The denoised audio rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many leading video rows are conditioning rows and are dropped here.",
            ),
            InputParam(
                name="num_condition_audio_rows",
                type_hint=int,
                default=0,
                description="How many leading audio rows are reference rows and are dropped here.",
            ),
            InputParam(
                name="num_latent_frames", type_hint=int, required=True, description="Number of video latent frames."
            ),
            InputParam(name="latent_height", type_hint=int, required=True, description="Height of the video latents."),
            InputParam(name="latent_width", type_hint=int, required=True, description="Width of the video latents."),
            InputParam(
                name="num_audio_latents",
                type_hint=int,
                required=True,
                description="Number of audio latents per channel.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated video latents, of shape `(1, latent_channels, num_latent_frames, "
                "latent_height, latent_width)`.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The generated audio latents, one batch item per stereo channel.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        patch_t, patch_h, patch_w = components.patch_size
        channels = components.vae_latent_channels

        # The inverse of the patchify in the prepare-latents step: rows are frame-major then row-major.
        rows = block_state.latents[block_state.num_condition_video_rows :]
        rows = rows.reshape(
            -1,
            block_state.num_latent_frames // patch_t,
            block_state.latent_height // patch_h,
            block_state.latent_width // patch_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        )
        rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
        block_state.latents = rows.reshape(
            -1, channels, block_state.num_latent_frames, block_state.latent_height, block_state.latent_width
        ).contiguous()

        # Audio rows are channel-major, and the mono audio VAE takes the two stereo channels as two batch items.
        audio_rows = block_state.audio_latents[block_state.num_condition_audio_rows :]
        audio_rows = audio_rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, block_state.num_audio_latents, audio_rows.shape[-1])
        block_state.audio_latents = audio_rows.permute(0, 2, 1).contiguous()

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3VideoDecodeStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Denormalizes the generated video latents and decodes them into video. The spatial tiling of the video "
            "VAE covers the canvas exactly, so the decoded frames need no crop back, but the decode itself runs under "
            "float16 autocast even though the VAE weights are float32, and the VAE produces ImageNet-normalized RGB "
            "that is reverted here."
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
                description="The generated video latents.",
            ),
            InputParam.template("output_type", description="Output format: 'pil', 'np' or 'pt'."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos", description="The generated video.")]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = block_state.latents * latents_std + latents_mean

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            video = components.vae.decode(latents, return_dict=False)[0]
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3AudioDecodeStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Denormalizes the generated audio latents and decodes them into a stereo waveform. The audio VAE is mono "
            "and takes the two stereo channels as two batch items."
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
                description="The generated audio latents, one batch item per stereo channel.",
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

        audio_latents_mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_latents_std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        audio_latents = block_state.audio_latents * audio_latents_std + audio_latents_mean

        audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
        block_state.audio = audio.float().permute(1, 0, 2)
        block_state.sampling_rate = components.audio_sampling_rate

        self.set_block_state(state, block_state)
        return components, state
