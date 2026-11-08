# Copyright 2026 Pruna AI, The MiniMax and HuggingFace Teams. All rights reserved.
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
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLPH3Image
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import PH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.modular_pipelines.minimax_h3.decoders.MiniMaxH3AfterDenoiseStep with \bMiniMaxH3->PH3Image, minimax-h3->p-h3-image
class PH3ImageAfterDenoiseStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

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
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
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
        audio_rows = audio_rows.reshape(components.audio_channels, block_state.num_audio_latents, audio_rows.shape[-1])
        block_state.audio_latents = audio_rows.permute(0, 2, 1).contiguous()

        self.set_block_state(state, block_state)
        return components, state


class PH3ImageDecodeStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return "Decodes the generated latents into images."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLPH3Image),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                # The VAE's ImageNet normalization is reverted by this block, not by the processor.
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
                description="The generated latents, of shape `(1, latent_channels, 1, latent_height, latent_width)`.",
            ),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = block_state.latents * latents_std + latents_mean

        image = components.vae.decode(latents, return_dict=False)[0][:, :, 0]
        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1)
        image = (image.float() * pixel_std + pixel_mean).clamp(0, 1)
        block_state.images = components.image_processor.postprocess(image, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
