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

import numpy as np
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLWan
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HeliosDecodeStep(ModularPipelineBlocks):
    """Decode all chunk latents with VAE, trim frames, and postprocess into final video output."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Decodes all chunk latents with the VAE, concatenates them, "
            "trims to the target frame count, and postprocesses into the final video output."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latent_chunks", required=True, type_hint=list, description="List of per-chunk denoised latent tensors"
            ),
            InputParam("num_frames", required=True, type_hint=int, description="The target number of output frames"),
            InputParam.template("output_type", default="np"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "videos",
                type_hint=list[list[PIL.Image.Image]] | list[torch.Tensor] | list[np.ndarray],
                description="The generated videos, can be a PIL.Image.Image, torch.Tensor or a numpy array",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        vae = components.vae

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        history_video = None
        for chunk_latents in block_state.latent_chunks:
            current_latents = chunk_latents.to(vae.dtype) / latents_std + latents_mean
            current_video = vae.decode(current_latents, return_dict=False)[0]

            if history_video is None:
                history_video = current_video
            else:
                history_video = torch.cat([history_video, current_video], dim=2)

        # Trim to proper frame count
        generated_frames = history_video.size(2)
        generated_frames = (
            generated_frames - 1
        ) // components.vae_scale_factor_temporal * components.vae_scale_factor_temporal + 1
        history_video = history_video[:, :, :generated_frames]

        block_state.videos = components.video_processor.postprocess_video(
            history_video, output_type=block_state.output_type
        )

        self.set_block_state(state, block_state)

        return components, state
