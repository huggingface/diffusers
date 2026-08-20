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

from typing import Union

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


class ABotWorldDecodeStep(ModularPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return "Step that de-normalizes the rollout's latents and VAE-decodes them into the output video."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("output_type", default="np"),
            InputParam(
                "video_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The rollout's accumulated latents `[B, C, T, h, w]`",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "videos",
                type_hint=Union[list[list[PIL.Image.Image]], list[torch.Tensor], list[np.ndarray]],
                description="The generated videos",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        if block_state.output_type == "latent":
            block_state.videos = block_state.video_latents
        else:
            latents = block_state.video_latents.to(vae.dtype)
            latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
                1, -1, 1, 1, 1
            )
            latents_std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
                1, -1, 1, 1, 1
            )
            latents = latents * latents_std + latents_mean
            video = vae.decode(latents, return_dict=False)[0]
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )

        self.set_block_state(state, block_state)
        return components, state
