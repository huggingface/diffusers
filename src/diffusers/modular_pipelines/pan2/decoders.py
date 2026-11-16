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

import numpy as np
import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLWan
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import PAN2ModularPipeline, PAN2SafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PAN2VaeDecoderStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into videos."

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
            ComponentSpec("safety_checker", PAN2SafetyChecker),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [ConfigSpec(name="enable_safety_checker", default=True)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True, description="Denoised latents to decode."),
            InputParam.template("output_type", default="np"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        vae = components.vae
        latents = block_state.latents
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
        latents = (latents * latents_std + latents_mean).to(vae.dtype)

        videos = vae.decode(latents, return_dict=False)[0]
        block_state.videos = components.video_processor.postprocess_video(videos, output_type=block_state.output_type)

        if components.requires_safety_checker:
            components.check_safety_checker_loaded()
            # The checker takes the uint8 `[num_frames, height, width, 3]` frames of one video.
            if block_state.output_type == "pil":
                frames = [np.stack([np.array(frame) for frame in video]) for video in block_state.videos]
            elif block_state.output_type == "np":
                frames = [(video * 255).astype(np.uint8) for video in block_state.videos]
            else:
                frames = [
                    (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8) for video in block_state.videos
                ]
            for video_frames in frames:
                if components.safety_checker.check_video_safety(video_frames) is None:
                    raise ValueError("The PAN 2 guardrail detected unsafe content in the generated video.")

        self.set_block_state(state, block_state)
        return components, state
