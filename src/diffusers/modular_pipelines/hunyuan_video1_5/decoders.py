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
from ...models import AutoencoderKLHunyuanVideo15
from ...pipelines.hunyuan_video1_5.image_processor import HunyuanVideo15ImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


class HunyuanVideo15VaeDecoderStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLHunyuanVideo15),
            ComponentSpec(
                "video_processor",
                HunyuanVideo15ImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into videos"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam.template("output_type", default="np"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("videos"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents.to(components.vae.dtype) / components.vae.config.scaling_factor
        video = components.vae.decode(latents, return_dict=False)[0]
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
