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
from ...image_processor import VaeImageProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import AnimaModularPipeline


class AnimaProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "anima"

    @property
    def description(self) -> str:
        return "Image Preprocess step for Anima."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8, "vae_latent_channels": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("image"), InputParam("height"), InputParam("width")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="processed_image")]

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):
        divisor = vae_scale_factor * 2
        if height is not None and height % divisor != 0:
            raise ValueError(f"Height must be divisible by {divisor} but is {height}")

        if width is not None and width % divisor != 0:
            raise ValueError(f"Width must be divisible by {divisor} but is {width}")

    @torch.no_grad()
    def __call__(self, components: AnimaModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        if block_state.image is None:
            raise ValueError("`image` cannot be None")

        image = block_state.image
        self.check_inputs(
            height=block_state.height, width=block_state.width, vae_scale_factor=components.vae_scale_factor
        )
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        block_state.processed_image = components.image_processor.preprocess(image=image, height=height, width=width)

        self.set_block_state(state, block_state)
        return components, state
