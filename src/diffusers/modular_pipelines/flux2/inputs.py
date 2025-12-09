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

from typing import List

import torch

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .modular_pipeline import Flux2ModularPipeline


logger = logging.get_logger(__name__)


class Flux2TextInputStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return (
            "This step:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Ensures all text embeddings have consistent batch sizes (batch_size * num_images_per_prompt)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "prompt_embeds",
                required=True,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings from Mistral3. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (determined by `prompt_embeds`)",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Text embeddings used to guide the image generation",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
        )

        self.set_block_state(state, block_state)
        return components, state
