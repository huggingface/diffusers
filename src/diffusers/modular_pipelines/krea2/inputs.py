# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


class Krea2TextInputsStep(ModularPipelineBlocks):
    """
    Expands Krea 2 text embeddings to match `num_images_per_prompt` and records batch metadata.
    """

    model_name = "krea2"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("prompt_embeds"),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds"),
            InputParam.template("negative_prompt_embeds_mask"),
            InputParam("guidance_scale", type_hint=float, default=4.5),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds", note="batch-expanded"),
            OutputParam.template("prompt_embeds_mask", note="batch-expanded"),
            OutputParam.template("negative_prompt_embeds", note="batch-expanded"),
            OutputParam.template("negative_prompt_embeds_mask", note="batch-expanded"),
            OutputParam(name="batch_size", type_hint=int, description="The batch size of the prompt embeddings."),
            OutputParam(name="dtype", type_hint=torch.dtype, description="The dtype of the prompt embeddings."),
        ]

    @staticmethod
    def check_inputs(
        prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, guidance_scale
    ):
        if prompt_embeds.ndim != 4:
            raise ValueError(f"`prompt_embeds` must have 4 dimensions, but got {prompt_embeds.ndim}.")
        if prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`prompt_embeds_mask` must have the same batch size as `prompt_embeds`.")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("`negative_prompt_embeds_mask` is required when `negative_prompt_embeds` is provided.")
        if guidance_scale > 0 and negative_prompt_embeds is None:
            raise ValueError("Krea 2 classifier-free guidance requires `negative_prompt_embeds`.")
        if negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`negative_prompt_embeds` must have the same batch size as `prompt_embeds`.")
        if negative_prompt_embeds_mask is not None and negative_prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`negative_prompt_embeds_mask` must have the same batch size as `prompt_embeds`.")

    def __call__(self, components: Krea2ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        self.check_inputs(
            block_state.prompt_embeds,
            block_state.prompt_embeds_mask,
            block_state.negative_prompt_embeds,
            block_state.negative_prompt_embeds_mask,
            block_state.guidance_scale,
        )

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        batch_size, seq_len, num_text_layers, dim = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            batch_size * block_state.num_images_per_prompt, seq_len, num_text_layers, dim
        )
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.repeat(1, block_state.num_images_per_prompt)
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.view(
            batch_size * block_state.num_images_per_prompt, seq_len
        )

        if block_state.negative_prompt_embeds is not None:
            batch_size, seq_len, num_text_layers, dim = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(
                1, block_state.num_images_per_prompt, 1, 1
            )
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                batch_size * block_state.num_images_per_prompt, seq_len, num_text_layers, dim
            )
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.repeat(
                1, block_state.num_images_per_prompt
            )
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.view(
                batch_size * block_state.num_images_per_prompt, seq_len
            )

        self.set_block_state(state, block_state)
        return components, state
