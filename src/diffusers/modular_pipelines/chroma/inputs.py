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

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .modular_pipeline import ChromaModularPipeline


logger = logging.get_logger(__name__)


class ChromaTextInputStep(ModularPipelineBlocks):
    model_name = "chroma"

    @property
    def description(self) -> str:
        return (
            "Text input processing step that standardizes text embeddings and attention masks for the pipeline.\n"
            "This step:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Ensures all text embeddings and attention masks have consistent batch sizes (batch_size * num_images_per_prompt)"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("prompt_embeds"),
            InputParam.template("negative_prompt_embeds"),
            InputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask for the prompt embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask for the negative prompt embeddings. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask for the prompt embeddings, expanded to the final batch size.",
            ),
            OutputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask for the negative prompt embeddings, expanded to the final batch size.",
            ),
        ]

    def check_inputs(self, components, block_state):
        if block_state.prompt_embeds is not None and block_state.prompt_attention_mask is None:
            raise ValueError("Cannot provide `prompt_embeds` without also providing `prompt_attention_mask`")

        if block_state.negative_prompt_embeds is not None and block_state.negative_prompt_attention_mask is None:
            raise ValueError(
                "Cannot provide `negative_prompt_embeds` without also providing `negative_prompt_attention_mask`"
            )

        if block_state.negative_prompt_embeds is not None:
            if block_state.prompt_embeds.shape[0] != block_state.negative_prompt_embeds.shape[0]:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same batch size, but got:"
                    f" `prompt_embeds` {block_state.prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {block_state.negative_prompt_embeds.shape}."
                )

    @staticmethod
    def expand_text_tensor(text_tensor: torch.Tensor, num_images_per_prompt: int) -> torch.Tensor:
        # duplicate text embeddings/attention masks for each generation per prompt, using mps friendly method
        batch_size, seq_len = text_tensor.shape[:2]
        if text_tensor.ndim == 2:
            text_tensor = text_tensor.repeat(1, num_images_per_prompt)
            return text_tensor.view(batch_size * num_images_per_prompt, seq_len)
        text_tensor = text_tensor.repeat(1, num_images_per_prompt, 1)
        return text_tensor.view(batch_size * num_images_per_prompt, seq_len, -1)

    @torch.no_grad()
    def __call__(self, components: ChromaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        for field_name in [
            "prompt_embeds",
            "negative_prompt_embeds",
            "prompt_attention_mask",
            "negative_prompt_attention_mask",
        ]:
            text_tensor = getattr(block_state, field_name)
            if text_tensor is None:
                continue
            setattr(block_state, field_name, self.expand_text_tensor(text_tensor, block_state.num_images_per_prompt))

        self.set_block_state(state, block_state)

        return components, state
