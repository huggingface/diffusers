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
from transformers import ProcessorMixin

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import DiffusionGemmaModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiffusionGemmaDecodeStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return (
            "Decode step that trims each generated sequence at its first EOS token and decodes the token IDs "
            "into text with the processor"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("processor", ProcessorMixin),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "sequences",
                required=True,
                type_hint=torch.LongTensor,
                description="The generated token IDs of shape `(batch_size, generated_length)`.",
            ),
            InputParam(
                "eos_token_id",
                type_hint=int,
                description="EOS token ID used to trim each sequence. Falls back to the processor's tokenizer.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "texts",
                type_hint=list,
                description="The decoded generated text, one string per prompt.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        eos_token_id = block_state.eos_token_id
        if eos_token_id is None:
            tokenizer = getattr(components.processor, "tokenizer", components.processor)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)

        sequences = block_state.sequences
        # Trim each row at its first EOS so post-EOS canvas tokens don't leak into the decoded text.
        decode_sequences = sequences
        if eos_token_id is not None:
            decode_sequences = [
                seq[: int((seq == eos_token_id).nonzero(as_tuple=True)[0][0]) + 1]
                if (seq == eos_token_id).any()
                else seq
                for seq in sequences
            ]

        block_state.texts = components.processor.batch_decode(decode_sequences, skip_special_tokens=True)

        self.set_block_state(state, block_state)
        return components, state
