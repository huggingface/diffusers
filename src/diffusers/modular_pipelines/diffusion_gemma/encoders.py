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

from ...image_processor import PipelineImageInput
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import DiffusionGemmaModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiffusionGemmaTextEncoderStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that applies the chat template to a `prompt` or a raw `messages` conversation "
            "and tokenizes it into the prompt token IDs consumed by the encoder prefill"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("processor", ProcessorMixin),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt", type_hint=str, description="Prompt text, wrapped in a chat template and tokenized"),
            InputParam(
                "messages",
                type_hint=list,
                description="A raw chat conversation to encode instead of `prompt`, e.g. "
                '`[{"role": "user", "content": "Hello"}]` or a multi-turn / multimodal conversation.',
            ),
            InputParam(
                "image",
                type_hint=PipelineImageInput,
                description="Image(s) to pair with `prompt` for multimodal generation. For richer layouts, put the "
                "image content directly in `messages`.",
            ),
            InputParam(
                "add_generation_prompt",
                type_hint=bool,
                default=True,
                description="Whether to add the generation prompt when applying the chat template.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_ids",
                type_hint=torch.LongTensor,
                description="Tokenized prompt of shape `(batch_size, prompt_length)`.",
            ),
            OutputParam(
                "prompt_attention_mask",
                type_hint=torch.LongTensor,
                description="Attention mask for `prompt_ids`.",
            ),
            OutputParam(
                "multimodal_inputs",
                type_hint=dict,
                description="Image tensors the processor produced for the encoder prefill.",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is None and block_state.messages is None:
            raise ValueError("Provide either `prompt` or `messages`.")
        if block_state.prompt is not None and block_state.messages is not None:
            raise ValueError("Provide either `prompt` or `messages`, not both.")

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        def build_content(text, img):
            if img is None:
                return text
            return [{"type": "image", "image": img}, {"type": "text", "text": text}]

        messages = block_state.messages
        if messages is None:
            prompt, image = block_state.prompt, block_state.image
            if isinstance(prompt, list):
                images = image if isinstance(image, list) else [image] * len(prompt)
                messages = [[{"role": "user", "content": build_content(p, im)}] for p, im in zip(prompt, images)]
            else:
                messages = [{"role": "user", "content": build_content(prompt, image)}]

        encoded = components.processor.apply_chat_template(
            messages,
            add_generation_prompt=block_state.add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"]
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.long)
        multimodal_keys = ("pixel_values", "image_position_ids", "mm_token_type_ids")

        block_state.prompt_ids = ids
        block_state.prompt_attention_mask = mask.to(dtype=torch.long)
        block_state.multimodal_inputs = {k: encoded[k] for k in multimodal_keys if k in encoded}

        self.set_block_state(state, block_state)
        return components, state
