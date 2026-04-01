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
from transformers import T5EncoderModel, T5TokenizerFast

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import LTXModularPipeline


logger = logging.get_logger(__name__)


class LTXTextEncoderStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings to guide the video generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", T5EncoderModel),
            ComponentSpec("tokenizer", T5TokenizerFast),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam("prompt_embeds", type_hint=torch.Tensor),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_attention_mask", type_hint=torch.Tensor),
            InputParam("max_sequence_length", default=128),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="text embeddings used to guide the video generation",
            ),
            OutputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="attention mask for text embeddings",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="negative text embeddings",
            ),
            OutputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="attention mask for negative text embeddings",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

    @staticmethod
    def _get_t5_prompt_embeds(
        components,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = components.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        prompt_embeds = components.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def encode_prompt(
        components,
        prompt: str,
        device: torch.device | None = None,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: str | None = None,
        max_sequence_length: int = 128,
    ):
        device = device or components._execution_device
        dtype = components.text_encoder.dtype

        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = LTXTextEncoderStep._get_t5_prompt_embeds(
            components=components,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

        if prepare_unconditional_embeds:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = LTXTextEncoderStep._get_t5_prompt_embeds(
                components=components,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device

        # Skip encoding if pre-computed embeddings are provided
        if getattr(block_state, "prompt_embeds", None) is not None:
            self.set_block_state(state, block_state)
            return components, state

        (
            block_state.prompt_embeds,
            block_state.prompt_attention_mask,
            block_state.negative_prompt_embeds,
            block_state.negative_prompt_attention_mask,
        ) = self.encode_prompt(
            components=components,
            prompt=block_state.prompt,
            device=block_state.device,
            prepare_unconditional_embeds=components.requires_unconditional_embeds,
            negative_prompt=block_state.negative_prompt,
            max_sequence_length=block_state.max_sequence_length,
        )

        self.set_block_state(state, block_state)
        return components, state
