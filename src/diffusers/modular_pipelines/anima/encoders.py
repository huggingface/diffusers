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
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import AnimaModularPipeline


class AnimaTextEncoderStep(ModularPipelineBlocks):
    model_name = "anima"

    @property
    def description(self) -> str:
        return "Text encoder step that encodes Anima prompts into Qwen states and T5 token ids."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3Model),
            ComponentSpec("tokenizer", Qwen2Tokenizer),
            ComponentSpec("t5_tokenizer", T5TokenizerFast),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam.template("max_sequence_length"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "qwen_prompt_embeds",
                type_hint=torch.Tensor,
                description="Qwen prompt embeddings to be consumed by the Anima text conditioner.",
            ),
            OutputParam(
                "qwen_attention_mask",
                type_hint=torch.Tensor,
                description="Qwen prompt attention mask to be consumed by the Anima text conditioner.",
            ),
            OutputParam(
                "t5_input_ids",
                type_hint=torch.Tensor,
                description="T5 prompt token ids to be consumed by the Anima text conditioner.",
            ),
            OutputParam(
                "t5_attention_mask",
                type_hint=torch.Tensor,
                description="T5 prompt attention mask to be consumed by the Anima text conditioner.",
            ),
            OutputParam(
                "negative_qwen_prompt_embeds",
                type_hint=torch.Tensor,
                description="Negative Qwen prompt embeddings to be consumed by the Anima text conditioner.",
            ),
            OutputParam(
                "negative_qwen_attention_mask",
                type_hint=torch.Tensor,
                description="Negative Qwen prompt attention mask to be consumed by the Anima text conditioner.",
            ),
            OutputParam(
                "negative_t5_input_ids",
                type_hint=torch.Tensor,
                description="Negative T5 prompt token ids to be consumed by the Anima text conditioner.",
            ),
            OutputParam(
                "negative_t5_attention_mask",
                type_hint=torch.Tensor,
                description="Negative T5 prompt attention mask to be consumed by the Anima text conditioner.",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")
        if block_state.max_sequence_length is not None and block_state.max_sequence_length > 4096:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 4096 but is {block_state.max_sequence_length}"
            )

    @staticmethod
    def _get_qwen_prompt_embeds(
        components: AnimaModularPipeline,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = components.tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)
        if text_input_ids.shape[-1] == 0:
            text_input_ids = text_input_ids.new_zeros((text_input_ids.shape[0], 1))
            prompt_attention_mask = prompt_attention_mask.new_zeros((prompt_attention_mask.shape[0], 1))

        prompt_embeds = components.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = prompt_embeds * prompt_attention_mask.to(prompt_embeds).unsqueeze(-1)

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def _get_t5_prompt_ids(
        components: AnimaModularPipeline,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = components.t5_tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)

    @classmethod
    def encode_prompt(
        cls,
        components: AnimaModularPipeline,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        prepare_unconditional_embeds: bool = True,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor | None]:
        device = device or components._execution_device
        dtype = dtype or components.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = cls._get_qwen_prompt_embeds(
            components=components,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        t5_input_ids, t5_attention_mask = cls._get_t5_prompt_ids(
            components=components,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        negative_t5_input_ids = None
        negative_t5_attention_mask = None
        if prepare_unconditional_embeds:
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = cls._get_qwen_prompt_embeds(
                components=components,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            negative_t5_input_ids, negative_t5_attention_mask = cls._get_t5_prompt_ids(
                components=components,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        return {
            "qwen_prompt_embeds": prompt_embeds,
            "qwen_attention_mask": prompt_attention_mask,
            "t5_input_ids": t5_input_ids,
            "t5_attention_mask": t5_attention_mask,
            "negative_qwen_prompt_embeds": negative_prompt_embeds,
            "negative_qwen_attention_mask": negative_prompt_attention_mask,
            "negative_t5_input_ids": negative_t5_input_ids,
            "negative_t5_attention_mask": negative_t5_attention_mask,
        }

    @torch.no_grad()
    def __call__(self, components: AnimaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        prompt_outputs = self.encode_prompt(
            components=components,
            prompt=block_state.prompt,
            negative_prompt=block_state.negative_prompt,
            prepare_unconditional_embeds=components.guider.num_conditions > 1,
            max_sequence_length=block_state.max_sequence_length,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        for name, value in prompt_outputs.items():
            setattr(block_state, name, value)

        self.set_block_state(state, block_state)
        return components, state
