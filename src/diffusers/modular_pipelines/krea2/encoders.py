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
from transformers import AutoTokenizer, Qwen3VLModel

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


class Krea2TextEncoderStep(ModularPipelineBlocks):
    """
    Text encoder step that produces Krea 2 prompt embeddings and optional unconditional embeddings.
    """

    model_name = "krea2"

    prompt_template_encode_prefix = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
    )
    prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 34
    prompt_template_encode_num_suffix_tokens = 5

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel),
            ComponentSpec("tokenizer", AutoTokenizer),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                "text_encoder_select_layers",
                default=(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
                description="Indices of Qwen3-VL hidden states used as Krea 2 text conditioning.",
            )
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=False),
            InputParam.template("negative_prompt"),
            InputParam.template("prompt_embeds", required=False),
            InputParam.template("prompt_embeds_mask", required=False),
            InputParam.template("negative_prompt_embeds", required=False),
            InputParam.template("negative_prompt_embeds_mask", required=False),
            InputParam("guidance_scale", type_hint=float, default=4.5),
            InputParam.template("max_sequence_length"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
        ]

    @staticmethod
    def check_inputs(
        prompt,
        negative_prompt,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        max_sequence_length,
    ):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please pass only one of them.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("`prompt_embeds_mask` is required when `prompt_embeds` is provided.")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("`negative_prompt_embeds_mask` is required when `negative_prompt_embeds` is provided.")
        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError(f"`max_sequence_length` must be a positive integer but is {max_sequence_length}")

    def get_text_hidden_states(
        self,
        components: Krea2ModularPipeline,
        prompt: str | list[str],
        max_sequence_length: int = 512,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or components._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in prompt]
        text_tokens = components.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - self.prompt_template_encode_num_suffix_tokens,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = components.tokenizer(
            [self.prompt_template_encode_suffix] * len(text),
            return_tensors="pt",
        ).to(device)

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = components.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack([outputs.hidden_states[i] for i in components.text_encoder_select_layers], dim=2)

        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        self.check_inputs(
            block_state.prompt,
            block_state.negative_prompt,
            block_state.prompt_embeds,
            block_state.prompt_embeds_mask,
            block_state.negative_prompt_embeds,
            block_state.negative_prompt_embeds_mask,
            block_state.max_sequence_length,
        )

        device = components._execution_device
        if block_state.prompt_embeds is None:
            block_state.prompt_embeds, block_state.prompt_embeds_mask = self.get_text_hidden_states(
                components,
                block_state.prompt,
                block_state.max_sequence_length,
                device,
            )

        if block_state.guidance_scale > 0 and block_state.negative_prompt_embeds is None:
            batch_size = (
                len(block_state.prompt) if isinstance(block_state.prompt, list) else block_state.prompt_embeds.shape[0]
            )
            negative_prompt = block_state.negative_prompt or ""
            negative_prompt = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
            if len(negative_prompt) != batch_size:
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, but the positive prompt batch size is {batch_size}."
                )
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = self.get_text_hidden_states(
                components,
                negative_prompt,
                block_state.max_sequence_length,
                device,
            )

        self.set_block_state(state, block_state)
        return components, state
