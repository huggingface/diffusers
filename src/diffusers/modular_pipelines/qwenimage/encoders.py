# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union
import torch
from ...utils import logging
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

from ...guiders import ClassifierFreeGuidance
from ...configuration_utils import FrozenDict

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam

from .modular_pipeline import QwenImageModularPipeline

logger = logging.get_logger(__name__)


def get_qwen_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]] = None,
    prompt_template_encode: str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    prompt_template_encode_start_idx: int = 34,
    tokenizer_max_length: int = 1024,
    device: Optional[torch.device] = None,
):

    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]
    txt_tokens = tokenizer(
        txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    encoder_hidden_states = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    split_hidden_states = _extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(device=device)

    return prompt_embeds, encoder_attention_mask

class QwenImageTextEncoderStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the image generation"
    
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen2_5_VLForConditionalGeneration, description="The text encoder to use"),
            ComponentSpec("tokenizer", Qwen2Tokenizer, description="The tokenizer to use"),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]
    
    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return [
            ConfigSpec(name="prompt_template_encode", default="<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"),
            ConfigSpec(name="prompt_template_encode_start_idx", default=34),
            ConfigSpec(name="tokenizer_max_length", default=1024),
        ]
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="prompt", required=True, type_hint=str, description="The prompt to encode"),
            InputParam(name="negative_prompt", type_hint=str, description="The negative prompt to encode"),
            InputParam(name="max_sequence_length", type_hint=int, description="The max sequence length to use", default=1024),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="prompt_embeds", kwargs_type="guider_input_fields",type_hint=torch.Tensor, description="The prompt embeddings"),
            OutputParam(name="prompt_embeds_mask", kwargs_type="guider_input_fields", type_hint=torch.Tensor, description="The encoder attention mask"),
            OutputParam(name="negative_prompt_embeds", kwargs_type="guider_input_fields", type_hint=torch.Tensor, description="The negative prompt embeddings"),
            OutputParam(name="negative_prompt_embeds_mask", kwargs_type="guider_input_fields", type_hint=torch.Tensor, description="The negative prompt embeddings mask"),
        ]

    @staticmethod
    def check_inputs(prompt, negative_prompt, max_sequence_length):

        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        
        if negative_prompt is not None and not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
        
        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        device = components._execution_device
        self.check_inputs(block_state.prompt, block_state.negative_prompt, block_state.max_sequence_length)

        block_state.prompt_embeds, block_state.prompt_embeds_mask = get_qwen_prompt_embeds(
            components.text_encoder,
            components.tokenizer,
            prompt=block_state.prompt,
            prompt_template_encode=components.config.prompt_template_encode,
            prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
            tokenizer_max_length=components.config.tokenizer_max_length,
            device=device,
        )

        block_state.prompt_embeds = block_state.prompt_embeds[:, :block_state.max_sequence_length]
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask[:, :block_state.max_sequence_length]

        if components.requires_unconditional_embeds:
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = get_qwen_prompt_embeds(
                components.text_encoder,
                components.tokenizer,
                prompt=block_state.negative_prompt,
                prompt_template_encode=components.config.prompt_template_encode,
                prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
                tokenizer_max_length=components.config.tokenizer_max_length,
                device=device,
            )
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds[:, :block_state.max_sequence_length]
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask[:, :block_state.max_sequence_length]

        self.set_block_state(state, block_state)
        return components, state