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

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Indices into the Qwen3-VL `hidden_states` tuple (0 is the embedding output) whose states are stacked per token as the
# transformer's text conditioning. Must have `transformer.config.num_text_layers` entries.
KREA2_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Krea 2 wraps the prompt in this Qwen-Image chat template before encoding. The prompt is padded to a fixed length
# first and the assistant suffix is appended *after* the padding (matching how the model was sampled at training time);
# the first `_PROMPT_TEMPLATE_ENCODE_START_IDX` (system prefix) tokens are dropped from the encoder outputs.
_PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
_PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PROMPT_TEMPLATE_ENCODE_START_IDX = 34
_PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS = 5


# auto_docstring
class Krea2TextEncoderStep(ModularPipelineBlocks):
    """
    Text encoder step that tokenizes the prompt(s) with the Krea 2 chat template, runs the Qwen3-VL text encoder, and
    stacks a fixed set of decoder-layer hidden states per token as the transformer's text conditioning. The negative
    prompt is encoded the same way when the guider enables CFG.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`AutoTokenizer`): The tokenizer paired
          with the text encoder. guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The negative prompt(s) for CFG.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`):
              Per-prompt negative text features (only when guidance is enabled).
          negative_prompt_embeds_mask (`Tensor`):
              Per-prompt negative text mask (only when guidance is enabled).
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that tokenizes the prompt(s) with the Krea 2 chat template, runs the Qwen3-VL text "
            "encoder, and stacks a fixed set of decoder-layer hidden states per token as the transformer's text "
            "conditioning. The negative prompt is encoded the same way when the guider enables CFG."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel, description="The Qwen3-VL text encoder."),
            ComponentSpec("tokenizer", AutoTokenizer, description="The tokenizer paired with the text encoder."),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.5, "use_original_formulation": True}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam(name="negative_prompt", type_hint=str, description="The negative prompt(s) for CFG."),
            InputParam.template("max_sequence_length", default=512),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="prompt_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).",
            ),
            OutputParam(
                name="prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Per-prompt boolean text mask (B, text_seq_len).",
            ),
            OutputParam(
                name="negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt negative text features (only when guidance is enabled).",
            ),
            OutputParam(
                name="negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Per-prompt negative text mask (only when guidance is enabled).",
            ),
        ]

    def _encode_prompt(self, components, prompt, max_sequence_length, device):
        """Tokenize `prompt` into the fixed-length Krea 2 layout and tap the selected encoder hidden states.

        Mirrors `Krea2Pipeline.get_text_hidden_states`. Returns a `(hidden_states, attention_mask)` tuple of shapes
        `(batch_size, text_seq_len, num_text_layers, text_hidden_dim)` and `(batch_size, text_seq_len)` (bool).
        """
        tokenizer = components.tokenizer
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prefix_idx = _PROMPT_TEMPLATE_ENCODE_START_IDX
        text = [_PROMPT_TEMPLATE_ENCODE_PREFIX + e for e in prompt]
        text_tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - _PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = tokenizer([_PROMPT_TEMPLATE_ENCODE_SUFFIX] * len(text), return_tensors="pt").to(device)

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

        # Krea 2 pads in the middle of the template (`[prefix | prompt | PAD | suffix]`), so the suffix tokens sit
        # downstream of the padding. The text features must use positions that count only real tokens (padding does
        # not consume a position) to match how the model was trained; otherwise the suffix gets a shifted mRoPE phase.
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = components.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack([outputs.hidden_states[i] for i in KREA2_TEXT_ENCODER_SELECT_LAYERS], dim=2)

        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else list(block_state.prompt)

        block_state.prompt_embeds, block_state.prompt_embeds_mask = self._encode_prompt(
            components, prompts, block_state.max_sequence_length, device
        )

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompts)
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = self._encode_prompt(
                components, negative_prompt, block_state.max_sequence_length, device
            )

        self.set_block_state(state, block_state)
        return components, state
