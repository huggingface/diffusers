# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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
from transformers import Qwen2Tokenizer, Qwen3VLModel
from transformers.masking_utils import create_causal_mask

from ...pipelines.ideogram4.prompt_enhancer import (
    PROMPT_UPSAMPLE_TEMPERATURE,
    Ideogram4PromptEnhancerHead,
    build_caption_logits_processor,
    build_prompt_enhancer,
    generate_captions,
)
from ...utils import is_outlines_available, logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Ideogram4ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Hidden states of these Qwen3-VL decoder layers are concatenated to form the per-token
# text conditioning consumed by the Ideogram4 transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


# auto_docstring
class Ideogram4PromptUpsampleStep(ModularPipelineBlocks):
    """
    Optional step that rewrites the prompt(s) into Ideogram4's native structured JSON caption (the format the model is
    trained on) when ``prompt_upsampling=True``. Requires the optional ``prompt_enhancer_head`` component, which is
    grafted onto the shared ``text_encoder`` body to make it generative; install ``outlines`` for schema-constrained
    captions.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`Qwen2Tokenizer`): The tokenizer paired
          with the text encoder. prompt_enhancer_head (`Ideogram4PromptEnhancerHead`): The LM head grafted onto the
          text encoder for upsampling.

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          prompt_upsampling (`bool`, *optional*, defaults to False):
              If True, rewrite the prompt into the native JSON caption before encoding.
          prompt_upsampling_temperature (`float`, *optional*, defaults to 1.0):
              Sampling temperature for prompt upsampling.
          height (`int`, *optional*):
              Together with width, sets the caption's target aspect ratio.
          width (`int`, *optional*):
              Together with height, sets the caption's target aspect ratio.
          generator (`Generator`, *optional*):
              Reused to make the upsampling reproducible.

      Outputs:
          prompt (`str`):
              The (possibly upsampled) prompt forwarded to the text encoder.
    """

    model_name = "ideogram4"

    def __init__(self):
        # Built lazily on first upsample: the head-less encoder body + `prompt_enhancer_head`, combined.
        self._prompt_enhancer = None
        # Outlines logits processor for schema-constrained captions; built lazily on first upsample.
        self._caption_logits_processor = None
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Optional step that rewrites the prompt(s) into Ideogram4's native structured JSON caption when "
            "`prompt_upsampling=True` (the format the model is trained on). Requires a generative `text_encoder` "
            "(a `Qwen3VLForConditionalGeneration`); install `outlines` for schema-constrained captions."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel, description="The Qwen3-VL text encoder."),
            ComponentSpec("tokenizer", Qwen2Tokenizer, description="The tokenizer paired with the text encoder."),
            ComponentSpec(
                "prompt_enhancer_head",
                Ideogram4PromptEnhancerHead,
                description="LM head grafted onto the text encoder for prompt upsampling.",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam(
                name="prompt_upsampling",
                type_hint=bool,
                default=False,
                description="If True, rewrite the prompt into Ideogram4's native JSON caption before encoding.",
            ),
            InputParam(
                name="prompt_upsampling_temperature",
                type_hint=float,
                default=PROMPT_UPSAMPLE_TEMPERATURE,
                description="Sampling temperature for prompt upsampling.",
            ),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="prompt",
                type_hint=list,
                description="The (possibly upsampled) prompt forwarded to the text encoder.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if block_state.prompt_upsampling:
            if components.prompt_enhancer_head is None:
                raise ValueError(
                    "Prompt upsampling requires the `prompt_enhancer_head` component, which is not loaded. Load an "
                    "`Ideogram4PromptEnhancerHead` and add it to the pipeline."
                )
            if self._prompt_enhancer is None:
                self._prompt_enhancer = build_prompt_enhancer(components.text_encoder, components.prompt_enhancer_head)
            if self._caption_logits_processor is None and is_outlines_available():
                self._caption_logits_processor = build_caption_logits_processor(
                    self._prompt_enhancer, components.tokenizer
                )
            if self._caption_logits_processor is None:
                logger.warning_once(
                    "`outlines` is not installed; prompt upsampling runs unconstrained and may not return "
                    "schema-valid JSON. Install with `pip install outlines` for structured captions."
                )
            height = block_state.height or components.default_height
            width = block_state.width or components.default_width
            block_state.prompt = generate_captions(
                self._prompt_enhancer,
                components.tokenizer,
                self._caption_logits_processor,
                block_state.prompt,
                height,
                width,
                temperature=block_state.prompt_upsampling_temperature,
                generator=block_state.generator,
                device=components._execution_device,
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Ideogram4TextEncoderStep(ModularPipelineBlocks):
    """
    Text encoder step that tokenizes the prompt(s) and runs the Qwen3-VL text encoder, returning the per-token text
    features (concatenated from a fixed set of activation layers). Only the text tokens are encoded; the packed image
    tokens are appended later (the encoder is causal with image after text, so they never affect the text features).

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`Qwen2Tokenizer`): The tokenizer paired
          with the text encoder.

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 2048):
              Maximum sequence length for prompt encoding.

      Outputs:
          text_features (`Tensor`):
              Per-prompt text features (B, max_sequence_length, llm_features_dim), padding zeroed.
          text_lengths (`list`):
              Per-prompt real text-token counts, used to lay out the packed sequence.
    """

    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that tokenizes the prompt(s) and runs the Qwen3-VL text encoder, returning the "
            "per-token text features (concatenated from a fixed set of activation layers). Only the text tokens are "
            "encoded; the packed image tokens are appended later (the encoder is causal with image after text, so "
            "they never affect the text features)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel, description="The Qwen3-VL text encoder."),
            ComponentSpec("tokenizer", Qwen2Tokenizer, description="The tokenizer paired with the text encoder."),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam.template("max_sequence_length", default=2048),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="text_features",
                type_hint=torch.Tensor,
                description="Per-prompt text features (B, max_sequence_length, llm_features_dim), padding zeroed.",
            ),
            OutputParam(
                name="text_lengths",
                type_hint=list,
                description="Per-prompt real text-token counts, used to lay out the packed sequence.",
            ),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.ideogram4.pipeline_ideogram4.Ideogram4Pipeline._get_text_encoder_hidden_states
    def _get_text_encoder_hidden_states(
        text_encoder,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_2d: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the text encoder's decoder layers, returning the hidden states tapped at each activation layer."""

        language_model = text_encoder.language_model

        inputs_embeds = language_model.embed_tokens(token_ids)

        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        tokenizer = components.tokenizer
        max_text_tokens = block_state.max_sequence_length

        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else list(block_state.prompt)
        batch_size = len(prompts)

        # Tokenize each chat-formatted prompt and left-pad to `max_sequence_length`.
        token_ids = torch.zeros(batch_size, max_text_tokens, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_text_tokens, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, max_text_tokens, dtype=torch.long)
        text_lengths = []
        for b, text_prompt in enumerate(prompts):
            messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            n = int(toks.shape[0])
            if n > max_text_tokens:
                raise ValueError(f"prompt has {n} tokens, exceeds max_sequence_length={max_text_tokens}")
            text_lengths.append(n)
            offset = max_text_tokens - n
            token_ids[b, offset:] = toks
            attention_mask[b, offset:] = 1
            text_position_ids[b, offset:] = torch.arange(n)

        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_position_ids = text_position_ids.to(device)

        # Run the text encoder, tapping the activation-layer hidden states, then concatenate them into per-token
        # text features (padding zeroed).
        selected = self._get_text_encoder_hidden_states(
            components.text_encoder, token_ids, attention_mask, text_position_ids
        )
        text_features = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(batch_size, max_text_tokens, -1)
        text_features = (text_features * attention_mask.to(text_features.dtype).unsqueeze(-1)).to(torch.float32)

        block_state.text_features = text_features
        block_state.text_lengths = text_lengths

        self.set_block_state(state, block_state)
        return components, state
