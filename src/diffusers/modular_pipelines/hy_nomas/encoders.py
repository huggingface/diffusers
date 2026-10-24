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

import json

import torch
from transformers import AutoTokenizer, Qwen2_5_VLTextModel

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HyNomasModularPipeline


SYSTEM_PROMPT = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "        1. The main content and theme of the video. "
    "        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "        4. background environment, light, style and atmosphere. "
    "        5. camera angles, movements, and transitions used in the video."
)


def _tokenize(tokenizer, prompt, max_length=None):
    kwargs = {} if max_length is None else {"truncation": True, "max_length": max_length, "padding": "max_length"}
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )


def _action_variants(prompt):
    try:
        obj = json.loads(prompt)
    except (ValueError, TypeError):
        return []
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return []
    actions = segments[0].get("contained_actions")
    if not isinstance(actions, (dict, list)) or not actions:
        return []
    variants = []
    for count in range(len(actions)):
        segment = dict(segments[0])
        if count == 0:
            segment.pop("contained_actions")
        elif isinstance(actions, dict):
            segment["contained_actions"] = dict(list(actions.items())[:count])
        else:
            segment["contained_actions"] = actions[:count]
        variants.append(
            json.dumps(
                {key: [segment] if key == "segments" else value for key, value in obj.items()}, ensure_ascii=False
            )
        )
    return variants


# auto_docstring
class HyNomasTextEncoderStep(ModularPipelineBlocks):
    """
    Encode each segment with the Qwen video template and locate its action tokens.

      Components:
          text_encoder (`Qwen2_5_VLTextModel`) tokenizer (`AutoTokenizer`)

      Inputs:
          prompt (`str | list`):
              One prompt or an ordered list of prompts describing successive segments of a single video.
          max_sequence_length (`int`, *optional*, defaults to 1280):
              Number of text tokens retained per segment after removing the Qwen chat prefix; shorter sequences are
              padded.

      Outputs:
          prompts (`list`):
              Ordered prompts for successive segments of one video, rather than a batch of independent videos.
          prompt_embeds (`list`):
              Qwen hidden states for each segment, shaped (1, max_sequence_length, hidden_size), with the chat prefix
              removed.
          prompt_attention_masks (`list`):
              Per-segment text masks shaped (1, max_sequence_length), with one for valid tokens and zero for padding.
          action_text_offsets (`list`):
              Per-segment action-token boundaries in the cropped text sequence, or None for prompts without structured
              actions.
          cache_action_starts (`list`):
              Per-segment text boundary after which tokens are hidden from future segments; None keeps all valid text
              visible.
    """

    model_name = "hy-nomas"

    @property
    def description(self):
        return "Encode each segment with the Qwen video template and locate its action tokens."

    @property
    def expected_components(self):
        return [ComponentSpec("text_encoder", Qwen2_5_VLTextModel), ComponentSpec("tokenizer", AutoTokenizer)]

    @property
    def inputs(self):
        return [
            InputParam(
                "prompt",
                type_hint=str | list[str],
                description="One prompt or an ordered list of prompts describing successive segments of a single video.",
                required=True,
            ),
            InputParam(
                "max_sequence_length",
                type_hint=int,
                description="Number of text tokens retained per segment after removing the Qwen chat prefix; shorter sequences are padded.",
                default=1280,
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "prompts",
                type_hint=list[str],
                description="Ordered prompts for successive segments of one video, rather than a batch of independent videos.",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=list[torch.Tensor],
                description="Qwen hidden states for each segment, shaped (1, max_sequence_length, hidden_size), with the chat prefix removed.",
            ),
            OutputParam(
                "prompt_attention_masks",
                type_hint=list[torch.Tensor],
                description="Per-segment text masks shaped (1, max_sequence_length), with one for valid tokens and zero for padding.",
            ),
            OutputParam(
                "action_text_offsets",
                type_hint=list[list[int] | None],
                description="Per-segment action-token boundaries in the cropped text sequence, or None for prompts without structured actions.",
            ),
            OutputParam(
                "cache_action_starts",
                type_hint=list[int | None],
                description="Per-segment text boundary after which tokens are hidden from future segments; None keeps all valid text visible.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state) -> tuple[HyNomasModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        if not prompts or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompt must be a string or a nonempty list of segment strings.")
        if block_state.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive.")
        tokenizer = components.tokenizer
        marker = tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        ids = _tokenize(tokenizer, "")["input_ids"][0].tolist()
        crop = next(
            (i + len(marker) for i in range(len(ids) - len(marker) + 1) if ids[i : i + len(marker)] == marker), None
        )
        if crop is None:
            raise ValueError("The tokenizer chat template must contain the Qwen user marker.")
        length = block_state.max_sequence_length
        embeds, masks, offsets, starts = [], [], [], []
        for prompt in prompts:
            tokens = _tokenize(tokenizer, prompt, length + crop).to(components._execution_device)
            encoded = components.text_encoder(**tokens, output_hidden_states=True, use_cache=False)
            embeds.append(encoded.hidden_states[-3][:, crop:])
            masks.append(tokens["attention_mask"][:, crop:])
            full_ids = tokens["input_ids"][0].tolist()
            boundaries = []
            for variant in _action_variants(prompt):
                variant_ids = _tokenize(tokenizer, variant, length + crop)["input_ids"][0].tolist()
                common = next(
                    (i for i, (a, b) in enumerate(zip(full_ids, variant_ids)) if a != b),
                    min(len(full_ids), len(variant_ids)),
                )
                boundaries.append(max(boundaries[-1] if boundaries else 0, min(length - 1, common - crop)))
            starts.append(boundaries[0] if boundaries else None)
            # The training mask's final boundary includes the uncropped valid length, capped at the context length.
            offsets.append(boundaries + [min(length, int(tokens["attention_mask"].sum()))] if boundaries else None)
        block_state.prompts = prompts
        block_state.prompt_embeds = embeds
        block_state.prompt_attention_masks = masks
        block_state.action_text_offsets = offsets
        block_state.cache_action_starts = starts
        self.set_block_state(state, block_state)
        return components, state
