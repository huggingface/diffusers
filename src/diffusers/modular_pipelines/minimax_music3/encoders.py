# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

import re
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

from ...models import MiniMaxMusic3RVQDepthDecoder
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxMusic3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# The prompt template and its token ids are part of the checkpoint contract: even whitespace-level changes to the
# assembled prompt change the generated audio.
_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END_TOKEN_ID = 151670
_AUDIO_CFG_TOKEN_ID = 151654
_AUDIO_CODE_OFFSET = 151675
_SEMANTIC_VOCAB_SIZE = 16384
_MAX_PROMPT_TOKENS = 5_000
_MAX_AUDIO_FRAMES = 9_000

# The autoregressive stage's sampling parameters are fixed by the reference inference recipe.
_AR_CFG_SCALE = 1.5
_AR_CFG_TOP_K = 50
_AR_SAMPLING_TOP_K = 50

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _clean_caption(caption: str) -> str:
    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    # Strip the markdown forms accepted by the model's input contract.
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def _normalize_lyrics(lyrics: str) -> str:
    # Keep only consecutive structural tags (e.g. "[verse]") at the start of a line; text on a tag line is dropped.
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def _sample_top_k(logits: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(_AR_SAMPLING_TOP_K, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # Sample on the generator's device so a CPU generator gives device-independent results (the diffusers convention).
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)


def _embed_audio_frame(components: MiniMaxMusic3ModularPipeline, frame_codes: torch.Tensor) -> torch.Tensor:
    # frame_codes: [2, num_codebooks]. Sum the semantic-code embedding with the residual-code embeddings.
    embed_tokens = components.language_model.model.embed_tokens
    embeds = embed_tokens(frame_codes[:, :1] + _AUDIO_CODE_OFFSET)
    offsets = (
        torch.arange(components.num_codebooks - 1, device=frame_codes.device) * components.audio_vocab_size
    ).unsqueeze(0)
    extra = components.rvq_depth_decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
    embeds = embeds + extra.to(embeds.dtype)
    return embeds * components.num_codebooks**-0.5


def _generate_depth_codes(
    components: MiniMaxMusic3ModularPipeline,
    last_hidden: torch.Tensor,
    semantic_code: torch.Tensor,
    generator: Optional[torch.Generator],
):
    # Autoregressively sample the residual codes c1..c7 for one frame and collect their hidden states.
    sequence = [components.rvq_depth_decoder.projection(last_hidden).unsqueeze(1)]
    code_embed = components.language_model.model.embed_tokens(semantic_code + _AUDIO_CODE_OFFSET)
    sequence.append(components.rvq_depth_decoder.projection(code_embed).unsqueeze(1))
    codes = [semantic_code]
    hidden_parts = []
    for index in range(1, components.num_codebooks):
        hidden = components.rvq_depth_decoder(torch.cat(sequence, dim=1))[:, -1]
        hidden_parts.append(hidden[:1])
        logits = components.rvq_depth_decoder.audio_heads[index - 1](hidden)
        conditional, unconditional = logits[:1].float(), logits[1:2].float()
        logits = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
        # The sampled code is repeated so the language-model feedback keeps the [conditional, unconditional] rows.
        code = _sample_top_k(logits, generator).repeat(2)
        codes.append(code)
        if index < components.num_codebooks - 1:
            embed = components.rvq_depth_decoder.audio_embeddings(code + (index - 1) * components.audio_vocab_size)
            sequence.append(components.rvq_depth_decoder.projection(embed).unsqueeze(1))
    return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)


class MiniMaxMusic3TextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-music3"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that assembles the checkpoint's special-token prompt from the music description and "
            "the lyrics and tokenizes it into the conditional/unconditional token id pair."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("tokenizer", Qwen2Tokenizer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompt",
                required=True,
                type_hint=str,
                description="The music description (genre, mood, vocals, instrumentation, arrangement).",
            ),
            InputParam(
                "lyrics",
                required=True,
                type_hint=str,
                description=(
                    "The lyrics to sing. Structure tags such as `[verse]` or `[chorus]` must each be on their own "
                    "line; text on the same line as a leading tag is dropped by the checkpoint's input contract."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "text_ids",
                type_hint=torch.Tensor,
                description=(
                    "Token ids of shape `[2, sequence_length]` holding the conditional prompt and its classifier-free "
                    "counterpart (every token except the first and the two trailing structure tokens replaced by the "
                    "audio-CFG token)."
                ),
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.prompt, str) or not block_state.prompt.strip():
            raise ValueError(
                f"`prompt` (the music description) must be a non-empty string, got {block_state.prompt!r}"
            )
        if not isinstance(block_state.lyrics, str) or not block_state.lyrics.strip():
            raise ValueError(f"`lyrics` must be a non-empty string, got {block_state.lyrics!r}")

    @torch.no_grad()
    def __call__(self, components: MiniMaxMusic3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        text = (
            f"{_IM_START}{_CAPTION_START}{_clean_caption(block_state.prompt)}{_CAPTION_END}"
            f"{_LYRICS_START}{_normalize_lyrics(block_state.lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
        )
        input_ids = components.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
            raise ValueError(
                f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {_MAX_PROMPT_TOKENS}"
            )
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
        block_state.text_ids = torch.cat((input_ids, unconditional_ids), dim=0).to(components._execution_device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxMusic3SemanticGenerationStep(ModularPipelineBlocks):
    model_name = "minimax-music3"

    @property
    def description(self) -> str:
        return (
            "Autoregressive generation step: frame by frame, the global language model samples a semantic code with "
            "classifier-free guidance and the depth decoder samples the residual codes; the concatenated per-frame "
            "hidden states condition the flow-matching stage."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("language_model", Qwen3ForCausalLM),
            ComponentSpec("rvq_depth_decoder", MiniMaxMusic3RVQDepthDecoder),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "text_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Tokenized conditional/unconditional prompt pair generated by the text encoder step.",
            ),
            InputParam(
                "audio_duration",
                default=60.0,
                type_hint=float,
                description=(
                    "Upper bound on the generated audio length in seconds. The language model may stop earlier. "
                    "Capped at 9000 frames (six minutes)."
                ),
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "frame_hiddens",
                type_hint=torch.Tensor,
                description=(
                    "Concatenated per-frame hidden states of shape `[1, frames, num_codebooks * hidden_size]` that "
                    "condition the flow-matching stage."
                ),
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.audio_duration <= 0:
            raise ValueError(f"`audio_duration` must be positive, got {block_state.audio_duration}")

    @torch.no_grad()
    def __call__(self, components: MiniMaxMusic3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        text_ids = block_state.text_ids
        max_frames = min(int(block_state.audio_duration * components.frame_rate), _MAX_AUDIO_FRAMES)
        if max_frames == 0:
            raise ValueError(
                f"`audio_duration` {block_state.audio_duration} is shorter than one audio frame "
                f"(1 / {components.frame_rate} s)"
            )
        generator = block_state.generator

        language_model = components.language_model
        text_embeds = language_model.model.embed_tokens(text_ids)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

        vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=text_ids.device)
        vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[_AUDIO_END_TOKEN_ID] = False

        frame_hiddens = []
        # The first decode step only advances the state past `<|audio_start|>` and is not an emitted frame.
        for frame_index in range(max_frames + 1):
            logits = language_model.lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
            # Restrict the guided distribution to the conditional branch's top candidates, then re-mask: guidance on
            # two `-inf` logits produces NaN on masked positions.
            threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = _sample_top_k(guided, generator)
            if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
                break

            semantic_code = sampled - _AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = _generate_depth_codes(
                components, last_hidden, semantic_code.repeat(2), generator
            )
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = _embed_audio_frame(components, frame_codes)
            output = language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately")
        block_state.frame_hiddens = torch.stack(frame_hiddens, dim=1)

        self.set_block_state(state, block_state)
        return components, state
