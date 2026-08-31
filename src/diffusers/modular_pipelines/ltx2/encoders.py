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

import math
from typing import Any

import numpy as np
import PIL.Image
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel

# NOTE (modular.md gotcha #1): `LTX2TextConnectors`, `LTX2DurationHead`, `LTX2VideoCondition`,
# `LTX2ReferenceCondition`, the prompt-enhancement config/helpers, and the system prompts live under
# `diffusers.pipelines.ltx2.*`, and modular blocks must not import from `diffusers.pipelines.*`.
# `LTX2TextConnectors` / `LTX2DurationHead` are `ModelMixin` / `ConfigMixin` model classes (relocate to
# `src/diffusers/models/`); the two condition dataclasses, the enhancement config, the response/image helpers, and
# the system prompts are plain data and utilities (relocate to a neutral shared module or copy into this package).
# Imported from the pipelines path here only so the draft is runnable.
from ...pipelines.ltx2.connectors import LTX2TextConnectors
from ...pipelines.ltx2.duration_head import LTX2DurationHead
from ...pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from ...pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
from ...pipelines.ltx2.prompt_enhancement import (
    _pad_inputs_for_attention_alignment,
    _prepare_enhance_image,
    clean_response,
)
from ...pipelines.ltx2.utils import (
    GEMMA4_PROMPT_ENHANCEMENT_CONFIG,
    LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
    LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
    apply_image_conditioning_crf,
    resolve_default_image_crf,
)
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


def _enhance_prompt(
    components,
    prompt: str | list[str],
    system_prompt: str,
    image: Any | None,
    max_new_tokens: int | None,
    seed: int,
    generator: torch.Generator | None,
    generation_kwargs: dict[str, Any] | None,
    device: torch.device,
) -> list[str]:
    # Mirrors `LTX2Pipeline.enhance_prompt` for the LTX-2.5 path only: a dedicated `prompt_enhancer`
    # (Gemma-4) with the greedy `GEMMA4_PROMPT_ENHANCEMENT_CONFIG` recipe. The LTX-2.0/2.3
    # `text_encoder`-as-enhancer fallback is intentionally dropped (LTX-2.5-only integration).
    config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG
    generation_kwargs = dict(generation_kwargs) if generation_kwargs is not None else dict(config.generation_kwargs)
    if max_new_tokens is None:
        max_new_tokens = config.max_new_tokens

    # Templates match ltx-core `LTXGemmaTextEncoder.enhance_t2v` / `enhance_i2v`.
    if image is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        enhance_image = None
    else:
        enhance_image = _prepare_enhance_image(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]

    template = components.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = components.processor(text=template, images=enhance_image, return_tensors="pt").to(device)
    pad_token_id = (
        components.processor.tokenizer.pad_token_id if components.processor.tokenizer.pad_token_id is not None else 0
    )
    model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)
    components.prompt_enhancer.to(device)

    # `transformers.GenerationMixin.generate` does not support a `torch.Generator`, so seed manually for
    # reproducibility. (Inert for LTX-2.5's greedy decoding, but honored if the user passes sampling kwargs.)
    if generator is not None:
        seed = generator.initial_seed() if not isinstance(generator, list) else generator[0].initial_seed()
    torch.manual_seed(seed)
    generated_sequences = components.prompt_enhancer.generate(
        **model_inputs, max_new_tokens=max_new_tokens, **generation_kwargs
    )

    generated_ids = [seq[len(model_inputs.input_ids[i]) :] for i, seq in enumerate(generated_sequences)]
    enhanced_prompt = components.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [clean_response(text) for text in enhanced_prompt]


def _get_gemma_prompt_embeds(
    components,
    prompt: str | list[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    # Mirrors `LTX2Pipeline._get_gemma_prompt_embeds`, minus the `num_videos_per_prompt` expansion. The whole text
    # stage stays at one row per prompt; `LTX2TextInputStep` expands the connector outputs to the effective batch at
    # the head of the denoise stage, so this stage's outputs are reusable across `num_videos_per_prompt` values.
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Gemma expects left padding for chat-style prompts.
    components.tokenizer.padding_side = "left"
    if components.tokenizer.pad_token is None:
        components.tokenizer.pad_token = components.tokenizer.eos_token

    prompt = [p.strip() for p in prompt]
    text_inputs = components.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_attention_mask = text_inputs.attention_mask.to(device)

    text_encoder_outputs = components.text_encoder(
        input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
    )
    # Stack all hidden-state layers into the packed per-layer representation the connectors expect.
    text_encoder_hidden_states = torch.stack(text_encoder_outputs.hidden_states, dim=-1)
    prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(dtype=dtype)  # pack to 3D

    return prompt_embeds, prompt_attention_mask


class LTX2PromptEnhancerStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Text-to-video prompt enhancer step. Rewrites `prompt` into a detailed caption using the dedicated "
            "LTX-2.5 `prompt_enhancer` (a Gemma conditional-generation model) and a system prompt."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("prompt_enhancer", PreTrainedModel),
            ComponentSpec("processor", ProcessorMixin),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam(
                "enable_prompt_enhancement",
                type_hint=bool,
                default=False,
                description=(
                    "Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines."
                ),
            ),
            InputParam(
                "system_prompt",
                type_hint=str,
                default=None,
                description="System prompt for enhancement. Defaults to `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`.",
            ),
            InputParam(
                "prompt_max_new_tokens",
                type_hint=int,
                default=None,
                description=(
                    "Maximum number of new tokens to generate during prompt enhancement. Defaults to 600, the "
                    "LTX-2.5 Gemma-4 enhancer's budget."
                ),
            ),
            InputParam(
                "prompt_enhancement_kwargs",
                type_hint=dict,
                default=None,
                description="Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.",
            ),
            InputParam(
                "prompt_enhancement_seed",
                type_hint=int,
                default=10,
                description="Random seed for prompt enhancement (inert under LTX-2.5's greedy decoding).",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt", type_hint=list, description="The prompt(s) after prompt-enhancer rewriting."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if not block_state.enable_prompt_enhancement:
            self.set_block_state(state, block_state)  # leave prompt unchanged
            return components, state
        if getattr(components, "prompt_enhancer", None) is None:
            raise ValueError(
                "`enable_prompt_enhancement=True` but no `prompt_enhancer` component is loaded. Load a "
                "`prompt_enhancer` (and `processor`), or set `enable_prompt_enhancement=False`."
            )

        system_prompt = block_state.system_prompt or LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT

        block_state.prompt = _enhance_prompt(
            components=components,
            prompt=block_state.prompt,
            system_prompt=system_prompt,
            image=None,
            max_new_tokens=block_state.prompt_max_new_tokens,
            seed=block_state.prompt_enhancement_seed,
            generator=block_state.generator,
            generation_kwargs=block_state.prompt_enhancement_kwargs,
            device=components._execution_device,
        )

        self.set_block_state(state, block_state)
        return components, state


class LTX2ImageToVideoPromptEnhancerStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Image-to-video prompt enhancer step. Rewrites `prompt` into a detailed caption grounded in the reference "
            "`image`, using the dedicated LTX-2.5 `prompt_enhancer` and a system prompt."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("prompt_enhancer", PreTrainedModel),
            ComponentSpec("processor", ProcessorMixin),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam.template("image", required=True),
            InputParam(
                "enable_prompt_enhancement",
                type_hint=bool,
                default=False,
                description=(
                    "Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines."
                ),
            ),
            InputParam(
                "system_prompt",
                type_hint=str,
                default=None,
                description="System prompt for enhancement. Defaults to `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`.",
            ),
            InputParam(
                "prompt_max_new_tokens",
                type_hint=int,
                default=None,
                description=(
                    "Maximum number of new tokens to generate during prompt enhancement. Defaults to 600, the "
                    "LTX-2.5 Gemma-4 enhancer's budget."
                ),
            ),
            InputParam(
                "prompt_enhancement_kwargs",
                type_hint=dict,
                default=None,
                description="Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.",
            ),
            InputParam(
                "prompt_enhancement_seed",
                type_hint=int,
                default=10,
                description="Random seed for prompt enhancement (inert under LTX-2.5's greedy decoding).",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt", type_hint=list, description="The prompt(s) after prompt-enhancer rewriting."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if not block_state.enable_prompt_enhancement:
            self.set_block_state(state, block_state)  # leave prompt unchanged
            return components, state
        if getattr(components, "prompt_enhancer", None) is None:
            raise ValueError(
                "`enable_prompt_enhancement=True` but no `prompt_enhancer` component is loaded. Load a "
                "`prompt_enhancer` (and `processor`), or set `enable_prompt_enhancement=False`."
            )

        system_prompt = block_state.system_prompt or LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT

        block_state.prompt = _enhance_prompt(
            components=components,
            prompt=block_state.prompt,
            system_prompt=system_prompt,
            image=block_state.image,
            max_new_tokens=block_state.prompt_max_new_tokens,
            seed=block_state.prompt_enhancement_seed,
            generator=block_state.generator,
            generation_kwargs=block_state.prompt_enhancement_kwargs,
            device=components._execution_device,
        )

        self.set_block_state(state, block_state)
        return components, state


class LTX2ConditionPromptEnhancerStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Condition prompt enhancer step. Rewrites `prompt` into a detailed caption using the dedicated LTX-2.5 "
            "`prompt_enhancer`, grounded in the first `PIL.Image.Image` frame found in `conditions` when there is "
            "one, and text-only otherwise. Reference conditions are never used for enhancement."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("prompt_enhancer", PreTrainedModel),
            ComponentSpec("processor", ProcessorMixin),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam(
                "conditions",
                type_hint=list,
                default=None,
                description=(
                    "`LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices "
                    "of the generated video."
                ),
            ),
            InputParam(
                "enable_prompt_enhancement",
                type_hint=bool,
                default=False,
                description=(
                    "Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines."
                ),
            ),
            InputParam(
                "system_prompt",
                type_hint=str,
                default=None,
                description=(
                    "System prompt for enhancement. Defaults to `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` when a "
                    "`PIL.Image.Image` condition frame is available, else `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`."
                ),
            ),
            InputParam(
                "prompt_max_new_tokens",
                type_hint=int,
                default=None,
                description=(
                    "Maximum number of new tokens to generate during prompt enhancement. Defaults to 600, the "
                    "LTX-2.5 Gemma-4 enhancer's budget."
                ),
            ),
            InputParam(
                "prompt_enhancement_kwargs",
                type_hint=dict,
                default=None,
                description="Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.",
            ),
            InputParam(
                "prompt_enhancement_seed",
                type_hint=int,
                default=10,
                description="Random seed for prompt enhancement (inert under LTX-2.5's greedy decoding).",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt", type_hint=list, description="The prompt(s) after prompt-enhancer rewriting."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if not block_state.enable_prompt_enhancement:
            self.set_block_state(state, block_state)  # leave prompt unchanged
            return components, state
        if getattr(components, "prompt_enhancer", None) is None:
            raise ValueError(
                "`enable_prompt_enhancement=True` but no `prompt_enhancer` component is loaded. Load a "
                "`prompt_enhancer` (and `processor`), or set `enable_prompt_enhancement=False`."
            )

        conditions = block_state.conditions
        if conditions is None:
            conditions = []
        elif isinstance(conditions, LTX2VideoCondition):
            conditions = [conditions]

        # First PIL frame across the conditions grounds the enhancement, matching `LTX2ConditionPipeline`.
        enhancement_image = None
        for condition in conditions:
            frames = condition.frames
            if isinstance(frames, PIL.Image.Image):
                enhancement_image = frames
                break
            if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], PIL.Image.Image):
                enhancement_image = frames[0]
                break

        system_prompt = block_state.system_prompt
        if system_prompt is None:
            system_prompt = (
                LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT if enhancement_image is not None else LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
            )

        block_state.prompt = _enhance_prompt(
            components=components,
            prompt=block_state.prompt,
            system_prompt=system_prompt,
            image=enhancement_image,
            max_new_tokens=block_state.prompt_max_new_tokens,
            seed=block_state.prompt_enhancement_seed,
            generator=block_state.generator,
            generation_kwargs=block_state.prompt_enhancement_kwargs,
            device=components._execution_device,
        )

        self.set_block_state(state, block_state)
        return components, state


class LTX2TextEncoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Text encoder step. Encodes `prompt` and `negative_prompt` into packed per-layer Gemma hidden states "
            "that the connectors adapt for the video and audio branches, and reports the prompt count (`batch_size`) "
            "and embedding `dtype`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        # No `guider`: LTX-2 applies CFG (+ STG + modality-isolation) manually in the denoise loop, so the encoder
        # always produces both conditional and unconditional embeddings and the denoiser decides what to use.
        return [
            ComponentSpec("text_encoder", PreTrainedModel),
            ComponentSpec("tokenizer", PreTrainedTokenizerBase),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam.template("max_sequence_length", default=1024),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Packed per-layer Gemma hidden states for the prompt.",
            ),
            OutputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Binary attention mask for `prompt_embeds`.",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Packed per-layer Gemma hidden states for the negative prompt.",
            ),
            OutputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Binary attention mask for `negative_prompt_embeds`.",
            ),
            OutputParam(
                "batch_size",
                type_hint=int,
                description="The number of prompts being denoised (before per-prompt expansion).",
            ),
            OutputParam("dtype", type_hint=torch.dtype, description="The dtype of the prompt embeddings."),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and not isinstance(block_state.prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        device = components._execution_device
        dtype = components.text_encoder.dtype
        max_sequence_length = block_state.max_sequence_length

        prompt = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        block_state.prompt_embeds, block_state.prompt_attention_mask = _get_gemma_prompt_embeds(
            components, prompt, max_sequence_length, device, dtype
        )

        negative_prompt = block_state.negative_prompt or ""
        negative_prompt = len(prompt) * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        block_state.negative_prompt_embeds, block_state.negative_prompt_attention_mask = _get_gemma_prompt_embeds(
            components, negative_prompt, max_sequence_length, device, dtype
        )

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        self.set_block_state(state, block_state)
        return components, state


class LTX2TextConnectorStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Connector step. Adapts the Gemma hidden states into the separate video- and audio-branch text "
            "conditioning consumed by the transformer, for both the conditional and unconditional prompts."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("connectors", LTX2TextConnectors),
            # Declared only to read `padding_side` (used by the LTX-2.0 connector branch); LTX-2.5 defaults to "left".
            ComponentSpec("tokenizer", PreTrainedTokenizerBase),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_attention_mask", type_hint=torch.Tensor, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        # Plain, explicitly-named cond/uncond outputs: the denoiser reads them by name through its guider
        # `guider_input_fields` map (transformer arg -> per-pass block-state attribute names), not via the
        # `denoiser_input_fields` tag, so they are intentionally not tagged here.
        return [
            OutputParam(
                "connector_prompt_embeds", type_hint=torch.Tensor, description="Video-branch text conditioning (cond)."
            ),
            OutputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                description="Audio-branch text conditioning (cond).",
            ),
            OutputParam(
                "connector_attention_mask", type_hint=torch.Tensor, description="Binary text attention mask (cond)."
            ),
            OutputParam(
                "negative_connector_prompt_embeds",
                type_hint=torch.Tensor,
                description="Video-branch text conditioning (uncond).",
            ),
            OutputParam(
                "negative_connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                description="Audio-branch text conditioning (uncond).",
            ),
            OutputParam(
                "negative_connector_attention_mask",
                type_hint=torch.Tensor,
                description="Binary text attention mask (uncond).",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        padding_side = components.tokenizer.padding_side

        # Run the connector once on the CFG-concatenated `[uncond, cond]` batch, matching the standard pipeline
        # (`LTX2Pipeline` concatenates before the single `self.connectors(...)` call). The connector is applied per
        # batch element, so cond/uncond are mathematically independent either way, but a single batched call keeps the
        # results bitwise-identical to the standard pipeline: the connector's GEMM/attention kernels round the same for
        # a given row only at batch >= 2, so running the branches separately would diverge by ~1e-6 at batch size 1.
        num_negative = block_state.negative_prompt_embeds.shape[0]
        prompt_embeds = torch.cat([block_state.negative_prompt_embeds, block_state.prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat(
            [block_state.negative_prompt_attention_mask, block_state.prompt_attention_mask], dim=0
        )
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = components.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=padding_side
        )

        # Split back into uncond (first `num_negative`) and cond (rest).
        block_state.negative_connector_prompt_embeds = connector_prompt_embeds[:num_negative]
        block_state.negative_connector_audio_prompt_embeds = connector_audio_prompt_embeds[:num_negative]
        block_state.negative_connector_attention_mask = connector_attention_mask[:num_negative]
        block_state.connector_prompt_embeds = connector_prompt_embeds[num_negative:]
        block_state.connector_audio_prompt_embeds = connector_audio_prompt_embeds[num_negative:]
        block_state.connector_attention_mask = connector_attention_mask[num_negative:]

        self.set_block_state(state, block_state)
        return components, state


class LTX2DurationStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Predicts `num_frames` from the connector text conditioning using the `duration_head`, producing a "
            "concrete frame count snapped to the VAE's temporal grid. Run only when `num_frames` was omitted (see "
            "`LTX2AutoDurationStep`)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("duration_head", LTX2DurationHead)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "min_seconds",
                type_hint=float,
                default=1.0,
                description="Lower bound on the auto-predicted duration.",
            ),
            InputParam(
                "max_seconds",
                type_hint=float,
                default=20.0,
                description="Upper bound on the auto-predicted duration. Must be strictly greater than `min_seconds`.",
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam(
                "connector_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Video-branch text conditioning from the connector (positive prompt).",
            ),
            InputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Audio-branch text conditioning from the connector (positive prompt).",
            ),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts being denoised, used to expand conditioning per prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("num_frames", type_hint=int, description="The predicted number of frames to generate."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if getattr(components, "duration_head", None) is None:
            raise ValueError(
                "`num_frames` was omitted so the duration head would auto-predict, but no `duration_head` component "
                "is loaded (the duration head ships from LTX-2.5 checkpoints onward). Load a `duration_head`, or pass "
                "`num_frames` as an integer."
            )

        # The head predicts one duration; prompts with different natural lengths cannot share a single frame count.
        if block_state.batch_size > 1:
            raise ValueError(
                f"`num_frames` was omitted so the duration head would auto-predict, but {block_state.batch_size} "
                "prompts were supplied. The duration head predicts one duration -- run one prompt at a time, or pass "
                "`num_frames` as an integer."
            )

        if block_state.min_seconds >= block_state.max_seconds:
            raise ValueError(
                f"`min_seconds` ({block_state.min_seconds}) must be less than `max_seconds` "
                f"({block_state.max_seconds}). A collapsed range leaves no room for a prediction, and cannot "
                "generally be satisfied by a frame count on the VAE's temporal grid."
            )

        # `connector_prompt_embeds` is the positive (conditional) conditioning, one row per prompt (per-prompt
        # expansion happens later, in the denoise stage), and `batch_size > 1` was rejected above -- so it is a
        # single row here.
        block_state.num_frames = components.duration_head.predict_num_frames(
            block_state.connector_prompt_embeds,
            block_state.connector_audio_prompt_embeds,
            frame_rate=block_state.frame_rate,
            temporal_compression_ratio=components.vae_temporal_compression_ratio,
            min_seconds=block_state.min_seconds,
            max_seconds=block_state.max_seconds,
        )

        self.set_block_state(state, block_state)
        return components, state


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Normalize video latents across the channel dimension [B, C, F, H, W].
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
    latents = latents.reshape(
        batch_size,
        -1,
        num_frames // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _downsample_mask_to_latent(
    mask: torch.Tensor, latent_num_frames: int, latent_height: int, latent_width: int
) -> torch.Tensor:
    """
    Downsample a pixel-space attention mask of shape `(B, 1, F, H, W)` (values in `[0, 1]`) to a flattened per-token
    latent-space mask of shape `(B, latent_num_frames * latent_height * latent_width)`. Spatial downsampling is area
    interpolation per frame; temporal downsampling is causal (the first frame is kept as-is).
    """
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected `conditioning_attention_mask` of shape (B, 1, F, H, W), got {tuple(mask.shape)}.")
    b, _, f_pix, _, _ = mask.shape

    mask_2d = mask.reshape(b * f_pix, 1, mask.shape[-2], mask.shape[-1])
    spatial_down = torch.nn.functional.interpolate(mask_2d, size=(latent_height, latent_width), mode="area")
    spatial_down = spatial_down.reshape(b, 1, f_pix, latent_height, latent_width)

    first_frame = spatial_down[:, :, :1, :, :]
    if f_pix > 1 and latent_num_frames > 1:
        t = (f_pix - 1) // (latent_num_frames - 1)
        if (f_pix - 1) % (latent_num_frames - 1) != 0:
            raise ValueError(
                f"Pixel frames ({f_pix}) not compatible with latent frames ({latent_num_frames}): "
                f"(f_pix - 1) must be divisible by (latent_num_frames - 1)."
            )
        rest = spatial_down[:, :, 1:, :, :]
        rest = rest.reshape(b, 1, latent_num_frames - 1, t, latent_height, latent_width).mean(dim=3)
        latent_mask = torch.cat([first_frame, rest], dim=2)
    else:
        latent_mask = first_frame

    return latent_mask.reshape(b, latent_num_frames * latent_height * latent_width)


class LTX2VaeEncoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "VAE encoder step that encodes the input `image` into normalized latents for image-to-video generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            # Only used to resolve the default `image_crf` from the text-encoder generation.
            ComponentSpec("text_encoder", PreTrainedModel),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32, "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "image_crf",
                type_hint=int,
                default=None,
                description=(
                    "H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the "
                    "compression the model was trained against. `None` (default) resolves from the text-encoder "
                    "generation (33 through LTX-2.3, 18 for LTX-2.5). Pass `0` to skip re-compression. Requires a "
                    "`PIL.Image.Image` when re-compression runs."
                ),
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="Normalized image latents (a single latent frame) for image-to-video conditioning.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        image = block_state.image
        if not isinstance(image, torch.Tensor):
            # H.264 re-compress before resize/normalize (ltx-pipelines `load_image_and_preprocess`).
            crf = (
                block_state.image_crf
                if block_state.image_crf is not None
                else resolve_default_image_crf(components.text_encoder)
            )
            if crf != 0:
                if not isinstance(image, PIL.Image.Image):
                    raise ValueError(
                        f"`image_crf` re-compression requires a `PIL.Image.Image` input, got {type(image)}. "
                        "Pass a PIL image, or set `image_crf=0` to skip re-compression."
                    )
                image = PIL.Image.fromarray(apply_image_conditioning_crf(np.array(image.convert("RGB")), crf))
            image = components.video_processor.preprocess(image, height=block_state.height, width=block_state.width)
        image = image.to(device=device, dtype=torch.float32)

        vae_dtype = components.vae.dtype
        if isinstance(block_state.generator, list):
            init_latents = [
                retrieve_latents(
                    components.vae.encode(image[i].unsqueeze(0).unsqueeze(2).to(vae_dtype)),
                    block_state.generator[i],
                    "argmax",
                )
                for i in range(image.shape[0])
            ]
        else:
            init_latents = [
                retrieve_latents(
                    components.vae.encode(img.unsqueeze(0).unsqueeze(2).to(vae_dtype)), block_state.generator, "argmax"
                )
                for img in image
            ]

        init_latents = torch.cat(init_latents, dim=0).to(torch.float32)
        block_state.image_latents = _normalize_latents(init_latents, components.latents_mean, components.latents_std)

        self.set_block_state(state, block_state)
        return components, state


class LTX2ConditionEncoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Condition encoder step. Resizes and center-crops each frame condition to the target resolution, H.264 "
            "re-compresses single-frame image conditions at the model CRF, trims them to the VAE's temporal grid, "
            "and VAE-encodes them into normalized latents for `LTX2ConditionPrepareLatentsStep`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        # No `video_processor`: `VideoProcessor.preprocess_video` resizes through `PIL.Image.resize`, which
        # anti-alias prefilters on downscale. The reference uses a plain `F.interpolate`, reproduced in `__call__`.
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            # Only used to resolve a condition's default `crf` from the text-encoder generation.
            ComponentSpec("text_encoder", PreTrainedModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "conditions",
                type_hint=list,
                default=None,
                description=(
                    "`LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices "
                    "of the generated video."
                ),
            ),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=list,
                description="Per-condition normalized VAE latents of shape [1, C, F, H, W].",
            ),
            OutputParam("condition_strengths", type_hint=list, description="Per-condition conditioning strengths."),
            OutputParam(
                "condition_indices",
                type_hint=list,
                description="Per-condition latent frame index at which the condition is applied.",
            ),
            OutputParam(
                "condition_pixel_frames",
                type_hint=list,
                description=(
                    "Per-condition trimmed pixel frame count, used to clamp the temporal extent of single-frame "
                    "keyframe coordinates."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        conditions = block_state.conditions
        if conditions is None:
            conditions = []
        elif isinstance(conditions, LTX2VideoCondition):
            conditions = [conditions]

        # First block on the condition path that needs a concrete frame count. In the condition blockset
        # `LTX2AutoDurationStep` has already resolved it; the in-context blockset has no duration head, so the
        # caller must supply it.
        if block_state.num_frames is None:
            raise ValueError(
                "`num_frames` must be a concrete integer here. Pass `num_frames`, or use a blockset that runs "
                "`LTX2AutoDurationStep` with a loaded `duration_head` (LTX-2.5 checkpoints onward) to auto-predict it."
            )

        height, width, num_frames = block_state.height, block_state.width, block_state.num_frames
        frame_scale_factor = components.vae_temporal_compression_ratio
        latent_num_frames = (num_frames - 1) // frame_scale_factor + 1
        generator = block_state.generator[0] if isinstance(block_state.generator, list) else block_state.generator

        condition_latents, condition_strengths, condition_indices, condition_pixel_frames = [], [], [], []
        for i, condition in enumerate(conditions):
            # Channels-last (F, H, W, C) array, the layout the resize below expects.
            if isinstance(condition.frames, PIL.Image.Image):
                arr = np.array(condition.frames.convert("RGB"))[None]
            elif isinstance(condition.frames, list) and all(isinstance(f, PIL.Image.Image) for f in condition.frames):
                arr = np.stack([np.array(f.convert("RGB")) for f in condition.frames])
            elif isinstance(condition.frames, np.ndarray):
                arr = condition.frames if condition.frames.ndim == 4 else condition.frames[None]
            elif isinstance(condition.frames, torch.Tensor):
                t = condition.frames if condition.frames.ndim == 4 else condition.frames.unsqueeze(0)
                # Video tensors are (F, C, H, W); convert to channels-last for the resize.
                arr = t.detach().cpu().permute(0, 2, 3, 1).numpy()
            else:
                raise TypeError(f"Unsupported `frames` type for condition {i}: {type(condition.frames)}")

            # Single-frame image keyframes are H.264 re-compressed at the model CRF (ltx-pipelines
            # `ImageConditioner.resolve_crf` + `media_io.preprocess`). Multi-frame video conditions are not.
            if arr.shape[0] == 1:
                crf = (
                    condition.crf if condition.crf is not None else resolve_default_image_crf(components.text_encoder)
                )
                if crf != 0 and arr.dtype != np.uint8:
                    raise ValueError(
                        f"Image conditioning CRF expects a uint8 RGB frame, got dtype={arr.dtype}. "
                        "Pass a PIL image / uint8 array, or set `crf=0` on the condition to skip re-compression."
                    )
                arr = apply_image_conditioning_crf(arr[0], crf)[None]

            src_h, src_w = arr.shape[1], arr.shape[2]
            num_cond_frames = arr.shape[0]
            pixels = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
            pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, C, F, H, W)

            # Resize so the longer side fills the target, then center-crop to exactly (height, width).
            scale = max(height / src_h, width / src_w)
            new_h = math.ceil(src_h * scale)
            new_w = math.ceil(src_w * scale)
            pixels = pixels.permute(0, 2, 1, 3, 4).reshape(num_cond_frames, 3, src_h, src_w)
            pixels = torch.nn.functional.interpolate(pixels, size=(new_h, new_w), mode="bilinear", align_corners=False)
            top = (new_h - height) // 2
            left = (new_w - width) // 2
            pixels = pixels[:, :, top : top + height, left : left + width]
            pixels = pixels.reshape(1, num_cond_frames, 3, height, width).permute(0, 2, 1, 3, 4)

            condition_pixels = pixels / 127.5 - 1.0  # [0, 255] -> [-1, 1] (VAE input convention)

            # `index` is interpreted as a latent index, following the reference. Negative indices wrap.
            latent_start_idx = condition.index
            if latent_start_idx < 0:
                latent_start_idx = latent_start_idx % latent_num_frames
            if latent_start_idx >= latent_num_frames:
                logger.warning(
                    f"The starting latent index {latent_start_idx} of condition {i} is too big for the specified"
                    f" number of latent frames {latent_num_frames}. This condition will be skipped."
                )
                continue

            # Trim the conditioning sequence to a multiple of the VAE temporal scale factor, plus one.
            start_idx = max((latent_start_idx - 1) * frame_scale_factor + 1, 0)
            truncated_cond_frames = min(condition_pixels.size(2), num_frames - start_idx)
            truncated_cond_frames = (truncated_cond_frames - 1) // frame_scale_factor * frame_scale_factor + 1
            condition_pixels = condition_pixels[:, :, :truncated_cond_frames]

            latents = retrieve_latents(
                components.vae.encode(condition_pixels.to(dtype=components.vae.dtype, device=device)),
                generator=generator,
                sample_mode="argmax",
            )
            latents = _normalize_latents(latents, components.latents_mean, components.latents_std)

            condition_latents.append(latents.to(device=device, dtype=torch.float32))
            condition_strengths.append(condition.strength)
            condition_indices.append(latent_start_idx)
            condition_pixel_frames.append(truncated_cond_frames)

        block_state.condition_latents = condition_latents
        block_state.condition_strengths = condition_strengths
        block_state.condition_indices = condition_indices
        block_state.condition_pixel_frames = condition_pixel_frames

        self.set_block_state(state, block_state)
        return components, state


class LTX2ReferenceEncoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Reference encoder step for in-context (IC-LoRA) generation. Preprocesses each reference video to the "
            "(optionally downscaled) target resolution, VAE-encodes and packs it into tokens, and computes the "
            "positional coordinates that map those tokens into the target coordinate space. When "
            "`conditioning_attention_strength < 1.0` or a pixel-space `conditioning_attention_mask` is supplied it "
            "also produces the per-token cross-attention strengths driving the video self-attention mask."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32, "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "reference_conditions",
                type_hint=list,
                required=True,
                description=(
                    "`LTX2ReferenceCondition` (or list of them) whose videos are encoded into extra latent tokens "
                    "the IC-LoRA adapter attends to."
                ),
            ),
            InputParam(
                "reference_downscale_factor",
                type_hint=int,
                default=1,
                description=(
                    "Ratio between the target and reference resolutions; 2 means the reference is preprocessed at "
                    "half the target resolution. Spatial coordinates are scaled by this factor so the reference "
                    "tokens land in the target coordinate space. Must match the factor the IC-LoRA was trained with."
                ),
            ),
            InputParam(
                "conditioning_attention_strength",
                type_hint=float,
                default=1.0,
                description=(
                    "Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each "
                    "other. 1.0 (default) leaves attention unmasked."
                ),
            ),
            InputParam(
                "conditioning_attention_mask",
                type_hint=torch.Tensor,
                default=None,
                description=(
                    "Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially "
                    "varying attention strength. Downsampled to the reference's latent grid and multiplied by "
                    "`conditioning_attention_strength`."
                ),
            ),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_latents",
                type_hint=torch.Tensor,
                description="Packed reference tokens of shape [1, total_reference_tokens, C].",
            ),
            OutputParam(
                "reference_coords",
                type_hint=torch.Tensor,
                description="RoPE coordinates for the reference tokens, of shape [1, 3, total_reference_tokens, 2].",
            ),
            OutputParam(
                "reference_token_counts",
                type_hint=list,
                description="Per-reference token counts, in `reference_conditions` order.",
            ),
            OutputParam(
                "reference_cross_mask",
                type_hint=torch.Tensor,
                description=(
                    "Per-reference-token noisy<->reference attention strengths of shape [1, "
                    "total_reference_tokens], or `None` when attention is left unmasked."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        reference_conditions = block_state.reference_conditions
        if isinstance(reference_conditions, LTX2ReferenceCondition):
            reference_conditions = [reference_conditions]
        if len(reference_conditions) == 0:
            raise ValueError(
                "`reference_conditions` is empty, so there is nothing to encode. Omit it entirely for IC-LoRAs that "
                "take no reference video -- `LTX2AutoReferenceEncoderStep` then skips this step."
            )

        downscale_factor = block_state.reference_downscale_factor
        ref_height = block_state.height // downscale_factor
        ref_width = block_state.width // downscale_factor
        strength = block_state.conditioning_attention_strength
        attention_mask = block_state.conditioning_attention_mask
        # An all-ones mask at full strength is the same as no mask, so only materialize one when it can bite.
        mask_needed = strength < 1.0 or attention_mask is not None
        generator = block_state.generator[0] if isinstance(block_state.generator, list) else block_state.generator

        all_latents, all_coords, all_cross_masks, token_counts = [], [], [], []
        for ref_cond in reference_conditions:
            if isinstance(ref_cond.frames, PIL.Image.Image):
                video_like = [ref_cond.frames]
            elif isinstance(ref_cond.frames, np.ndarray) and ref_cond.frames.ndim == 3:
                video_like = np.expand_dims(ref_cond.frames, axis=0)
            elif isinstance(ref_cond.frames, torch.Tensor) and ref_cond.frames.ndim == 3:
                video_like = ref_cond.frames.unsqueeze(0)
            else:
                video_like = ref_cond.frames

            ref_pixels = components.video_processor.preprocess_video(
                video_like, ref_height, ref_width, resize_mode="crop"
            )
            ref_pixels = ref_pixels[:, :, : block_state.num_frames]
            ref_pixels = ref_pixels.to(dtype=components.vae.dtype, device=device)

            ref_latent = retrieve_latents(components.vae.encode(ref_pixels), generator=generator, sample_mode="argmax")
            ref_latent = _normalize_latents(ref_latent, components.latents_mean, components.latents_std).to(
                device=device, dtype=torch.float32
            )
            _, _, ref_latent_frames, ref_latent_height, ref_latent_width = ref_latent.shape
            ref_latent_packed = _pack_latents(
                ref_latent, components.transformer_spatial_patch_size, components.transformer_temporal_patch_size
            )

            # Coordinates are computed on the reference's own latent grid, then scaled spatially so the tokens map
            # into the target's coordinate space (preserving the positional relationship the IC-LoRA was trained on).
            ref_coords = components.transformer.rope.prepare_video_coords(
                batch_size=1,
                num_frames=ref_latent_frames,
                height=ref_latent_height,
                width=ref_latent_width,
                device=device,
                fps=block_state.frame_rate,
            )
            if downscale_factor != 1:
                ref_coords[:, 1, :, :] = ref_coords[:, 1, :, :] * downscale_factor
                ref_coords[:, 2, :, :] = ref_coords[:, 2, :, :] * downscale_factor

            if mask_needed:
                if attention_mask is not None:
                    ref_cross = _downsample_mask_to_latent(
                        attention_mask, ref_latent_frames, ref_latent_height, ref_latent_width
                    ).to(device=device, dtype=torch.float32)
                else:
                    ref_cross = torch.ones((1, ref_latent_packed.shape[1]), device=device, dtype=torch.float32)
                all_cross_masks.append(ref_cross * strength)

            all_latents.append(ref_latent_packed)
            all_coords.append(ref_coords)
            token_counts.append(ref_latent_packed.shape[1])

        block_state.reference_latents = torch.cat(all_latents, dim=1)
        block_state.reference_coords = torch.cat(all_coords, dim=2)
        block_state.reference_token_counts = token_counts
        block_state.reference_cross_mask = torch.cat(all_cross_masks, dim=1) if mask_needed else None

        self.set_block_state(state, block_state)
        return components, state
