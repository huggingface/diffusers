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

from typing import Any

import numpy as np
import PIL.Image
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLLTX2Video

# NOTE (modular.md gotcha #1): `LTX2TextConnectors`, `LTX2DurationHead`, the prompt-enhancement config/helpers, and
# the system prompts live under `diffusers.pipelines.ltx2.*`, and modular blocks must not import from
# `diffusers.pipelines.*`. `LTX2TextConnectors` / `LTX2DurationHead` are `ModelMixin` / `ConfigMixin` model classes
# (relocate to `src/diffusers/models/`); the enhancement config, the response/image helpers, and the system prompts are
# plain data and utilities (relocate to a neutral shared module or copy into this package). Imported from the pipelines
# path here only so the draft is runnable.
from ...pipelines.ltx2.connectors import LTX2TextConnectors
from ...pipelines.ltx2.duration_head import LTX2DurationHead
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
    # Mirrors `LTX2PromptEnhancementMixin.enhance_prompt` for the LTX-2.5 path only: a dedicated `prompt_enhancer`
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
        block_state.image_latents = _normalize_latents(
            init_latents, components.vae.latents_mean, components.vae.latents_std
        )

        self.set_block_state(state, block_state)
        return components, state
