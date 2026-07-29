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

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLLTX2Video

# NOTE (modular.md gotcha #1): `LTX2TextConnectors`, `LTX2DurationHead`, `LTX2AutoDuration`, and the
# prompt-enhancement constants live under `diffusers.pipelines.ltx2.*`, and modular blocks must not import from
# `diffusers.pipelines.*`. `LTX2TextConnectors` / `LTX2DurationHead` are `ModelMixin` / `ConfigMixin` model classes
# (relocate to `src/diffusers/models/`); `LTX2AutoDuration`, the enhancement config, and the system prompts are plain
# data (relocate to a neutral shared module or copy into this package). Imported from the pipelines path here only so
# the draft is runnable.
from ...pipelines.ltx2.connectors import LTX2TextConnectors
from ...pipelines.ltx2.duration_head import LTX2AutoDuration, LTX2DurationHead
from ...pipelines.ltx2.utils import (
    GEMMA4_PROMPT_ENHANCEMENT_CONFIG,
    LTX2_4_I2V_DEFAULT_SYSTEM_PROMPT,
    LTX2_4_T2V_DEFAULT_SYSTEM_PROMPT,
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
    max_new_tokens: int,
    seed: int,
    generator: torch.Generator | None,
    generation_kwargs: dict[str, Any] | None,
    device: torch.device,
) -> list[str]:
    # Mirrors `LTX2Pipeline.enhance_prompt` / `LTX2ImageToVideoPipeline.enhance_prompt` for the LTX-2.4 path only:
    # a dedicated `prompt_enhancer` (Gemma-4) with the greedy `GEMMA4_PROMPT_ENHANCEMENT_CONFIG` recipe. The
    # LTX-2.0/2.3 `text_encoder`-as-enhancer fallback is intentionally dropped (LTX-2.4-only integration).
    config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG
    generation_kwargs = generation_kwargs if generation_kwargs is not None else config.generation_kwargs

    user_text = f"{config.user_prompt_prefix}: {prompt}"
    if image is None:
        user_content = user_text
    else:
        user_content = [{"type": "image"}, {"type": "text", "text": user_text}]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    template = components.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = components.processor(text=template, images=image, return_tensors="pt").to(device)
    components.prompt_enhancer.to(device)

    # `transformers.GenerationMixin.generate` does not support a `torch.Generator`, so seed manually for
    # reproducibility. (Inert for LTX-2.4's greedy decoding, but honored if the user passes sampling kwargs.)
    if generator is not None:
        seed = generator.initial_seed()
    torch.manual_seed(seed)
    generated_sequences = components.prompt_enhancer.generate(
        **model_inputs, max_new_tokens=max_new_tokens, **generation_kwargs
    )

    generated_ids = [seq[len(model_inputs.input_ids[i]) :] for i, seq in enumerate(generated_sequences)]
    return components.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def _get_gemma_prompt_embeds(
    components,
    prompt: str | list[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    # Mirrors `LTX2Pipeline._get_gemma_prompt_embeds`, minus the `num_videos_per_prompt` expansion (that happens
    # in the dedicated text-input step so pre-encoded embeds stay reusable across runs).
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
            "LTX-2.4 `prompt_enhancer` (a Gemma conditional-generation model) and a system prompt."
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
                "system_prompt",
                type_hint=str,
                default=None,
                description="System prompt for enhancement. Defaults to `LTX2_4_T2V_DEFAULT_SYSTEM_PROMPT`.",
            ),
            InputParam(
                "prompt_max_new_tokens",
                type_hint=int,
                default=512,
                description="Maximum number of new tokens to generate during prompt enhancement.",
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
                description="Random seed for prompt enhancement (inert under LTX-2.4's greedy decoding).",
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
        system_prompt = block_state.system_prompt or LTX2_4_T2V_DEFAULT_SYSTEM_PROMPT

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
            "`image`, using the dedicated LTX-2.4 `prompt_enhancer` and a system prompt."
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
                "system_prompt",
                type_hint=str,
                default=None,
                description="System prompt for enhancement. Defaults to `LTX2_4_I2V_DEFAULT_SYSTEM_PROMPT`.",
            ),
            InputParam(
                "prompt_max_new_tokens",
                type_hint=int,
                default=512,
                description="Maximum number of new tokens to generate during prompt enhancement.",
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
                description="Random seed for prompt enhancement (inert under LTX-2.4's greedy decoding).",
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
        system_prompt = block_state.system_prompt or LTX2_4_I2V_DEFAULT_SYSTEM_PROMPT

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
            "that the connectors adapt for the video and audio branches."
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
            # Declared only to read `padding_side` (used by the LTX-2.0 connector branch); LTX-2.4 defaults to "left".
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
        # NOTE: the exact names / `kwargs_type` tagging here should be reconciled with the denoise step's input
        # contract when `denoise.py` is written (the manual-guidance denoiser assembles the CFG/STG/modality batches
        # from these). Kept as plain, explicitly-named cond/uncond outputs for now.
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
        padding_side = getattr(components.tokenizer, "padding_side", "left")

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
            "Predicts `num_frames` from the connector text conditioning using the `duration_head`, replacing an "
            "`LTX2AutoDuration` request with a concrete frame count snapped to the VAE's temporal grid. Run only when "
            "`num_frames` is an `LTX2AutoDuration` (see `LTX2AutoDurationStep`)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("duration_head", LTX2DurationHead)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_frames",
                type_hint=LTX2AutoDuration,
                required=True,
                description="An `LTX2AutoDuration` request carrying the `[min_seconds, max_seconds]` bounds.",
            ),
            InputParam("frame_rate", type_hint=float, default=24.0),
            InputParam("connector_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("connector_audio_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("batch_size", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("num_frames", type_hint=int, description="The predicted number of frames to generate."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # The head predicts one duration; prompts with different natural lengths cannot share a single frame count.
        if block_state.batch_size > 1:
            raise ValueError(
                f"`num_frames` was an `LTX2AutoDuration` but {block_state.batch_size} prompts were supplied. The "
                "duration head predicts one duration -- run one prompt at a time, or pass `num_frames` as an integer."
            )

        auto_duration = block_state.num_frames
        # `connector_prompt_embeds` is already the positive (conditional) conditioning; rows past the first are
        # `num_videos_per_prompt` duplicates of the same prompt, so predict from the first.
        block_state.num_frames = components.duration_head.predict_num_frames(
            block_state.connector_prompt_embeds[:1],
            block_state.connector_audio_prompt_embeds[:1],
            frame_rate=block_state.frame_rate,
            temporal_compression_ratio=components.vae_temporal_compression_ratio,
            min_seconds=auto_duration.min_seconds,
            max_seconds=auto_duration.max_seconds,
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
