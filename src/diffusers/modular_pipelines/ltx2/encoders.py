# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer, GemmaTokenizerFast

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models.autoencoders import AutoencoderKLLTX2Video
from ...pipelines.ltx2.connectors import LTX2TextConnectors
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


@dataclass
class LTX2VideoCondition:
    """
    Defines a single frame-conditioning item for LTX-2 Video.

    Attributes:
        frames: The image (or video) to condition on.
        index: The latent index at which to insert the condition.
        strength: The strength of the conditioning effect (0-1).
    """

    frames: PIL.Image.Image | list[PIL.Image.Image] | np.ndarray | torch.Tensor
    index: int = 0
    strength: float = 1.0


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


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: str | torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype

    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = mask[:, :, None, None]

    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized_hidden_states = normalized_hidden_states * scale_factor

    normalized_hidden_states = normalized_hidden_states.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
    normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
    normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)
    return normalized_hidden_states


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size, -1, post_patch_num_frames, patch_size_t, post_patch_height, patch_size, post_patch_width, patch_size
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


class LTX2TextEncoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Text encoder step that encodes prompts using Gemma3 for LTX2 video generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Gemma3ForConditionalGeneration),
            ComponentSpec("tokenizer", GemmaTokenizerFast),
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
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam("max_sequence_length", default=1024),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Packed text embeddings from Gemma3",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Packed negative text embeddings from Gemma3",
            ),
            OutputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Attention mask for prompt embeddings",
            ),
            OutputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Attention mask for negative prompt embeddings",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

    @staticmethod
    def _get_gemma_prompt_embeds(
        text_encoder,
        tokenizer,
        prompt: str | list[str],
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        text_encoder_outputs = text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        # Return raw stacked hidden states [B, T, D, L] — the connector handles normalization
        # (per_token_rms_norm + rescaling for LTX-2.3, or _pack_text_embeds for LTX-2.0)
        prompt_embeds = text_encoder_hidden_states.to(dtype=dtype)

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def encode_prompt(
        components,
        prompt: str | list[str],
        device: torch.device | None = None,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 1024,
    ):
        device = device or components._execution_device

        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = LTX2TextEncoderStep._get_gemma_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

        if prepare_unconditional_embeds:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = LTX2TextEncoderStep._get_gemma_prompt_embeds(
                text_encoder=components.text_encoder,
                tokenizer=components.tokenizer,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        device = components._execution_device

        (
            block_state.prompt_embeds,
            block_state.prompt_attention_mask,
            block_state.negative_prompt_embeds,
            block_state.negative_prompt_attention_mask,
        ) = self.encode_prompt(
            components=components,
            prompt=block_state.prompt,
            device=device,
            prepare_unconditional_embeds=components.requires_unconditional_embeds,
            negative_prompt=block_state.negative_prompt,
            max_sequence_length=block_state.max_sequence_length,
        )

        self.set_block_state(state, block_state)
        return components, state


class LTX2ConnectorStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Connector step that transforms text embeddings into video and audio conditioning embeddings"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("connectors", LTX2TextConnectors),
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
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("prompt_attention_mask", required=True, type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_attention_mask", type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "connector_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Video text embeddings from connector",
            ),
            OutputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Audio text embeddings from connector",
            ),
            OutputParam(
                "connector_attention_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Attention mask from connector",
            ),
            OutputParam(
                "connector_negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Negative video text embeddings from connector",
            ),
            OutputParam(
                "connector_audio_negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Negative audio text embeddings from connector",
            ),
            OutputParam(
                "connector_negative_attention_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Negative attention mask from connector",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_embeds = block_state.prompt_embeds
        prompt_attention_mask = block_state.prompt_attention_mask
        negative_prompt_embeds = block_state.negative_prompt_embeds
        negative_prompt_attention_mask = block_state.negative_prompt_attention_mask

        do_cfg = negative_prompt_embeds is not None

        if do_cfg:
            combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        else:
            combined_embeds = prompt_embeds
            combined_mask = prompt_attention_mask

        connector_embeds, connector_audio_embeds, connector_mask = components.connectors(
            combined_embeds, combined_mask
        )

        if do_cfg:
            batch_size = prompt_embeds.shape[0]
            block_state.connector_negative_prompt_embeds = connector_embeds[:batch_size]
            block_state.connector_prompt_embeds = connector_embeds[batch_size:]
            block_state.connector_audio_negative_prompt_embeds = connector_audio_embeds[:batch_size]
            block_state.connector_audio_prompt_embeds = connector_audio_embeds[batch_size:]
            block_state.connector_negative_attention_mask = connector_mask[:batch_size]
            block_state.connector_attention_mask = connector_mask[batch_size:]
        else:
            block_state.connector_prompt_embeds = connector_embeds
            block_state.connector_audio_prompt_embeds = connector_audio_embeds
            block_state.connector_attention_mask = connector_mask
            block_state.connector_negative_prompt_embeds = None
            block_state.connector_audio_negative_prompt_embeds = None
            block_state.connector_negative_attention_mask = None

        self.set_block_state(state, block_state)
        return components, state


class LTX2ConditionEncoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Condition encoder step that VAE-encodes conditioning frames for I2V and conditional generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("conditions", type_hint=list, description="List of LTX2VideoCondition objects"),
            InputParam("image", type_hint=PIL.Image.Image, description="Sugar for I2V: image to condition on frame 0"),
            InputParam("height", default=512, type_hint=int),
            InputParam("width", default=768, type_hint=int),
            InputParam("num_frames", default=121, type_hint=int),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("condition_latents", type_hint=list, description="List of packed condition latent tensors"),
            OutputParam("condition_strengths", type_hint=list, description="List of conditioning strengths"),
            OutputParam("condition_indices", type_hint=list, description="List of latent frame indices"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        conditions = block_state.conditions
        image = block_state.image

        # Convert image sugar to conditions list
        if image is not None and conditions is None:
            conditions = [LTX2VideoCondition(frames=image, index=0, strength=1.0)]

        if conditions is None:
            block_state.condition_latents = []
            block_state.condition_strengths = []
            block_state.condition_indices = []
            self.set_block_state(state, block_state)
            return components, state

        if isinstance(conditions, LTX2VideoCondition):
            conditions = [conditions]

        height = block_state.height
        width = block_state.width
        num_frames = block_state.num_frames
        device = components._execution_device
        generator = block_state.generator

        vae_temporal_compression_ratio = components.vae.temporal_compression_ratio
        vae_spatial_compression_ratio = components.vae.spatial_compression_ratio
        transformer_spatial_patch_size = 1
        transformer_temporal_patch_size = 1

        latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1

        conditioning_frames, conditioning_strengths, conditioning_indices = [], [], []

        for i, condition in enumerate(conditions):
            if isinstance(condition.frames, PIL.Image.Image):
                video_like_cond = [condition.frames]
            elif isinstance(condition.frames, np.ndarray) and condition.frames.ndim == 3:
                video_like_cond = np.expand_dims(condition.frames, axis=0)
            elif isinstance(condition.frames, torch.Tensor) and condition.frames.ndim == 3:
                video_like_cond = condition.frames.unsqueeze(0)
            else:
                video_like_cond = condition.frames

            condition_pixels = components.video_processor.preprocess_video(
                video_like_cond, height, width, resize_mode="crop"
            )

            latent_start_idx = condition.index
            if latent_start_idx < 0:
                latent_start_idx = latent_start_idx % latent_num_frames
            if latent_start_idx >= latent_num_frames:
                logger.warning(
                    f"The starting latent index {latent_start_idx} of condition {i} is too big for {latent_num_frames} "
                    f"latent frames. This condition will be skipped."
                )
                continue

            cond_num_frames = condition_pixels.size(2)
            start_idx = max((latent_start_idx - 1) * vae_temporal_compression_ratio + 1, 0)
            frame_num = min(cond_num_frames, num_frames - start_idx)
            frame_num = (frame_num - 1) // vae_temporal_compression_ratio * vae_temporal_compression_ratio + 1
            condition_pixels = condition_pixels[:, :, :frame_num]

            conditioning_frames.append(condition_pixels.to(dtype=components.vae.dtype, device=device))
            conditioning_strengths.append(condition.strength)
            conditioning_indices.append(latent_start_idx)

        condition_latents = []
        for condition_tensor in conditioning_frames:
            condition_latent = retrieve_latents(
                components.vae.encode(condition_tensor), generator=generator, sample_mode="argmax"
            )
            condition_latent = _normalize_latents(
                condition_latent, components.vae.latents_mean, components.vae.latents_std
            ).to(device=device, dtype=torch.float32)
            condition_latent = _pack_latents(
                condition_latent, transformer_spatial_patch_size, transformer_temporal_patch_size
            )
            condition_latents.append(condition_latent)

        block_state.condition_latents = condition_latents
        block_state.condition_strengths = conditioning_strengths
        block_state.condition_indices = conditioning_indices

        self.set_block_state(state, block_state)
        return components, state
