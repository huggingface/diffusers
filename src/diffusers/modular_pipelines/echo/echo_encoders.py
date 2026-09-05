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

from __future__ import annotations

import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ...models import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
from ...pipelines.ltx2.connectors import LTX2TextConnectors
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)

MAX_MEMORY_SLOTS = 7
MAX_MEMORY_AUDIO_DURATION_SECONDS = 9.62


def _validate_memory_slot_count(slot_count: int) -> None:
    if slot_count > MAX_MEMORY_SLOTS:
        raise ValueError(f"Echo accepts at most {MAX_MEMORY_SLOTS} memory slots, but received {slot_count}.")


def _get_prompt_embeds(
    components,
    prompt: str | list[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    components.tokenizer.padding_side = "left"
    if components.tokenizer.pad_token is None:
        components.tokenizer.pad_token = components.tokenizer.eos_token

    prompt = [text.strip() for text in prompt]
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
        input_ids=text_input_ids,
        attention_mask=prompt_attention_mask,
        output_hidden_states=True,
    )
    text_encoder_hidden_states = torch.stack(text_encoder_outputs.hidden_states, dim=-1)
    prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(dtype=dtype)

    return prompt_embeds, prompt_attention_mask


class EchoTextEncoderStep(ModularPipelineBlocks):
    """Encode only the positive prompt used by the guidance-free Echo DMD checkpoint."""

    model_name = "echo"

    @property
    def description(self) -> str:
        return "Encodes the positive Echo prompt into packed per-layer Gemma hidden states."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", PreTrainedModel),
            ComponentSpec("tokenizer", PreTrainedTokenizerBase),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
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
                "batch_size",
                type_hint=int,
                description="Number of prompts before per-prompt expansion.",
            ),
            OutputParam("dtype", type_hint=torch.dtype, description="Prompt embedding dtype."),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and not isinstance(block_state.prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        prompt = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        block_state.prompt_embeds, block_state.prompt_attention_mask = _get_prompt_embeds(
            components,
            prompt,
            block_state.max_sequence_length,
            components._execution_device,
            components.text_encoder.dtype,
        )
        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        self.set_block_state(state, block_state)
        return components, state


class EchoTextConnectorStep(ModularPipelineBlocks):
    """Project positive Gemma embeddings into Echo's video and audio context spaces."""

    model_name = "echo"

    @property
    def description(self) -> str:
        return "Adapts positive Gemma embeddings for the Echo video and audio transformer branches."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("connectors", LTX2TextConnectors),
            ComponentSpec("tokenizer", PreTrainedTokenizerBase),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "connector_prompt_embeds",
                type_hint=torch.Tensor,
                description="Video-branch positive text conditioning.",
            ),
            OutputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                description="Audio-branch positive text conditioning.",
            ),
            OutputParam(
                "connector_attention_mask",
                type_hint=torch.Tensor,
                description="Binary attention mask for the positive text conditioning.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        (
            block_state.connector_prompt_embeds,
            block_state.connector_audio_prompt_embeds,
            block_state.connector_attention_mask,
        ) = components.connectors(
            block_state.prompt_embeds,
            block_state.prompt_attention_mask,
            padding_side=components.tokenizer.padding_side,
        )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class EchoTextConditioningStep(SequentialPipelineBlocks):
    """
    Positive-only text conditioning for the guidance-free Echo DMD checkpoint.

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              Packed per-layer Gemma hidden states for the prompt.
          prompt_attention_mask (`Tensor`):
              Binary attention mask for `prompt_embeds`.
          batch_size (`int`):
              Number of prompts before per-prompt expansion.
          dtype (`dtype`):
              Prompt embedding dtype.
          connector_prompt_embeds (`Tensor`):
              Video-branch positive text conditioning.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch positive text conditioning.
          connector_attention_mask (`Tensor`):
              Binary attention mask for the positive text conditioning.
    """

    model_name = "echo"
    block_classes = [EchoTextEncoderStep, EchoTextConnectorStep]
    block_names = ["text_encoder", "connectors"]

    @property
    def description(self) -> str:
        return "Positive-only text conditioning for the guidance-free Echo DMD checkpoint."


# Copied from diffusers.modular_pipelines.ltx2.before_denoise._pack_latents
def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


# Copied from diffusers.modular_pipelines.ltx2.before_denoise._pack_audio_latents
def _pack_audio_latents(
    latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
) -> torch.Tensor:
    # Audio latents of shape [B, C, L, M] (L = latent audio length, M = mel bins). With no patch sizes this packs to
    # [B, L, C * M] (implicit mel patch_size of M, temporal patch_size of 1).
    if patch_size is not None and patch_size_t is not None:
        batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
        post_patch_latent_length = latent_length / patch_size_t
        post_patch_mel_bins = latent_mel_bins / patch_size
        latents = latents.reshape(
            batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
    else:
        latents = latents.transpose(1, 2).flatten(2, 3)  # [B, C, L, M] -> [B, L, C * M]
    return latents


# Copied from diffusers.modular_pipelines.ltx2.before_denoise._normalize_latents
def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


# Copied from diffusers.modular_pipelines.ltx2.before_denoise._normalize_audio_latents
def _normalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _prepare_image(image, height: int, width: int) -> torch.Tensor:
    if isinstance(image, PIL.Image.Image):
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), PIL.Image.Resampling.BICUBIC)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1.0
        return image.unsqueeze(0)

    image = torch.as_tensor(image).detach().float()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(
            "Each Echo image must be a PIL image or a tensor with shape (3, height, width) or "
            f"(1, 3, height, width), but got {tuple(image.shape)}."
        )
    if image.shape[-2:] != (height, width):
        image = F.interpolate(image, size=(height, width), mode="bicubic", align_corners=False)
    if image.amin() >= 0:
        image = image / 127.5 - 1.0 if image.amax() > 1.0 else image * 2.0 - 1.0
    return image


def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = torch.as_tensor(waveform).detach().float()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] < 1:
        raise ValueError(
            "Each Echo memory waveform must have shape (samples,) or (channels, samples), "
            f"but got {tuple(waveform.shape)}."
        )
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform.contiguous()


def _max_response_window_start(mel: torch.Tensor, window_size: int) -> int:
    num_time_steps = mel.shape[2]
    max_start = num_time_steps - window_size
    scan_stride = max(1, window_size // 4)
    candidate_starts = list(range(0, max_start + 1, scan_stride))
    if candidate_starts[-1] != max_start:
        candidate_starts.append(max_start)

    response = mel.float().exp().sum(dim=(0, 1, 3))
    cumulative_response = torch.cat([response.new_zeros(1), response.cumsum(dim=0)])
    starts = torch.tensor(candidate_starts, device=mel.device, dtype=torch.long)
    scores = cumulative_response[starts + window_size] - cumulative_response[starts]
    return candidate_starts[int(scores.argmax().item())]


class EchoConditionEncoderStep(ModularPipelineBlocks):
    """Encode Echo's clean first frame and ordered image/audio memory slots."""

    model_name = "echo"

    @property
    def description(self) -> str:
        return (
            "Encodes an optional clean first frame and ordered image/audio memory slots into normalized tokens, "
            "and assigns each memory slot its Echo slot-center RoPE coordinates."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image",
                type_hint=PIL.Image.Image | torch.Tensor,
                default=None,
                description="Optional single first frame used as a clean reference condition.",
            ),
            InputParam(
                "memory_images",
                type_hint=list,
                default=None,
                description="Ordered reference images, one per Echo memory slot.",
            ),
            InputParam(
                "memory_audio_waveforms",
                type_hint=list,
                default=None,
                description=(
                    "Ordered memory waveforms as `(channels, samples)` tensors. Inputs longer than 9.62 seconds are "
                    "cropped to their highest-response window. Use `None` for a silent slot."
                ),
            ),
            InputParam(
                "memory_audio_sample_rates",
                type_hint=int | list,
                default=None,
                description="Sampling rate shared by all memory waveforms, or one rate per slot.",
            ),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "model_frame_rate",
                type_hint=float,
                default=24.0,
                description="Training-time frame rate used for Echo video RoPE coordinates.",
            ),
            InputParam(
                "memory_position_offset",
                type_hint=float,
                default=500.0,
                description="Temporal center assigned to the first memory slot.",
            ),
            InputParam(
                "memory_position_slot_stride",
                type_hint=float,
                default=50.0,
                description="Temporal distance between consecutive memory-slot centers.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "first_frame_tokens", type_hint=torch.Tensor, description="Normalized packed first-frame tokens."
            ),
            OutputParam(
                "memory_video_tokens", type_hint=torch.Tensor, description="Normalized packed image-memory tokens."
            ),
            OutputParam(
                "memory_video_coords", type_hint=torch.Tensor, description="RoPE coordinates for image-memory tokens."
            ),
            OutputParam(
                "memory_audio_tokens", type_hint=torch.Tensor, description="Normalized packed audio-memory tokens."
            ),
            OutputParam(
                "memory_audio_coords", type_hint=torch.Tensor, description="RoPE coordinates for audio-memory tokens."
            ),
        ]

    @staticmethod
    def _encode_image(components, image, height: int, width: int, device: torch.device) -> torch.Tensor:
        pixels = _prepare_image(image, height, width).unsqueeze(2).to(device=device, dtype=components.vae.dtype)
        latents = components.vae.encode(pixels).latent_dist.mode().float()
        latents = _normalize_latents(
            latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
        )
        return _pack_latents(
            latents,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        )

    @staticmethod
    def _encode_audio(components, waveform: torch.Tensor, sample_rate: int, device: torch.device) -> torch.Tensor:
        try:
            import torchaudio
        except ImportError as error:
            raise ImportError(
                "Encoding raw `memory_audio_waveforms` requires torchaudio. Install torchaudio before running the "
                "Echo condition encoder."
            ) from error

        if sample_rate <= 0:
            raise ValueError(f"`memory_audio_sample_rates` values must be positive, but got {sample_rate}.")

        waveform = _normalize_waveform(waveform).to(device=device, dtype=torch.float32)
        target_rate = int(components.audio_vae.config.sample_rate)
        n_fft = 1024
        min_input_samples = math.ceil(n_fft * sample_rate / target_rate)
        if waveform.shape[-1] < min_input_samples:
            waveform = F.pad(waveform, (0, min_input_samples - waveform.shape[-1]))
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        if waveform.shape[-1] < n_fft:
            waveform = F.pad(waveform, (0, n_fft - waveform.shape[-1]))
        hop_length = int(components.audio_vae.config.mel_hop_length)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=0.0,
            f_max=target_rate / 2.0,
            n_mels=int(components.audio_vae.config.mel_bins),
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        ).to(device)
        mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5)).permute(0, 2, 1).unsqueeze(0)

        max_samples = round(MAX_MEMORY_AUDIO_DURATION_SECONDS * target_rate)
        if waveform.shape[-1] > max_samples:
            max_mel_steps = max_samples // hop_length + 1
            start_mel_step = _max_response_window_start(mel, max_mel_steps)
            start_sample = min(start_mel_step * hop_length, waveform.shape[-1] - max_samples)
            waveform = waveform[..., start_sample : start_sample + max_samples]
            mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5)).permute(0, 2, 1).unsqueeze(0)

        latents = components.audio_vae.encode(mel.to(components.audio_vae.dtype)).latent_dist.mode()
        latents = _pack_audio_latents(latents)
        return _normalize_audio_latents(latents, components.audio_latents_mean, components.audio_latents_std).float()

    @staticmethod
    def _video_memory_coords(
        components,
        slots: int,
        latent_height: int,
        latent_width: int,
        frame_rate: float,
        position_offset: float,
        slot_stride: float,
        device: torch.device,
    ) -> torch.Tensor:
        base = components.transformer.rope.prepare_video_coords(
            1, 1, latent_height, latent_width, device, fps=frame_rate
        )
        coords = []
        for slot_index in range(slots):
            slot_coords = base.clone()
            center = position_offset + slot_index * slot_stride
            midpoint = (slot_coords[:, 0, :1, 0] + slot_coords[:, 0, :1, 1]) * 0.5
            slot_coords[:, 0] += (center - midpoint).view(1, 1, 1)
            coords.append(slot_coords)
        return torch.cat(coords, dim=2)

    @staticmethod
    def _audio_memory_coords(
        components,
        lengths: list[int],
        position_offset: float,
        slot_stride: float,
        device: torch.device,
    ) -> torch.Tensor:
        coords = []
        for slot_index, length in enumerate(lengths):
            slot_coords = components.transformer.audio_rope.prepare_audio_coords(1, length, device)
            center = position_offset + slot_index * slot_stride
            midpoint = (slot_coords[:, 0, :1, 0] + slot_coords[:, 0, -1:, 1]) * 0.5
            slot_coords[:, 0] += (center - midpoint).view(1, 1, 1)
            coords.append(slot_coords)
        return torch.cat(coords, dim=2)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        transformer_dtype = components.transformer.dtype

        if block_state.image is not None:
            first_frame_tokens = self._encode_image(
                components, block_state.image, block_state.height, block_state.width, device
            )
        else:
            first_frame_tokens = None

        memory_images = _as_list(block_state.memory_images)
        slot_count = len(memory_images)
        _validate_memory_slot_count(slot_count)

        video_slots = [
            self._encode_image(components, image, block_state.height, block_state.width, device)
            for image in memory_images
        ]
        memory_video_tokens = torch.cat(video_slots, dim=1) if video_slots else None

        raw_audio = _as_list(block_state.memory_audio_waveforms)
        if raw_audio and len(raw_audio) != slot_count:
            raise ValueError("`memory_audio_waveforms` must have one entry per `memory_images` slot.")

        audio_slots: list[torch.Tensor | None] = []
        if raw_audio:
            rates = block_state.memory_audio_sample_rates
            if rates is None:
                rates = int(components.audio_vae.config.sample_rate)
            rates = [rates] * len(raw_audio) if isinstance(rates, int) else list(rates)
            if len(rates) != len(raw_audio):
                raise ValueError("`memory_audio_sample_rates` must be an int or have one entry per waveform.")
            if components.audio_vae.dtype != torch.float32:
                logger.warning_once(
                    "Echo was trained with FP32 audio-memory encoding. For parity, load `audio_vae` with "
                    "`dtype={'default': torch.bfloat16, 'audio_vae': torch.float32}`."
                )
            for waveform, sample_rate in zip(raw_audio, rates):
                audio_slots.append(
                    None if waveform is None else self._encode_audio(components, waveform, int(sample_rate), device)
                )
        template = next((value for value in audio_slots if value is not None), None)
        memory_audio_tokens = None
        memory_audio_coords = None
        if template is not None:
            aligned_audio = [value if value is not None else torch.zeros_like(template) for value in audio_slots]
            lengths = [value.shape[1] for value in aligned_audio]
            memory_audio_tokens = torch.cat(aligned_audio, dim=1)
            memory_audio_coords = self._audio_memory_coords(
                components,
                lengths,
                block_state.memory_position_offset,
                block_state.memory_position_slot_stride,
                device,
            )

        memory_video_coords = None
        if memory_video_tokens is not None:
            latent_height = block_state.height // components.vae_spatial_compression_ratio
            latent_width = block_state.width // components.vae_spatial_compression_ratio
            memory_video_coords = self._video_memory_coords(
                components,
                len(video_slots),
                latent_height,
                latent_width,
                block_state.model_frame_rate,
                block_state.memory_position_offset,
                block_state.memory_position_slot_stride,
                device,
            )

        block_state.first_frame_tokens = (
            None if first_frame_tokens is None else first_frame_tokens.to(dtype=transformer_dtype)
        )
        block_state.memory_video_tokens = (
            None if memory_video_tokens is None else memory_video_tokens.to(dtype=transformer_dtype)
        )
        block_state.memory_video_coords = memory_video_coords
        block_state.memory_audio_tokens = (
            None if memory_audio_tokens is None else memory_audio_tokens.to(dtype=transformer_dtype)
        )
        block_state.memory_audio_coords = memory_audio_coords

        self.set_block_state(state, block_state)
        return components, state
