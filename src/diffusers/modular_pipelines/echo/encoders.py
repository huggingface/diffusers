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

from ...models import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...pipelines.ltx2.connectors import LTX2TextConnectors
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)

MAX_MEMORY_SLOTS = 7
MAX_MEMORY_AUDIO_DURATION_SECONDS = 9.62


def _validate_memory_slot_count(slot_count: int) -> None:
    if slot_count > MAX_MEMORY_SLOTS:
        raise ValueError(f"Echo accepts at most {MAX_MEMORY_SLOTS} memory slots, but received {slot_count}.")


def _get_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: str | list[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = [text.strip() for text in prompt]
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
            components.tokenizer,
            components.text_encoder,
            prompt,
            block_state.max_sequence_length,
            components._execution_device,
            components.text_encoder.dtype,
        )

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


def _encode_image(
    vae,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float,
    image,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    pixels = _prepare_image(image, height, width).unsqueeze(2).to(device=device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode().float()
    return _normalize_latents(latents, latents_mean, latents_std, scaling_factor)


def _encode_audio(
    audio_vae,
    audio_latents_mean: torch.Tensor,
    audio_latents_std: torch.Tensor,
    waveform: torch.Tensor,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        import torchaudio
    except ImportError as error:
        raise ImportError(
            "Encoding raw `memory_audio_waveforms` requires torchaudio. Install torchaudio before running the "
            "Echo VAE encoder."
        ) from error

    if sample_rate <= 0:
        raise ValueError(f"`memory_audio_sample_rates` values must be positive, but got {sample_rate}.")

    waveform = _normalize_waveform(waveform).to(device=device, dtype=torch.float32)
    target_rate = int(audio_vae.config.sample_rate)
    n_fft = 1024
    min_input_samples = math.ceil(n_fft * sample_rate / target_rate)
    if waveform.shape[-1] < min_input_samples:
        waveform = F.pad(waveform, (0, min_input_samples - waveform.shape[-1]))
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    if waveform.shape[-1] < n_fft:
        waveform = F.pad(waveform, (0, n_fft - waveform.shape[-1]))
    hop_length = int(audio_vae.config.mel_hop_length)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=target_rate / 2.0,
        n_mels=int(audio_vae.config.mel_bins),
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
        num_time_steps = mel.shape[2]
        max_start = num_time_steps - max_mel_steps
        scan_stride = max(1, max_mel_steps // 4)
        candidate_starts = list(range(0, max_start + 1, scan_stride))
        if candidate_starts[-1] != max_start:
            candidate_starts.append(max_start)

        response = mel.float().exp().sum(dim=(0, 1, 3))
        cumulative_response = torch.cat([response.new_zeros(1), response.cumsum(dim=0)])
        starts = torch.tensor(candidate_starts, device=mel.device, dtype=torch.long)
        scores = cumulative_response[starts + max_mel_steps] - cumulative_response[starts]
        start_mel_step = candidate_starts[int(scores.argmax().item())]
        start_sample = min(start_mel_step * hop_length, waveform.shape[-1] - max_samples)
        waveform = waveform[..., start_sample : start_sample + max_samples]
        mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5)).permute(0, 2, 1).unsqueeze(0)

    latents = audio_vae.encode(mel.to(audio_vae.dtype)).latent_dist.mode()
    latents = _pack_audio_latents(latents)
    return _normalize_audio_latents(latents, audio_latents_mean, audio_latents_std).float()


class EchoVaeEncoderStep(ModularPipelineBlocks):
    """Encode Echo's clean first frame and ordered image/audio memory slots."""

    model_name = "echo"

    @property
    def description(self) -> str:
        return "Encodes the optional first frame and image/audio memory slots into normalized VAE latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
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
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "first_frame_latents", type_hint=torch.Tensor, description="Normalized first-frame VAE latents."
            ),
            OutputParam(
                "memory_video_latents", type_hint=list, description="Normalized VAE latents for each memory image."
            ),
            OutputParam(
                "memory_audio_latents",
                type_hint=list,
                description="Normalized packed audio VAE latents for each memory slot.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        if block_state.image is not None:
            first_frame_latents = _encode_image(
                components.vae,
                components.latents_mean,
                components.latents_std,
                components.vae_scaling_factor,
                block_state.image,
                block_state.height,
                block_state.width,
                device,
            )
        else:
            first_frame_latents = None

        memory_images = _as_list(block_state.memory_images)
        slot_count = len(memory_images)
        _validate_memory_slot_count(slot_count)

        memory_video_latents = [
            _encode_image(
                components.vae,
                components.latents_mean,
                components.latents_std,
                components.vae_scaling_factor,
                image,
                block_state.height,
                block_state.width,
                device,
            )
            for image in memory_images
        ]

        raw_audio = _as_list(block_state.memory_audio_waveforms)
        if raw_audio and len(raw_audio) != slot_count:
            raise ValueError("`memory_audio_waveforms` must have one entry per `memory_images` slot.")

        memory_audio_latents: list[torch.Tensor | None] = []
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
                memory_audio_latents.append(
                    None
                    if waveform is None
                    else _encode_audio(
                        components.audio_vae,
                        components.audio_latents_mean,
                        components.audio_latents_std,
                        waveform,
                        int(sample_rate),
                        device,
                    )
                )

        block_state.first_frame_latents = first_frame_latents
        block_state.memory_video_latents = memory_video_latents or None
        block_state.memory_audio_latents = memory_audio_latents or None

        self.set_block_state(state, block_state)
        return components, state
