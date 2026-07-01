# Copyright 2026 JD.com and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Callable

import torch
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor, GemmaTokenizer, GemmaTokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import FromSingleFileMixin, LTX2LoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...models.transformers import LTX2VideoTransformer3DModel
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..ltx2.connectors import LTX2TextConnectors
from ..ltx2.pipeline_ltx2 import LTX2Pipeline
from ..ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
from .pipeline_output import JoyAIEchoPipelineOutput, JoyAIEchoShotOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class JoyAIEchoMemorySlot:
    r"""
    A paired audio-video memory slot used by [`JoyAIEchoPipeline`].

    Args:
        latents (`torch.Tensor`):
            Packed video latent tokens of shape `(batch_size, sequence_length, channels)`.
        video_coords (`torch.Tensor`):
            Video positional coordinates of shape `(batch_size, 3, sequence_length, 2)`.
        audio_latents (`torch.Tensor`, *optional*):
            Packed audio latent tokens of shape `(batch_size, sequence_length, channels)`.
        audio_coords (`torch.Tensor`, *optional*):
            Audio positional coordinates of shape `(batch_size, 1, sequence_length, 2)`.
    """

    latents: torch.Tensor
    video_coords: torch.Tensor
    audio_latents: torch.Tensor | None = None
    audio_coords: torch.Tensor | None = None


class JoyAIEchoMemoryBank:
    r"""
    FIFO paired audio-video memory bank for JoyAI-Echo multi-shot generation.

    The official JoyAI-Echo inference script stores selected frames and audio windows. In diffusers we keep the already
    packed latent tokens, which avoids an additional VAE encode pass between shots and matches the in-context token
    interface used by existing LTX-2 pipelines.
    """

    def __init__(self, max_size: int = 7):
        self.max_size = int(max_size)
        self.slots: list[JoyAIEchoMemorySlot] = []

    def __len__(self) -> int:
        return len(self.slots)

    def append(self, slot: JoyAIEchoMemorySlot) -> None:
        if self.max_size <= 0:
            return
        self.slots.append(slot)
        if len(self.slots) > self.max_size:
            self.slots = self.slots[-self.max_size :]

    def get_video_memory(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor] | None:
        if len(self.slots) == 0:
            return None
        latents = torch.cat([slot.latents.to(device=device, dtype=dtype) for slot in self.slots], dim=1)
        coords = torch.cat([slot.video_coords.to(device=device) for slot in self.slots], dim=2)
        return latents, coords

    def get_audio_memory(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor] | None:
        audio_slots = [slot for slot in self.slots if slot.audio_latents is not None and slot.audio_coords is not None]
        if len(audio_slots) == 0:
            return None
        latents = torch.cat([slot.audio_latents.to(device=device, dtype=dtype) for slot in audio_slots], dim=1)
        coords = torch.cat([slot.audio_coords.to(device=device) for slot in audio_slots], dim=2)
        return latents, coords


def _as_prompt_list(prompt: str | list[str]) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]
    return prompt


def _select_memory_video_tokens(
    latents: torch.Tensor,
    video_coords: torch.Tensor,
    latent_num_frames: int,
    frame_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_per_frame = latents.shape[1] // int(latent_num_frames)
    if tokens_per_frame <= 0:
        raise ValueError("Cannot select JoyAI-Echo memory video tokens from an empty latent sequence.")
    frame_index = int(latent_num_frames) // 2 if frame_index is None else int(frame_index)
    frame_index = max(0, min(frame_index, int(latent_num_frames) - 1))
    start = frame_index * tokens_per_frame
    end = start + tokens_per_frame
    return latents[:, start:end].contiguous(), video_coords[:, :, start:end].contiguous()


def _select_memory_audio_tokens(
    audio_latents: torch.Tensor,
    audio_coords: torch.Tensor,
    window_size: int = 96,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_frames = audio_latents.shape[1]
    window_len = min(int(total_frames), max(1, int(window_size)))
    start = max((int(total_frames) - window_len) // 2, 0)
    end = start + window_len
    return audio_latents[:, start:end].contiguous(), audio_coords[:, :, start:end].contiguous()


class JoyAIEchoPipeline(LTX2Pipeline, FromSingleFileMixin, LTX2LoraLoaderMixin):
    r"""
    Pipeline for JoyAI-Echo text-to-audio-video multi-shot generation.

    JoyAI-Echo extends LTX-2 audio-video generation with few-step DMD denoising and a paired cross-shot memory bank.
    This pipeline keeps the diffusers LTX-2 component layout (`transformer`, `vae`, `audio_vae`, `vocoder`,
    `text_encoder`, `connectors`) and adds the JoyAI-Echo inference loop on top.

    Args:
        scheduler:
            Scheduler registered for compatibility with LTX-2 checkpoints. JoyAI-Echo's distilled inference uses the
            explicit `denoising_sigmas` schedule passed to `__call__`.
        vae ([`AutoencoderKLLTX2Video`]):
            Video VAE used to decode video latents.
        audio_vae ([`AutoencoderKLLTX2Audio`]):
            Audio VAE used to decode audio latents to mel spectrograms.
        text_encoder ([`Gemma3ForConditionalGeneration`]):
            Gemma text encoder.
        tokenizer ([`GemmaTokenizer`] or [`GemmaTokenizerFast`]):
            Gemma tokenizer.
        connectors ([`LTX2TextConnectors`]):
            Connector stack adapting Gemma hidden states to video and audio contexts.
        transformer ([`LTX2VideoTransformer3DModel`]):
            Distilled LTX-2 audio-video transformer.
        vocoder ([`LTX2Vocoder`] or [`LTX2VocoderWithBWE`]):
            Vocoder used to convert generated mel spectrograms to waveform.
        processor ([`Gemma3Processor`], *optional*):
            Optional Gemma processor.
    """

    model_cpu_offload_seq = "text_encoder->connectors->transformer->vae->audio_vae->vocoder"
    _optional_components = ["processor", "scheduler"]
    _callback_tensor_inputs = ["latents", "audio_latents", "prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Gemma3ForConditionalGeneration,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        connectors: LTX2TextConnectors,
        transformer: LTX2VideoTransformer3DModel,
        vocoder: LTX2Vocoder | LTX2VocoderWithBWE,
        processor: Gemma3Processor | None = None,
        scheduler=None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=transformer,
            vocoder=vocoder,
            processor=processor,
        )

    @staticmethod
    def _add_flow_noise(sample: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * sample + sigma * noise

    @staticmethod
    def _repeat_token_timestep(sigma: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if sigma.ndim == 0:
            sigma = sigma[None]
        if sigma.ndim == 1:
            return sigma[:, None].expand(-1, num_tokens).clone()
        return sigma

    @staticmethod
    def _build_video_memory_attention_mask(
        num_memory_tokens: int,
        num_target_tokens: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        total_tokens = num_memory_tokens + num_target_tokens
        attention_mask = torch.ones(batch_size, total_tokens, total_tokens, device=device, dtype=dtype)
        attention_mask[:, :, :num_memory_tokens] = 0
        attention_mask[:, :num_memory_tokens, :] = 0
        attention_mask[:, :num_memory_tokens, :num_memory_tokens] = 1
        return attention_mask

    def _prepare_prompt_context(
        self,
        prompt: str,
        device: torch.device,
        max_sequence_length: int,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask, _, _ = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=None,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        elif prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)

        tokenizer_padding_side = "left"
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")

        return self.connectors(prompt_embeds, prompt_attention_mask, padding_side=tokenizer_padding_side)

    def _get_execution_device(self) -> torch.device:
        try:
            return self._execution_device
        except AttributeError:
            pass

        for component in self.components.values():
            if not isinstance(component, torch.nn.Module):
                continue
            for tensor in component.parameters(recurse=True):
                return tensor.device
            for tensor in component.buffers(recurse=True):
                return tensor.device
        return torch.device("cpu")

    def _decode_latents(
        self,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        audio_num_frames: int,
        latent_mel_bins: int,
        output_type: str,
        decode_timestep: float = 0.0,
        decode_noise_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Any, Any]:
        device = latents.device
        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)

        if output_type == "latent":
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            return latents, audio_latents

        if not self.vae.config.timestep_conditioning:
            timestep = None
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timestep = torch.tensor([decode_timestep], device=device, dtype=latents.dtype)
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )
        video = self.vae.decode(latents.to(self.vae.dtype), timestep, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        mel_spectrograms = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0]
        audio = self.vocoder(mel_spectrograms)
        return video, audio

    def _denoise_shot(
        self,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        denoising_sigmas: torch.Tensor,
        prompt_embeds: torch.Tensor,
        audio_prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        audio_num_frames: int,
        frame_rate: float,
        memory_bank: JoyAIEchoMemoryBank,
        transformer_outputs_x0: bool,
        generator: torch.Generator | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[Any, int, torch.Tensor, dict], dict] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        memory_video = memory_bank.get_video_memory(device=device, dtype=dtype)
        memory_audio = memory_bank.get_audio_memory(device=device, dtype=dtype)
        memory_video_tokens = 0 if memory_video is None else memory_video[0].shape[1]
        memory_audio_tokens = 0 if memory_audio is None else memory_audio[0].shape[1]

        with self.progress_bar(total=max(len(denoising_sigmas) - 1, 0)) as progress_bar:
            for step_idx, sigma in enumerate(denoising_sigmas[:-1]):
                sigma = sigma.to(device=device, dtype=torch.float32)
                next_sigma = denoising_sigmas[step_idx + 1].to(device=device, dtype=torch.float32)

                video_model_input = latents
                audio_model_input = audio_latents
                video_model_coords = video_coords
                audio_model_coords = audio_coords
                video_attention_mask = None

                if memory_video is not None:
                    video_memory_latents, video_memory_coords = memory_video
                    video_model_input = torch.cat([video_memory_latents, latents], dim=1)
                    video_model_coords = torch.cat([video_memory_coords, video_coords], dim=2)

                if memory_audio is not None:
                    audio_memory_latents, audio_memory_coords = memory_audio
                    audio_model_input = torch.cat([audio_memory_latents, audio_latents], dim=1)
                    audio_model_coords = torch.cat([audio_memory_coords, audio_coords], dim=2)

                video_timestep = self._repeat_token_timestep(sigma.expand(batch_size), video_model_input.shape[1])
                audio_timestep = self._repeat_token_timestep(sigma.expand(batch_size), audio_model_input.shape[1])
                if memory_video_tokens > 0:
                    video_timestep[:, :memory_video_tokens] = 0
                if memory_audio_tokens > 0:
                    audio_timestep[:, :memory_audio_tokens] = 0

                pred_video, pred_audio = self.transformer(
                    hidden_states=video_model_input.to(dtype=prompt_embeds.dtype),
                    audio_hidden_states=audio_model_input.to(dtype=prompt_embeds.dtype),
                    encoder_hidden_states=prompt_embeds,
                    audio_encoder_hidden_states=audio_prompt_embeds,
                    timestep=video_timestep,
                    audio_timestep=audio_timestep,
                    sigma=sigma.expand(batch_size),
                    audio_sigma=sigma.expand(batch_size),
                    encoder_attention_mask=prompt_attention_mask,
                    audio_encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_model_coords,
                    audio_coords=audio_model_coords,
                    isolate_modalities=False,
                    spatio_temporal_guidance_blocks=None,
                    perturbation_mask=None,
                    use_cross_timestep=False,
                    attention_kwargs=attention_kwargs,
                    video_self_attention_mask=video_attention_mask,
                    return_dict=False,
                )
                pred_video = pred_video[:, memory_video_tokens:].float()
                pred_audio = pred_audio[:, memory_audio_tokens:].float()

                if not transformer_outputs_x0:
                    pred_video = latents.float() - pred_video * sigma
                    pred_audio = audio_latents.float() - pred_audio * sigma

                if next_sigma > 0:
                    video_noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                    audio_noise = randn_tensor(
                        audio_latents.shape, generator=generator, device=device, dtype=audio_latents.dtype
                    )
                    latents = self._add_flow_noise(pred_video, video_noise, next_sigma).to(dtype=dtype)
                    audio_latents = self._add_flow_noise(pred_audio, audio_noise, next_sigma).to(dtype=dtype)
                else:
                    latents = pred_video.to(dtype=dtype)
                    audio_latents = pred_audio.to(dtype=dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for name in callback_on_step_end_tensor_inputs or []:
                        callback_kwargs[name] = locals()[name]
                    callback_outputs = callback_on_step_end(self, step_idx, sigma, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    audio_latents = callback_outputs.pop("audio_latents", audio_latents)

                progress_bar.update()

        return latents, audio_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        height: int = 736,
        width: int = 1280,
        num_frames: int = 241,
        frame_rate: float = 25.0,
        denoising_sigmas: list[float] | torch.Tensor | None = None,
        memory_max_size: int = 7,
        generator: torch.Generator | None = None,
        prompt_embeds: list[torch.Tensor] | torch.Tensor | None = None,
        prompt_attention_mask: list[torch.Tensor] | torch.Tensor | None = None,
        output_type: str = "pil",
        return_latents: bool = False,
        return_dict: bool = True,
        decode_timestep: float = 0.0,
        decode_noise_scale: float | None = None,
        transformer_outputs_x0: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[Any, int, torch.Tensor, dict], dict] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents", "audio_latents"],
        max_sequence_length: int = 1024,
    ) -> JoyAIEchoPipelineOutput | tuple:
        r"""
        Generates one or more JoyAI-Echo shots.

        Args:
            prompt (`str` or `list[str]`):
                One prompt per shot. Passing a list enables cross-shot memory conditioning.
            height (`int`, *optional*, defaults to `736`):
                Generated video height.
            width (`int`, *optional*, defaults to `1280`):
                Generated video width.
            num_frames (`int`, *optional*, defaults to `241`):
                Number of video frames per shot.
            frame_rate (`float`, *optional*, defaults to `25.0`):
                Video frame rate.
            denoising_sigmas (`list[float]` or `torch.Tensor`, *optional*):
                JoyAI-Echo DMD sigma schedule. Defaults to the official inference schedule.
            memory_max_size (`int`, *optional*, defaults to `7`):
                Maximum number of previous shots kept as paired audio-video memory.
            transformer_outputs_x0 (`bool`, *optional*, defaults to `True`):
                Whether the transformer directly predicts `x0`. Set to `False` for velocity-prediction LTX-2
                transformers.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if denoising_sigmas is None:
            denoising_sigmas = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

        if isinstance(denoising_sigmas, torch.Tensor):
            denoising_sigmas = denoising_sigmas.detach().float()
        else:
            denoising_sigmas = torch.tensor(denoising_sigmas, dtype=torch.float32)
        if denoising_sigmas.ndim != 1 or denoising_sigmas.shape[0] < 2:
            raise ValueError("`denoising_sigmas` must be a 1D sequence with at least two values.")

        prompts = _as_prompt_list(prompt)
        device = self._get_execution_device()
        batch_size = 1

        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        num_channels_latents = self.transformer.config.in_channels

        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        num_mel_bins = self.audio_vae.config.mel_bins
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = self.audio_vae.config.latent_channels

        video_coords = self.transformer.rope.prepare_video_coords(
            batch_size, latent_num_frames, latent_height, latent_width, device, fps=frame_rate
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(batch_size, audio_num_frames, device)

        memory_bank = JoyAIEchoMemoryBank(max_size=memory_max_size)
        shots: list[JoyAIEchoShotOutput] = []

        for shot_idx, shot_prompt in enumerate(prompts):
            current_prompt_embeds = None
            current_prompt_attention_mask = None
            if isinstance(prompt_embeds, list):
                current_prompt_embeds = prompt_embeds[shot_idx]
            elif isinstance(prompt_embeds, torch.Tensor):
                current_prompt_embeds = prompt_embeds
            if isinstance(prompt_attention_mask, list):
                current_prompt_attention_mask = prompt_attention_mask[shot_idx]
            elif isinstance(prompt_attention_mask, torch.Tensor):
                current_prompt_attention_mask = prompt_attention_mask

            connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = (
                self._prepare_prompt_context(
                    shot_prompt,
                    device=device,
                    max_sequence_length=max_sequence_length,
                    prompt_embeds=current_prompt_embeds,
                    prompt_attention_mask=current_prompt_attention_mask,
                )
            )

            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                noise_scale=0.0,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            audio_latents = self.prepare_audio_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents_audio,
                audio_latent_length=audio_num_frames,
                num_mel_bins=num_mel_bins,
                noise_scale=0.0,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

            latents, audio_latents = self._denoise_shot(
                latents=latents,
                audio_latents=audio_latents,
                denoising_sigmas=denoising_sigmas.to(device),
                prompt_embeds=connector_prompt_embeds,
                audio_prompt_embeds=connector_audio_prompt_embeds,
                prompt_attention_mask=connector_attention_mask,
                video_coords=video_coords,
                audio_coords=audio_coords,
                latent_num_frames=latent_num_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                audio_num_frames=audio_num_frames,
                frame_rate=frame_rate,
                memory_bank=memory_bank,
                transformer_outputs_x0=transformer_outputs_x0,
                generator=generator,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            )

            memory_video_latents, memory_video_coords = _select_memory_video_tokens(
                latents.detach().cpu(),
                video_coords.detach().cpu(),
                latent_num_frames=latent_num_frames,
            )
            memory_audio_latents, memory_audio_coords = _select_memory_audio_tokens(
                audio_latents.detach().cpu(),
                audio_coords.detach().cpu(),
                window_size=96,
            )
            memory_bank.append(
                JoyAIEchoMemorySlot(
                    latents=memory_video_latents,
                    video_coords=memory_video_coords,
                    audio_latents=memory_audio_latents,
                    audio_coords=memory_audio_coords,
                )
            )

            frames, audio = self._decode_latents(
                latents,
                audio_latents,
                latent_num_frames=latent_num_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                audio_num_frames=audio_num_frames,
                latent_mel_bins=latent_mel_bins,
                output_type=output_type,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                generator=generator,
            )
            shots.append(
                JoyAIEchoShotOutput(
                    frames=frames,
                    audio=audio,
                    latents=latents if return_latents else None,
                    audio_latents=audio_latents if return_latents else None,
                )
            )

        self.maybe_free_model_hooks()

        frames = [shot.frames for shot in shots]
        audio = [shot.audio for shot in shots]
        if not return_dict:
            return frames, audio
        return JoyAIEchoPipelineOutput(frames=frames, audio=audio, shots=shots)
