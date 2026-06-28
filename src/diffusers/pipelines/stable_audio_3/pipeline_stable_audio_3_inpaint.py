# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
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

"""
Audio inpainting pipeline for Stable Audio 3.

Extends ``StableAudio3Pipeline`` with a local-additive conditioning path:

    ``local_add_cond = cat([mask, masked_latent], dim=1)`` shape: ``(batch, 1 + latent_dim, L)``

At each DiT block a small MLP (``to_local_embed``) projects this tensor along the channel dimension and adds it to the
per-frame hidden states before the self-attention operation. This pathway is implemented in [`StableAudio3DiTModel`]
(the ``local_add_cond`` forward argument), so the conditioning is active end-to-end.
"""

import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F

from ...utils import logging
from ..pipeline_utils import AudioPipelineOutput
from .pipeline_stable_audio_3 import StableAudio3Pipeline


logger = logging.get_logger(__name__)


class StableAudio3InpaintPipeline(StableAudio3Pipeline):
    r"""
    Audio inpainting pipeline for Stable Audio 3.

    Inherits all text-to-audio logic from [`StableAudio3Pipeline`]. When ``audio`` and ``mask`` are provided, encodes
    the reference audio with the frozen SAME encoder and injects ``masked_latent ∥ mask`` as local-additive
    conditioning into each DiT block via the transformer's ``local_add_cond`` pathway (``to_local_embed``).

    Args:
        Inherits all args from [`StableAudio3Pipeline`].

    Call signature extension (see :meth:`__call__`):
        audio (`torch.Tensor` of shape ``(batch, channels, samples)``):
            Reference audio waveform at ``vae.config.sampling_rate`` Hz.
        mask (`torch.Tensor` of shape ``(batch, 1, latent_length)``):
            Per-frame binary mask in latent space. ``1`` = preserve original audio; ``0`` = region to be inpainted.
    """

    def _encode_reference_audio(
        self,
        audio: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode a reference waveform to latent space.

        Args:
            audio: ``(batch, channels, samples)`` at ``vae.config.sampling_rate``.
            device: Target device.

        Returns:
            ``(batch, latent_dim, L)`` latent tensor.
        """
        audio = audio.to(device=device, dtype=next(self.vae.parameters()).dtype)
        return self.vae.encode(audio).latents

    def _build_local_add_cond(
        self,
        audio_latents: torch.Tensor,
        mask: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Build the local-additive conditioning tensor.

        Concatenates the binary mask with the masked latent along the channel axis to form ``(batch, 1 + latent_dim,
        L)`` — the expected shape for the DiT's ``local_add_cond`` argument.

        Args:
            audio_latents: ``(batch, latent_dim, L_ref)`` encoded reference.
            mask: ``(batch, 1, L_ref)`` per-frame mask (0/1).
            target_length: Required latent length ``L`` (may differ from
                ``L_ref`` due to rounding).

        Returns:
            ``(batch, 1 + latent_dim, L)`` tensor.
        """
        # Resize mask to target latent length via nearest-neighbor interpolation
        if mask.shape[-1] != target_length:
            mask = F.interpolate(mask.float(), size=target_length, mode="nearest")

        # Pad or crop reference audio latents to match target length
        L_ref = audio_latents.shape[-1]
        if L_ref < target_length:
            pad = audio_latents.new_zeros(*audio_latents.shape[:2], target_length - L_ref)
            audio_latents = torch.cat([audio_latents, pad], dim=-1)
        elif L_ref > target_length:
            audio_latents = audio_latents[:, :, :target_length]

        # Masked latent: zero out the inpaint region
        masked_latent = audio_latents * mask

        # Concat along channel dim: (batch, 1 + latent_dim, L)
        return torch.cat([mask, masked_latent], dim=1)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        duration: float = 10.0,
        audio: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mask_start_seconds: Optional[Union[float, List[float]]] = None,
        mask_end_seconds: Optional[Union[float, List[float]]] = None,
        num_inference_steps: int = 8,
        silence_padding_duration: float = 0.0,
        num_waveforms_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        output_type: str = "pt",
    ) -> Union[AudioPipelineOutput, tuple]:
        r"""
        Generate inpainted audio conditioned on a text prompt and reference.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompt(s).
            duration (`float`, defaults to 10.0):
                Output duration in seconds. Should match the reference audio.
            audio (`torch.Tensor`, *optional*):
                Reference waveform ``(batch, channels, samples)`` at ``vae.config.sampling_rate`` Hz. Required for
                inpainting.
            mask (`torch.Tensor`, *optional*):
                Per-frame latent-space mask ``(batch, 1, L)`` with 0 = inpaint region, 1 = preserve. Either ``mask`` or
                ``mask_start_seconds`` / ``mask_end_seconds`` must be provided.
            mask_start_seconds (`float` or `list[float]`, *optional*):
                Start time(s) of the inpaint region in seconds.
            mask_end_seconds (`float` or `list[float]`, *optional*):
                End time(s) of the inpaint region (must pair with ``mask_start_seconds``).
            num_inference_steps (`int`, defaults to 8):
                Number of denoising steps.
            silence_padding_duration (`float`, defaults to 0.0):
                Extra latent headroom after the target content.
            num_waveforms_per_prompt (`int`, defaults to 1):
                Waveforms per prompt.
            generator: RNG for reproducibility.
            latents: Pre-generated starting noise (``None`` → sample fresh).
            prompt_embeds: Pre-computed text embeddings.
            encoder_attention_mask: Mask for pre-computed embeddings.
            return_dict (`bool`, defaults to `True`):
                Return `AudioPipelineOutput` or tuple.
            callback: Called every ``callback_steps`` with
                ``(step_idx, timestep, latents)``.
            callback_steps (`int`, defaults to 1): Callback frequency.
            output_type (`str`, defaults to ``"pt"``): ``"pt"`` / ``"np"`` / ``"latent"``.

        Returns:
            [`~pipelines.AudioPipelineOutput`] with ``.audios``.
        """
        if audio is None and mask is None and mask_start_seconds is None:
            # Fall back to the base text-to-audio pipeline when no audio is supplied
            logger.warning("No `audio` or `mask` provided — falling back to unconditional text-to-audio generation.")
            return super().__call__(
                prompt=prompt,
                duration=duration,
                num_inference_steps=num_inference_steps,
                silence_padding_duration=silence_padding_duration,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                output_type=output_type,
            )

        # Validate inpaint inputs
        if audio is None:
            raise ValueError("`audio` (reference waveform) is required for inpainting.")
        if mask is None and (mask_start_seconds is None or mask_end_seconds is None):
            raise ValueError(
                "Provide either a pre-built `mask` tensor or both `mask_start_seconds` and `mask_end_seconds`."
            )

        # 0. Common setup (shared with base class)
        self.check_inputs(prompt, duration, prompt_embeds, encoder_attention_mask)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 1. Text encoding
        prompt_embeds, encoder_attention_mask = self.encode_prompt(
            prompt, device, num_waveforms_per_prompt, prompt_embeds, encoder_attention_mask
        )

        # 2. Duration encoding (global AdaLN cond + appended cross-attention token)
        global_hidden_states = self.encode_duration(duration, device, num_waveforms_per_prompt, batch_size)
        prompt_embeds, encoder_attention_mask = self.prepare_cross_attention(
            prompt_embeds, encoder_attention_mask, global_hidden_states
        )

        # 3. Latent dimensions
        sampling_rate: int = self.vae.config.sampling_rate
        downsampling_ratio: int = self.vae.downsampling_ratio
        latent_dim: int = self.vae.config.latent_dim

        total_audio_samples = (
            int(math.ceil((duration + silence_padding_duration) * sampling_rate / downsampling_ratio))
            * downsampling_ratio
        )
        latent_length = total_audio_samples // downsampling_ratio
        waveform_length = int(duration * sampling_rate)

        # 4. Build mask tensor in latent space
        if mask is None:
            starts = [mask_start_seconds] if isinstance(mask_start_seconds, float) else list(mask_start_seconds)
            ends = [mask_end_seconds] if isinstance(mask_end_seconds, float) else list(mask_end_seconds)
            if len(starts) != len(ends):
                raise ValueError("`mask_start_seconds` and `mask_end_seconds` must have the same length.")
            # Mask in latent frame space (1 = preserve, 0 = inpaint)
            mask_audio = torch.ones(batch_size, 1, latent_length, device=device)
            for start_s, end_s in zip(starts, ends):
                start_f = int(start_s * sampling_rate / downsampling_ratio)
                end_f = min(int(end_s * sampling_rate / downsampling_ratio), latent_length)
                mask_audio[:, :, start_f:end_f] = 0.0
        else:
            mask_audio = mask.to(device=device, dtype=torch.float32)

        # 5. Encode reference audio
        audio_latents = self._encode_reference_audio(audio, device)

        # 6. Build local-additive conditioning: (batch, 1 + latent_dim, L)
        local_add_cond = self._build_local_add_cond(audio_latents, mask_audio, latent_length)
        local_add_cond = local_add_cond.to(prompt_embeds.dtype)
        # Tile for num_waveforms_per_prompt
        local_add_cond = local_add_cond.repeat_interleave(num_waveforms_per_prompt, dim=0)

        # 7. Starting latents: pure noise — frame preservation comes entirely from local_add_cond,
        #    not RePaint-style blending.
        noise_latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            latent_dim,
            latent_length,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 8. Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 9. Denoising loop with local-additive conditioning
        latents = noise_latents
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # local_add_cond is projected by to_local_embed in each DiT block.
                velocity = self.transformer(
                    latent_model_input,
                    t.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    global_hidden_states=global_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    local_add_cond=local_add_cond,
                    return_dict=False,
                )[0]

                latents = self.scheduler.step(velocity, t, latents, generator=generator).prev_sample

                if i == len(timesteps) - 1 or (i + 1) % callback_steps == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 10. Decode
        if output_type == "latent":
            return AudioPipelineOutput(audios=latents)

        audio_out = self.vae.decode(latents).sample
        audio_out = audio_out[:, :, :waveform_length].clamp(-1.0, 1.0)

        if output_type == "np":
            audio_out = audio_out.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio_out,)
        return AudioPipelineOutput(audios=audio_out)
