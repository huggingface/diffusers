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

Adds a local-additive conditioning path on top of the text-to-audio logic shared with ``StableAudio3Pipeline``:

    ``local_add_cond = cat([mask, masked_latent], dim=1)`` shape: ``(batch, 1 + latent_dim, L)``

At each DiT block a small MLP (``to_local_embed``) projects this tensor along the channel dimension and adds it to the
per-frame hidden states before the self-attention operation. This pathway is implemented in [`StableAudio3DiTModel`]
(the ``local_add_cond`` forward argument), so the conditioning is active end-to-end.
"""

import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import GemmaTokenizer, GemmaTokenizerFast, T5GemmaEncoderModel

from ...models.autoencoders.autoencoder_same import AutoencoderSAME
from ...models.transformers.transformer_stable_audio3 import StableAudio3DiTModel
from ...schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_stable_audio_3 import StableAudio3DurationEmbedder


logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`, *optional*):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `sigmas` must
            be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.logsnr_sigma_schedule
def logsnr_sigma_schedule(
    num_inference_steps: int,
    logsnr_min: float,
    logsnr_max: float,
    sigma_max: float = 1.0,
) -> List[float]:
    """
    Build the log-SNR-warped sigma schedule used by Stable Audio 3's rectified-flow sampling.

    A linear grid ``t`` of ``num_inference_steps + 1`` breakpoints over ``[sigma_max, 0]`` is warped through the
    log-SNR affine map ``logsnr = logsnr_max − t · (logsnr_max − logsnr_min)`` and converted to the flow-matching sigma
    variable via ``sigma = sigmoid(-logsnr)``. The start is then forced to exactly ``sigma_max`` — required because
    that's also the exact noise level the starting latents were mixed at, whereas the natural ``sigmoid(-logsnr)``
    value would be off by a small amount — before the terminal breakpoint is dropped, leaving ``num_inference_steps``
    sigmas to pass to `FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...)`. The dropped terminal breakpoint
    doesn't need to be forced to 0 itself: the scheduler unconditionally appends its own terminal zero sigma, which is
    what actually determines the final denoising step's target.

    When ``sigma_max=1.0`` this is equivalent to placing the breakpoints uniformly in log-SNR space over ``[logsnr_min,
    logsnr_max]``. For ``sigma_max < 1.0`` (audio-to-audio variation via ``init_noise_level``), the schedule is *not*
    uniform in log-SNR space, and its first two breakpoints are not guaranteed to be monotonic — both are properties of
    the reference implementation, reproduced here for parity.

    Args:
        num_inference_steps (`int`): Number of denoising steps.
        logsnr_min (`float`): Minimum log-SNR value — maps to the high-noise end of the schedule (``t=sigma_max``).
        logsnr_max (`float`): Maximum log-SNR value — maps to the low-noise end of the schedule (``t=0``).
        sigma_max (`float`, defaults to 1.0):
            Starting noise level. ``1.0`` for full generation; ``< 1.0`` for audio-to-audio variation
            (``init_noise_level``), where the starting latents are already partially denoised.

    Returns:
        `List[float]`: `num_inference_steps` sigma values, starting at exactly `sigma_max` and decreasing from
            there. The scheduler appends its own terminal 0 after this list, so the full schedule used at inference
            time ends at exactly 0 even though the last value here does not.
    """
    t = torch.linspace(sigma_max, 0.0, num_inference_steps + 1)
    logsnr = logsnr_max - t * (logsnr_max - logsnr_min)
    sigmas = torch.sigmoid(-logsnr)  # (N+1,)
    sigmas[0] = sigma_max
    return sigmas[:-1].tolist()


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import soundfile as sf
        >>> import torchaudio
        >>> from diffusers import StableAudio3InpaintPipeline

        >>> pipe = StableAudio3InpaintPipeline.from_pretrained(
        ...     "stabilityai/stable-audio-3-medium", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> audio, sr = torchaudio.load("reference.wav")
        >>> audio = torchaudio.functional.resample(audio, sr, pipe.vae.config.sampling_rate).unsqueeze(0).to("cuda")

        >>> generator = torch.Generator("cuda").manual_seed(0)
        >>> audio = pipe(
        ...     "A gentle piano melody with soft strings in a concert hall",
        ...     duration=10.0,
        ...     audio=audio,
        ...     mask_start_seconds=4.0,
        ...     mask_end_seconds=6.0,
        ...     generator=generator,
        ... ).audios

        >>> sf.write("output.wav", audio[0].T.cpu().float().numpy(), samplerate=pipe.vae.config.sampling_rate)
        ```
"""


class StableAudio3InpaintPipeline(DiffusionPipeline):
    r"""
    Audio inpainting pipeline for Stable Audio 3.

    Shares its text-to-audio logic with [`StableAudio3Pipeline`] (kept in sync via `# Copied from`). When ``audio`` and
    ``mask`` are provided, encodes the reference audio with the frozen SAME encoder and injects ``masked_latent ∥
    mask`` as local-additive conditioning into each DiT block via the transformer's ``local_add_cond`` pathway
    (``to_local_embed``).

    Args:
        vae ([`AutoencoderSAME`]):
            SAME autoencoder used to encode and decode audio latents.
        text_encoder ([`~transformers.T5GemmaEncoderModel`]):
            Frozen T5Gemma text encoder (``google/t5gemma-b-b-ul2``).
        tokenizer ([`~transformers.GemmaTokenizerFast`]):
            Tokenizer for the text encoder.
        duration_embedder ([`StableAudio3DurationEmbedder`]):
            Maps ``duration`` in seconds to a global conditioning vector for AdaLN in each DiT block.
        transformer ([`StableAudio3DiTModel`]):
            The rectified-flow velocity-prediction DiT.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler for the iterative denoising loop. The production (distilled) SA3 Medium checkpoint uses
            `stochastic_sampling=True` for exactly 8 ping-pong steps; the non-distilled base checkpoint uses
            `stochastic_sampling=False` for ~100 deterministic Euler steps.

    Call signature extension (see :meth:`__call__`):
        audio (`torch.Tensor` of shape ``(batch, channels, samples)``):
            Reference audio waveform at ``vae.config.sampling_rate`` Hz.
        mask (`torch.Tensor` of shape ``(batch, 1, latent_length)``):
            Per-frame binary mask in latent space. ``1`` = preserve original audio; ``0`` = region to be inpainted.
    """

    model_cpu_offload_seq = "text_encoder->duration_embedder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    # Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.StableAudio3Pipeline.__init__
    def __init__(
        self,
        vae: AutoencoderSAME,
        text_encoder: T5GemmaEncoderModel,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        duration_embedder: StableAudio3DurationEmbedder,
        transformer: StableAudio3DiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            duration_embedder=duration_embedder,
            transformer=transformer,
            scheduler=scheduler,
        )

    @property
    # Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.StableAudio3Pipeline.num_timesteps
    def num_timesteps(self):
        return self._num_timesteps

    # ------------------------------------------------------------------
    # Encoding helpers

    # Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.StableAudio3Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]],
        device: torch.device,
        num_waveforms_per_prompt: int,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Encode text prompt(s) into cross-attention conditioning tensors.

        Args:
            prompt: Text prompt or list of prompts.  Ignored when
                ``prompt_embeds`` is provided.
            device: Target device.
            num_waveforms_per_prompt: How many output waveforms to generate
                per prompt; conditioning tensors are tiled accordingly.
            prompt_embeds: Pre-computed text embeddings
                ``(batch, seq_len, hidden_size)``.
            encoder_attention_mask: Boolean mask ``(batch, seq_len)`` for
                pre-computed embeddings; ``1`` = real token, ``0`` = pad.

        Returns:
            ``(prompt_embeds, encoder_attention_mask)`` both tiled to ``batch * num_waveforms_per_prompt``.
        """
        # SA3 uses T5Gemma with max_length=256 (from the reference checkpoint model_config.json).
        # Cap at 256 regardless of the tokenizer's own model_max_length.
        _max_length = min(getattr(self.tokenizer, "model_max_length", 256), 256)

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            encoder_attention_mask = text_inputs.attention_mask.to(device)

            # Warn on token truncation
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, _max_length - 1 : -1])
                logger.warning(
                    f"The following part of your input was truncated because {self.text_encoder.config.model_type} can "
                    f"only handle sequences up to {_max_length} tokens: {removed_text}"
                )

            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(text_input_ids, attention_mask=encoder_attention_mask)[0]

        if encoder_attention_mask is None:
            # All-ones mask for pre-computed embeddings with no mask provided
            encoder_attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)

        # Tile for num_waveforms_per_prompt
        bs, seq_len, hidden_size = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs * num_waveforms_per_prompt, seq_len, hidden_size)
        encoder_attention_mask = encoder_attention_mask.repeat_interleave(num_waveforms_per_prompt, dim=0)

        return prompt_embeds, encoder_attention_mask

    # Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.StableAudio3Pipeline.encode_duration
    def encode_duration(
        self,
        duration: float,
        device: torch.device,
        num_waveforms_per_prompt: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Embed the duration value into the global conditioning vector.

        Args:
            duration: Duration in seconds, applied to every sample in the batch.
            device: Target device.
            num_waveforms_per_prompt: Tile factor.
            batch_size: Number of prompts.

        Returns:
            ``(batch * num_waveforms_per_prompt, output_dim)`` tensor.
        """
        duration_tensor = torch.tensor([float(duration)] * batch_size, dtype=torch.float32, device=device)
        global_hidden_states = self.duration_embedder(duration_tensor)  # (batch, output_dim)
        global_hidden_states = global_hidden_states.repeat_interleave(num_waveforms_per_prompt, dim=0)
        return global_hidden_states

    # Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.StableAudio3Pipeline.prepare_cross_attention
    def prepare_cross_attention(
        self,
        prompt_embeds: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        global_hidden_states: torch.Tensor,
    ) -> tuple:
        """
        Build the cross-attention context by appending the duration embedding as an extra token.

        SA3 routes the ``seconds_total`` conditioner to *both* the global (AdaLN) input and the cross-attention context
        (``cross_attention_cond_ids = ["prompt", "seconds_total"]``). The duration embedding is concatenated after the
        text tokens, and the attention mask is extended with one valid entry.

        Returns:
            ``(context, context_mask)`` of shapes ``(batch, T_text + 1, dim)`` and ``(batch, T_text + 1)``.
        """
        duration_token = global_hidden_states.unsqueeze(1).to(prompt_embeds.dtype)
        context = torch.cat([prompt_embeds, duration_token], dim=1)
        duration_mask = encoder_attention_mask.new_ones(encoder_attention_mask.shape[0], 1)
        context_mask = torch.cat([encoder_attention_mask, duration_mask], dim=1)
        return context, context_mask

    # ------------------------------------------------------------------

    # Copied from diffusers.pipelines.stable_audio_3.pipeline_stable_audio_3.StableAudio3Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        latent_dim: int,
        latent_length: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (batch_size, latent_dim, latent_length)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents

    def check_inputs(
        self,
        prompt: Optional[Union[str, List[str]]],
        duration: float,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ) -> None:
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds`. Use one of the two.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` must be `str` or `list[str]`, got {type(prompt)}.")
        if duration <= 0:
            raise ValueError(f"`duration` must be positive, got {duration}.")
        if (
            prompt_embeds is not None
            and encoder_attention_mask is not None
            and prompt_embeds.shape[:2] != encoder_attention_mask.shape
        ):
            raise ValueError(
                f"`encoder_attention_mask` shape {encoder_attention_mask.shape} must match "
                f"`prompt_embeds` batch and sequence dimensions {prompt_embeds.shape[:2]}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    # ------------------------------------------------------------------
    # Inpaint-specific helpers

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
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        duration: float = 10.0,
        audio: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mask_start_seconds: Optional[Union[float, List[float]]] = None,
        mask_end_seconds: Optional[Union[float, List[float]]] = None,
        num_inference_steps: Optional[int] = None,
        logsnr_min: float = -6.2,
        logsnr_max: float = 2.0,
        silence_padding_duration: float = 0.0,
        num_waveforms_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], dict]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
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
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. When ``None`` (default), the step count is chosen from the scheduler's
                `stochastic_sampling` config, matching [`StableAudio3Pipeline`].
            logsnr_min (`float`, defaults to -6.2):
                Minimum log-SNR value for the noise schedule — maps to the high-noise start of the schedule.
            logsnr_max (`float`, defaults to 2.0):
                Maximum log-SNR value for the noise schedule — maps to the low-noise end of the schedule.
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
            callback_on_step_end (`Callable`, *optional*):
                Called at the end of each denoising step with `(self, step_idx, timestep, callback_kwargs)`. Must
                return a dict with the (optionally modified) tensors to use for the rest of the loop.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                The tensors passed to `callback_on_step_end`. Must be a subset of `self._callback_tensor_inputs`.
            output_type (`str`, defaults to ``"pt"``): ``"pt"`` / ``"np"`` / ``"latent"``.

        Returns:
            [`~pipelines.AudioPipelineOutput`] with ``.audios``.

        Examples:
        """
        # Validate inpaint inputs
        if audio is None:
            raise ValueError(
                "`audio` (reference waveform) is required for inpainting. Use `StableAudio3Pipeline` for plain "
                "text-to-audio generation."
            )
        if mask is None and (mask_start_seconds is None or mask_end_seconds is None):
            raise ValueError(
                "Provide either a pre-built `mask` tensor or both `mask_start_seconds` and `mask_end_seconds`."
            )

        # 0. Common setup (shared with base class)
        self.check_inputs(prompt, duration, prompt_embeds, encoder_attention_mask, callback_on_step_end_tensor_inputs)

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
            starts = [mask_start_seconds] if isinstance(mask_start_seconds, (int, float)) else list(mask_start_seconds)
            ends = [mask_end_seconds] if isinstance(mask_end_seconds, (int, float)) else list(mask_end_seconds)
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

        # 8. Timesteps: fall back to the scheduler's `stochastic_sampling` config, matching StableAudio3Pipeline
        if num_inference_steps is None:
            num_inference_steps = 8 if self.scheduler.config.stochastic_sampling else 100
        sigmas = logsnr_sigma_schedule(num_inference_steps, logsnr_min, logsnr_max)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, sigmas=sigmas, device=device)
        self._num_timesteps = len(timesteps)

        # 9. Denoising loop with local-additive conditioning
        latents = noise_latents
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # local_add_cond is projected by to_local_embed in each DiT block.
                velocity = self.transformer(
                    latents,
                    t.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    global_hidden_states=global_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    local_add_cond=local_add_cond,
                    return_dict=False,
                )[0]

                latents = self.scheduler.step(velocity, t, latents, generator=generator).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        # 10. Decode
        if output_type == "latent":
            audio_out = latents
        else:
            audio_out = self.vae.decode(latents).sample
            audio_out = audio_out[:, :, :waveform_length].clamp(-1.0, 1.0)

            if output_type == "np":
                audio_out = audio_out.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio_out,)
        return AudioPipelineOutput(audios=audio_out)
