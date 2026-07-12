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
Audio-to-audio variation pipeline for Stable Audio 3.

Unlike [`StableAudio3InpaintPipeline`] (which preserves specific frames exactly via local-additive conditioning), this
pipeline noises the *entire* reference audio to a given ``init_noise_level`` and denoises it from there — matching
Stability's reference ``init_audio`` / ``init_noise_level`` workflow. ``init_noise_level=1.0`` is equivalent to plain
text-to-audio generation (the reference contributes nothing); lower values retain progressively more of the reference's
structure.
"""

import math
from typing import Callable, List, Optional, Union

import torch
from transformers import GemmaTokenizer, GemmaTokenizerFast, T5GemmaEncoderModel

from ...models.autoencoders.autoencoder_same import AutoencoderSAME
from ...models.transformers.transformer_stable_audio3 import StableAudio3DiTModel
from ...schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ...schedulers.scheduling_ping_pong import PingPongScheduler
from ...schedulers.scheduling_stable_audio3_euler import StableAudio3EulerScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_stable_audio_3 import StableAudio3DurationEmbedder


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import soundfile as sf
        >>> import torchaudio
        >>> from diffusers import StableAudio3AudioToAudioPipeline, PingPongScheduler

        >>> pipe = StableAudio3AudioToAudioPipeline.from_pretrained(
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
        ...     init_noise_level=0.6,
        ...     generator=generator,
        ... ).audios

        >>> sf.write("output.wav", audio[0].T.cpu().float().numpy(), samplerate=44100)
        ```
"""


class StableAudio3AudioToAudioPipeline(DiffusionPipeline):
    r"""
    Audio-to-audio variation pipeline for Stable Audio 3.

    Shares its text-to-audio logic with [`StableAudio3Pipeline`] (kept in sync via `# Copied from`). Encodes the
    reference audio with the frozen SAME encoder, mixes it with fresh noise according to ``init_noise_level``, and
    denoises from there — the whole signal is noised/denoised globally, unlike [`StableAudio3InpaintPipeline`]'s
    per-frame local-additive conditioning.

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
        scheduler ([`PingPongScheduler`] or compatible):
            Scheduler for the iterative denoising loop. The production SA3 Medium model is distilled for exactly 8
            ping-pong steps.

    Call signature extension (see :meth:`__call__`):
        audio (`torch.Tensor` of shape ``(batch, channels, samples)``):
            Reference audio waveform at ``vae.config.sampling_rate`` Hz.
        init_noise_level (`float`):
            How much noise to mix into the reference before denoising. ``1.0`` = full noise (equivalent to
            text-to-audio); lower values retain more of the reference.
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
        scheduler: Union[PingPongScheduler, StableAudio3EulerScheduler, FlowMatchEulerDiscreteScheduler],
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
            latents = latents.to(device)
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
    # Audio-to-audio-specific helpers

    def _encode_reference_audio(
        self,
        audio: torch.Tensor,
        device: torch.device,
        target_length: int,
    ) -> torch.Tensor:
        """
        Encode a reference waveform to latent space and pad/crop it to `target_length`.

        Args:
            audio: ``(batch, channels, samples)`` at ``vae.config.sampling_rate``.
            device: Target device.
            target_length: Required latent length ``L`` (may differ from the reference's own encoded length due to
                rounding or a `duration` that doesn't match the reference's length).

        Returns:
            ``(batch, latent_dim, L)`` latent tensor.
        """
        audio = audio.to(device=device, dtype=next(self.vae.parameters()).dtype)
        audio_latents = self.vae.encode(audio).latents

        L_ref = audio_latents.shape[-1]
        if L_ref < target_length:
            pad = audio_latents.new_zeros(*audio_latents.shape[:2], target_length - L_ref)
            audio_latents = torch.cat([audio_latents, pad], dim=-1)
        elif L_ref > target_length:
            audio_latents = audio_latents[:, :, :target_length]
        return audio_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        duration: float = 10.0,
        audio: Optional[torch.Tensor] = None,
        init_noise_level: float = 1.0,
        num_inference_steps: Optional[int] = None,
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
        Generate an audio variation conditioned on a text prompt and a reference waveform.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompt(s).
            duration (`float`, defaults to 10.0):
                Output duration in seconds. Should match the reference audio.
            audio (`torch.Tensor`, *optional*):
                Reference waveform ``(batch, channels, samples)`` at ``vae.config.sampling_rate`` Hz. Required.
            init_noise_level (`float`, defaults to 1.0):
                Noise level (in ``(0, 1]``) mixed into the reference before denoising: ``x_start = (1 -
                init_noise_level) * reference_latents + init_noise_level * noise``. ``1.0`` discards the reference
                entirely (equivalent to [`StableAudio3Pipeline`]); lower values retain progressively more of the
                reference's structure while still running the full step count.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. When ``None`` (default), the step count is taken from the checkpoint's
                scheduler config, matching [`StableAudio3Pipeline`].
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
        # Validate audio-to-audio inputs
        if audio is None:
            raise ValueError(
                "`audio` (reference waveform) is required. Use `StableAudio3Pipeline` for plain text-to-audio "
                "generation."
            )
        if not (0.0 < init_noise_level <= 1.0):
            raise ValueError(f"`init_noise_level` must be in (0, 1], got {init_noise_level}.")

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

        # 4. Encode reference audio and pad/crop to the target latent length
        audio_latents = self._encode_reference_audio(audio, device, latent_length)
        audio_latents = audio_latents.to(prompt_embeds.dtype)
        audio_latents = audio_latents.repeat_interleave(num_waveforms_per_prompt, dim=0)

        # 5. Mix reference latents with fresh noise: x_start = (1 - init_noise_level) * ref + init_noise_level * noise
        noise = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            latent_dim,
            latent_length,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        init_noise_level_tensor = torch.full(
            (batch_size * num_waveforms_per_prompt,), init_noise_level, dtype=prompt_embeds.dtype, device=device
        )
        latents = self.scheduler.add_noise(audio_latents, noise, init_noise_level_tensor)

        # 6. Timesteps: compress the schedule to [init_noise_level, 0] while keeping the full step count
        if num_inference_steps is None:
            num_inference_steps = getattr(self.scheduler.config, "num_inference_steps", None)
            if num_inference_steps is None:
                raise ValueError(
                    "`num_inference_steps` was not provided and the scheduler "
                    f"({self.scheduler.__class__.__name__}) does not define a default "
                    "`num_inference_steps` in its config. Pass `num_inference_steps` explicitly."
                )
        self.scheduler.set_timesteps(num_inference_steps, device=device, sigma_max=init_noise_level)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                velocity = self.transformer(
                    latent_model_input,
                    t.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    global_hidden_states=global_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
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

        # 8. Decode
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
