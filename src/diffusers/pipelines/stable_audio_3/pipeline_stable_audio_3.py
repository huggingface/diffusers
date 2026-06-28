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
Text-to-audio pipeline for Stable Audio 3 (distilled rectified flow).

Key differences from the original ``StableAudioPipeline``:

* **No ``guidance_scale``** — SA3 Medium is adversarially distilled; CFG is baked into the model weights.
* **``duration`` in seconds** (single float) replaces the ``audio_start_in_s`` / ``audio_end_in_s`` pair.
* **No ``projection_model``** — T5Gemma output (768 d) feeds cross-attention directly; duration is embedded by
  ``StableAudio3DurationEmbedder``.
* **PingPongScheduler** (N=8 stochastic re-noise steps) is the production scheduler, but any compatible scheduler can
  be substituted.
"""

import math
from typing import Callable, List, Optional, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast

from ...models.autoencoders.autoencoder_same import AutoencoderSAME
from ...models.transformers.transformer_stable_audio3 import StableAudio3DiTModel
from ...schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ...schedulers.scheduling_ping_pong import PingPongScheduler
from ...utils import is_torch_xla_available, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_stable_audio_3 import StableAudio3DurationEmbedder


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import soundfile as sf
        >>> from diffusers import StableAudio3Pipeline, PingPongScheduler

        >>> pipe = StableAudio3Pipeline.from_pretrained("stabilityai/stable-audio-3-medium", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.Generator("cuda").manual_seed(0)
        >>> audio = pipe(
        ...     "A gentle piano melody with soft strings in a concert hall",
        ...     duration=10.0,
        ...     num_inference_steps=8,
        ...     generator=generator,
        ... ).audios

        >>> sf.write("output.wav", audio[0].T.cpu().float().numpy(), samplerate=44100)
        ```
"""


class StableAudio3Pipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using Stable Audio 3.

    SA3 uses a distilled rectified-flow DiT with ping-pong sampling — no classifier-free guidance at inference.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines.

    Args:
        vae ([`AutoencoderSAME`]):
            SAME autoencoder used to encode and decode audio latents.
        text_encoder ([`~transformers.T5EncoderModel`]):
            Frozen T5Gemma text encoder (``google/t5gemma-b-b-ul2``).
        tokenizer ([`~transformers.T5TokenizerFast`]):
            Tokenizer for the text encoder.
        duration_embedder ([`StableAudio3DurationEmbedder`]):
            Maps ``duration`` in seconds to a global conditioning vector for AdaLN in each DiT block.
        transformer ([`StableAudio3DiTModel`]):
            The rectified-flow velocity-prediction DiT.
        scheduler ([`PingPongScheduler`] or compatible):
            Scheduler for the iterative denoising loop. The production SA3 Medium model is distilled for exactly 8
            ping-pong steps.
    """

    model_cpu_offload_seq = "text_encoder->duration_embedder->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderSAME,
        text_encoder: T5EncoderModel,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        duration_embedder: StableAudio3DurationEmbedder,
        transformer: StableAudio3DiTModel,
        scheduler: Union[PingPongScheduler, FlowMatchEulerDiscreteScheduler],
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

    # ------------------------------------------------------------------
    # Encoding helpers

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

        # Zero out padded positions.  The reference SA3 implementation
        # replaces these with a learned padding_embedding; zeroing is
        # equivalent to that embedding being zero (default initialization).
        prompt_embeds = prompt_embeds * encoder_attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)

        # Tile for num_waveforms_per_prompt
        bs, seq_len, hidden_size = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs * num_waveforms_per_prompt, seq_len, hidden_size)
        encoder_attention_mask = encoder_attention_mask.repeat_interleave(num_waveforms_per_prompt, dim=0)

        return prompt_embeds, encoder_attention_mask

    def encode_duration(
        self,
        duration: Union[float, List[float]],
        device: torch.device,
        num_waveforms_per_prompt: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Embed duration value(s) into the global conditioning vector.

        Args:
            duration: Duration in seconds, or list of per-sample durations.
            device: Target device.
            num_waveforms_per_prompt: Tile factor.
            batch_size: Number of prompts.

        Returns:
            ``(batch * num_waveforms_per_prompt, output_dim)`` tensor.
        """
        if isinstance(duration, (int, float)):
            duration = [float(duration)] * batch_size
        elif len(duration) == 1:
            duration = [float(duration[0])] * batch_size
        elif len(duration) != batch_size:
            raise ValueError(
                f"`duration` has {len(duration)} entries but batch_size is {batch_size}. "
                "Pass a single float, a list of length 1, or a list with one entry per prompt."
            )

        duration_tensor = torch.tensor(duration, dtype=torch.float32, device=device)
        global_hidden_states = self.duration_embedder(duration_tensor)  # (batch, output_dim)
        global_hidden_states = global_hidden_states.repeat_interleave(num_waveforms_per_prompt, dim=0)
        return global_hidden_states

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

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        duration: float = 10.0,
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
        Generate audio from a text prompt.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompt(s). Pass ``prompt_embeds`` instead to skip tokenization and encoding.
            duration (`float`, defaults to 10.0):
                Requested output duration in seconds.
            num_inference_steps (`int`, defaults to 8):
                Number of denoising steps. SA3 Medium is distilled for exactly 8 ping-pong steps.
            silence_padding_duration (`float`, defaults to 0.0):
                Extra seconds of latent context appended after the target content, to give the model headroom at the
                boundary. The reference implementation uses 6.0 s; set to 0.0 to skip.
            num_waveforms_per_prompt (`int`, defaults to 1):
                Number of waveforms to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                For deterministic generation and reproducible re-noise in the ping-pong loop.
            latents (`torch.Tensor`, *optional*):
                Pre-generated starting latents. If ``None`` a fresh Gaussian tensor is sampled.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings ``(batch, seq_len, 768)``.
            encoder_attention_mask (`torch.LongTensor`, *optional*):
                Boolean mask for pre-computed embeddings.
            return_dict (`bool`, defaults to `True`):
                Return an `AudioPipelineOutput` or a plain tuple.
            callback (`Callable`, *optional*):
                Called every ``callback_steps`` denoising steps with ``(step_idx, timestep, latents)``.
            callback_steps (`int`, defaults to 1):
                Frequency of ``callback`` calls.
            output_type (`str`, defaults to ``"pt"``):
                ``"pt"`` for a PyTorch tensor, ``"np"`` for a NumPy array, or ``"latent"`` to skip decoding and return
                the raw latents.

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`:
                ``.audios`` is a tensor / array of shape ``(batch * num_waveforms_per_prompt, audio_channels,
                samples)``.
        """
        # 0. Validate
        self.check_inputs(prompt, duration, prompt_embeds, encoder_attention_mask)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 1. Encode text
        prompt_embeds, encoder_attention_mask = self.encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            prompt_embeds,
            encoder_attention_mask,
        )

        # 2. Encode duration → global conditioning for AdaLN, and append it as a cross-attn token
        global_hidden_states = self.encode_duration(duration, device, num_waveforms_per_prompt, batch_size)
        prompt_embeds, encoder_attention_mask = self.prepare_cross_attention(
            prompt_embeds, encoder_attention_mask, global_hidden_states
        )

        # 3. Compute latent shape
        sampling_rate: int = self.vae.config.sampling_rate
        downsampling_ratio: int = self.vae.downsampling_ratio
        latent_dim: int = self.vae.config.latent_dim

        # Total audio samples rounded up to a downsampling-ratio boundary
        total_audio_samples = (
            int(math.ceil((duration + silence_padding_duration) * sampling_rate / downsampling_ratio))
            * downsampling_ratio
        )
        latent_length = total_audio_samples // downsampling_ratio

        # Exact output sample count for post-decode trimming
        waveform_length = int(duration * sampling_rate)

        # 4. Prepare noise latents
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            latent_dim,
            latent_length,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Ping-pong denoising loop  (no CFG — distillation baked in)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # Predict velocity v(x_t, t)
                velocity = self.transformer(
                    latent_model_input,
                    t.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    global_hidden_states=global_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # x̂₀ = x_t − t·v  →  re-noise with fresh ε
                latents = self.scheduler.step(velocity, t, latents, generator=generator).prev_sample

                if i == len(timesteps) - 1 or (i + 1) % callback_steps == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 7. Decode latents
        if output_type == "latent":
            return AudioPipelineOutput(audios=latents)

        audio = self.vae.decode(latents).sample
        # Trim to the exact requested duration
        audio = audio[:, :, :waveform_length].clamp(-1.0, 1.0)

        if output_type == "np":
            audio = audio.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)
        return AudioPipelineOutput(audios=audio)
