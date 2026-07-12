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

* **``guidance_scale`` defaults to 1.0 (off)** — SA3 Medium is adversarially distilled; CFG is baked into the model
  weights, so classifier-free guidance is unnecessary (and adds compute) for the distilled checkpoint. It is meaningful
  for the non-distilled ``stable-audio-3-medium-base`` checkpoint.
* **``duration`` in seconds** (single float) replaces the ``audio_start_in_s`` / ``audio_end_in_s`` pair.
* **No ``projection_model``** — T5Gemma output (768 d) feeds cross-attention directly; duration is embedded by
  ``StableAudio3DurationEmbedder``.
* **PingPongScheduler** (N=8 stochastic re-noise steps) is the production scheduler, but any compatible scheduler can
  be substituted.
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
from ...utils import is_torch_xla_available, logging, replace_example_docstring
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
        ...     generator=generator,
        ... ).audios

        >>> sf.write("output.wav", audio[0].T.cpu().float().numpy(), samplerate=44100)
        ```
"""


class StableAudio3Pipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using Stable Audio 3.

    SA3 uses a distilled rectified-flow DiT with ping-pong sampling. Classifier-free guidance (``guidance_scale`` /
    ``negative_prompt``) is unnecessary for the distilled checkpoint (leave ``guidance_scale=1.0``, the default) but is
    meaningful for the non-distilled ``stable-audio-3-medium-base`` checkpoint.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines.

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
    """

    model_cpu_offload_seq = "text_encoder->duration_embedder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

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
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

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

        # Tile for num_waveforms_per_prompt
        bs, seq_len, hidden_size = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs * num_waveforms_per_prompt, seq_len, hidden_size)
        encoder_attention_mask = encoder_attention_mask.repeat_interleave(num_waveforms_per_prompt, dim=0)

        return prompt_embeds, encoder_attention_mask

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
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
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
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot provide both `negative_prompt` and `negative_prompt_embeds`. Use one of the two.")
        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(f"`negative_prompt` must be `str` or `list[str]`, got {type(negative_prompt)}.")
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        duration: float = 10.0,
        num_inference_steps: Optional[int] = None,
        silence_padding_duration: float = 0.0,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_encoder_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], dict]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        output_type: str = "pt",
    ) -> Union[AudioPipelineOutput, tuple]:
        r"""
        Generate audio from a text prompt.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompt(s). Pass ``prompt_embeds`` instead to skip tokenization and encoding.
            duration (`float`, defaults to 10.0):
                Requested output duration in seconds.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. When ``None`` (default), the step count is taken from the checkpoint's
                scheduler config: **8** for the distilled model (`PingPongScheduler`) and **100** for the base model
                (`StableAudio3EulerScheduler`). Pass an explicit value to override.
            silence_padding_duration (`float`, defaults to 0.0):
                Extra seconds of latent context generated beyond the target content, giving the model headroom at the
                boundary; the output is trimmed back to `duration`. Defaults to 0.0 (disabled). Increase only if the
                model is trained/distilled to mask this padding — otherwise the extra frames drain output energy.
            guidance_scale (`float`, defaults to 1.0):
                Classifier-free guidance scale. ``1.0`` disables guidance (the default, and the only sensible value for
                the distilled SA3 Medium checkpoint, whose CFG is baked into the weights). Values ``> 1.0`` are
                meaningful for the non-distilled ``stable-audio-3-medium-base`` checkpoint; higher values follow the
                prompt more closely at the cost of diversity.
            negative_prompt (`str` or `list[str]`, *optional*):
                Prompt(s) describing what to steer away from when ``guidance_scale > 1.0``. Defaults to an empty string
                (unconditional) when ``guidance_scale > 1.0`` and neither this nor `negative_prompt_embeds` is given.
                Ignored when ``guidance_scale <= 1.0``.
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
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed negative text embeddings, as an alternative to `negative_prompt`.
            negative_encoder_attention_mask (`torch.LongTensor`, *optional*):
                Boolean mask for pre-computed negative embeddings.
            return_dict (`bool`, defaults to `True`):
                Return an `AudioPipelineOutput` or a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                Called at the end of each denoising step with `(self, step_idx, timestep, callback_kwargs)`, where
                `callback_kwargs` contains the tensors listed in `callback_on_step_end_tensor_inputs`. Must return a
                dict with the (optionally modified) tensors to use for the rest of the loop.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                The tensors passed to `callback_on_step_end`. Must be a subset of `self._callback_tensor_inputs`.
            output_type (`str`, defaults to ``"pt"``):
                ``"pt"`` for a PyTorch tensor, ``"np"`` for a NumPy array, or ``"latent"`` to skip decoding and return
                the raw latents.

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`:
                ``.audios`` is a tensor / array of shape ``(batch * num_waveforms_per_prompt, audio_channels,
                samples)``.

        Examples:
        """
        # 0. Validate
        self.check_inputs(
            prompt,
            duration,
            prompt_embeds,
            encoder_attention_mask,
            negative_prompt,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale

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

        # 2. Encode duration for AdaLN/global conditioning; shared by all branches.
        global_hidden_states = self.encode_duration(duration, device, num_waveforms_per_prompt, batch_size)
        prompt_embeds, encoder_attention_mask = self.prepare_cross_attention(
            prompt_embeds, encoder_attention_mask, global_hidden_states
        )

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt] * batch_size
                elif negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                else:
                    uncond_tokens = negative_prompt
                negative_prompt_embeds, negative_encoder_attention_mask = self.encode_prompt(
                    uncond_tokens, device, num_waveforms_per_prompt, None, None
                )
            else:
                negative_prompt_embeds, negative_encoder_attention_mask = self.encode_prompt(
                    None, device, num_waveforms_per_prompt, negative_prompt_embeds, negative_encoder_attention_mask
                )
            negative_prompt_embeds, negative_encoder_attention_mask = self.prepare_cross_attention(
                negative_prompt_embeds, negative_encoder_attention_mask, global_hidden_states
            )
            model_global_hidden_states = torch.cat([global_hidden_states, global_hidden_states])
        else:
            model_global_hidden_states = global_hidden_states

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

        # 5. Set timesteps: (8 for the distilled PingPong model, 100 for the base Euler model)
        if num_inference_steps is None:
            num_inference_steps = getattr(self.scheduler.config, "num_inference_steps", None)
            if num_inference_steps is None:
                raise ValueError(
                    "`num_inference_steps` was not provided and the scheduler "
                    f"({self.scheduler.__class__.__name__}) does not define a default "
                    "`num_inference_steps` in its config. Pass `num_inference_steps` explicitly."
                )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Ping-pong denoising loop (CFG only if `guidance_scale > 1.0`; the distilled model bakes it in)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents])
                    model_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                    model_encoder_attention_mask = torch.cat([negative_encoder_attention_mask, encoder_attention_mask])
                else:
                    latent_model_input = latents
                    model_prompt_embeds = prompt_embeds
                    model_encoder_attention_mask = encoder_attention_mask
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict velocity v(x_t, t)
                velocity = self.transformer(
                    latent_model_input,
                    t.expand(latent_model_input.shape[0]),
                    encoder_hidden_states=model_prompt_embeds,
                    global_hidden_states=model_global_hidden_states,
                    encoder_attention_mask=model_encoder_attention_mask,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    velocity_uncond, velocity_text = velocity.chunk(2)
                    velocity = velocity_uncond + self.guidance_scale * (velocity_text - velocity_uncond)

                # x̂₀ = x_t − t·v  →  re-noise with fresh ε
                latents = self.scheduler.step(velocity, t, latents, generator=generator).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 7. Decode latents
        if output_type == "latent":
            audio = latents
        else:
            audio = self.vae.decode(latents).sample
            audio = audio[:, :, :waveform_length].clamp(-1.0, 1.0)

            if output_type == "np":
                audio = audio.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)
        return AudioPipelineOutput(audios=audio)
