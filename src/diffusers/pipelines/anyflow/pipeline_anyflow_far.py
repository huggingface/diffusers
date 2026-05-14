# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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
#
# Adapted from diffusers.pipelines.wan.pipeline_wan.WanPipeline (v0.35.1) for FAR causal flow-map sampling.

import copy
import html
from typing import Any, Callable, Dict, List, Optional, Union

import regex as re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import WanLoraLoaderMixin
from ...models import AnyFlowFARTransformer3DModel, AutoencoderKLWan
from ...models.autoencoders.vae import DiagonalGaussianDistribution
from ...schedulers import FlowMapEulerDiscreteScheduler
from ...utils import is_ftfy_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import AnyFlowPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import numpy as np
        >>> import torch
        >>> from diffusers import AnyFlowFARPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> pipe = AnyFlowFARPipeline.from_pretrained(
        ...     "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
        ... ).to("cuda")

        >>> # Single-frame I2V: wrap the conditioning image as a (1, 3, 1, H, W) tensor in [0, 1].
        >>> first_frame = load_image("path/to/first_frame.png").resize((832, 480))
        >>> arr = np.asarray(first_frame).astype("float32") / 255.0
        >>> context = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to("cuda")

        >>> video = pipe(
        ...     prompt="a cat walks across a sunlit lawn",
        ...     context_sequence={"raw": context},
        ...     num_inference_steps=4,
        ...     num_frames=81,
        ... ).frames[0]
        >>> export_to_video(video, "anyflow_far.mp4", fps=16)
        ```
"""


# Copied from diffusers.pipelines.wan.pipeline_wan.basic_clean
def basic_clean(text):
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


# Copied from diffusers.pipelines.wan.pipeline_wan.whitespace_clean
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# Copied from diffusers.pipelines.wan.pipeline_wan.prompt_clean
def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class AnyFlowFARPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Causal (FAR-based) text-to-video / image-to-video / video-to-video pipeline for AnyFlow checkpoints.

    The pipeline drives a frame-level autoregressive sampling loop over chunks: each chunk is denoised with
    flow-map steps while attending only to past chunks via block-sparse causal attention, and intermediate
    KV cache is reused across chunks.

    The task mode (T2V / I2V / V2V) is selected by the ``context_sequence`` argument passed to ``__call__``:

    - ``context_sequence=None`` — pure text-to-video.
    - ``context_sequence={"raw": <video tensor of shape (B, C, T, H, W) with T = 4n + 1>}`` — pre-VAE
      conditioning frames; the pipeline VAE-encodes them. Pass a single-frame video for I2V or a multi-frame
      clip for V2V.
    - ``context_sequence={"latent": <latent tensor of shape (B, C, T_latent, H_latent, W_latent)>}`` —
      already-encoded latents (skips the VAE encode step).

    The FAR backbone is the causal Wan2.1 variant introduced by FAR (Gu et al., 2025; arXiv:2503.19325).
    Inference is plain Euler in mean-velocity form per chunk with no re-noising. Joint T2V / I2V / V2V is
    supported by a single distilled model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer from [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).
        text_encoder ([`UMT5EncoderModel`]):
            [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) text encoder.
        transformer ([`AnyFlowFARTransformer3DModel`]):
            FAR causal flow-map 3D Transformer.
        vae ([`AutoencoderKLWan`]):
            VAE that encodes/decodes videos to and from latent representations.
        scheduler ([`FlowMapEulerDiscreteScheduler`]):
            Flow-map sampler.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # Default chunk partition for the released NVIDIA AnyFlow-FAR checkpoints (81 frames at the diffusers
    # VAE temporal stride of 4 → 21 latent frames split into 1 + 3*6 + 2 = [1, 3, 3, 3, 3, 3, 3, 2]). Override
    # via the ``chunk_partition`` argument to ``__call__`` for other frame counts.
    default_chunk_partition: List[int] = [1, 3, 3, 3, 3, 3, 3, 2]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: AnyFlowFARTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMapEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"  # noqa: E501
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def vae_encode(self, context_sequence):
        # normalize: [0, 1] -> [-1, 1]
        context_sequence = context_sequence * 2 - 1
        context_sequence = self.encode_latents(
            context_sequence.to(dtype=self.vae.dtype, device=self._execution_device), sample=False
        )
        context_sequence = context_sequence.permute(0, 2, 1, 3, 4)
        return context_sequence

    def _normalize_latents(self, latents, latents_mean, latents_std):
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    @torch.no_grad()
    def encode_latents(self, videos, sample=True):
        videos = videos.permute(0, 2, 1, 3, 4)
        moments = self.vae._encode(videos)

        latents_mean = torch.tensor(self.vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std)

        mu, logvar = torch.chunk(moments, 2, dim=1)
        mu = self._normalize_latents(mu, latents_mean, latents_std)

        if sample:
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)

            latents = torch.cat([mu, logvar], dim=1)
            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=None)
            del posterior
        else:
            latents = mu
        return latents

    def _inference(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        kv_cache=None,
        kv_cache_flag=None,
        chunk_partition=None,
        use_mean_velocity: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ):
        device = self._execution_device

        if negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps[:-1]):
            r = timesteps[i + 1]

            if t == r:
                continue

            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1)
            timestep = timestep.repeat((1, latent_model_input.shape[1]))

            if use_mean_velocity:
                r_timestep = r.expand(latent_model_input.shape[0]).unsqueeze(-1)
                r_timestep = r_timestep.repeat((1, latent_model_input.shape[1]))
            else:
                r_timestep = timestep

            noise_pred, _ = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                r_timestep=r_timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                chunk_partition=chunk_partition,
                kv_cache=kv_cache,
                kv_cache_flag=copy.deepcopy(kv_cache_flag),
            )
            if self.do_classifier_free_guidance:
                noise_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, r_timestep=r, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs or []:
                    if k == "latents":
                        callback_kwargs[k] = latents
                    elif k == "prompt_embeds":
                        callback_kwargs[k] = prompt_embeds
                    elif k == "negative_prompt_embeds":
                        callback_kwargs[k] = negative_prompt_embeds
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        return latents

    def _denoise_rollout(
        self,
        context_sequence=None,
        num_inference_steps: int = 50,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        use_mean_velocity: bool = True,
        use_kv_cache: bool = True,
        show_progress: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        chunk_partition: Optional[List[int]] = None,
    ):
        latents = latents.permute(0, 2, 1, 3, 4)
        batch_size, num_frame, _, height, width = latents.shape

        # 5. Prepare latent variables
        init_latents = latents

        if chunk_partition is None:
            chunk_partition = list(self.default_chunk_partition)

        if init_latents.shape[1] != sum(chunk_partition):
            raise ValueError(
                f"chunk_partition={chunk_partition} sums to {sum(chunk_partition)}, but the input latent "
                f"sequence has {init_latents.shape[1]} frames; pass an explicit chunk_partition that matches "
                "your num_frames if you are not using the default 81-frame schedule."
            )

        full_token_per_frame = (init_latents.shape[3] // self.transformer.config.patch_size[1]) * (
            init_latents.shape[4] // self.transformer.config.patch_size[2]
        )  # noqa: E501
        compressed_token_per_frame = (init_latents.shape[3] // self.transformer.config.compressed_patch_size[1]) * (
            init_latents.shape[4] // self.transformer.config.compressed_patch_size[2]
        )  # noqa: E501

        # init kv cache
        if use_kv_cache:
            kv_cache = {}

            batch_size = batch_size * 2 if self.do_classifier_free_guidance else batch_size

            for layer_idx in range(self.transformer.config.num_layers):
                kv_cache[layer_idx] = {
                    "full_cache": torch.zeros(
                        (
                            2,
                            batch_size,
                            self.transformer.config.num_attention_heads,
                            self.transformer.config.full_chunk_limit * max(chunk_partition) * full_token_per_frame,
                            self.transformer.config.attention_head_dim,
                        ),
                        device=init_latents.device,
                        dtype=init_latents.dtype,
                    ),
                    "compressed_cache": torch.zeros(
                        (
                            2,
                            batch_size,
                            self.transformer.config.num_attention_heads,
                            (len(chunk_partition) - self.transformer.config.full_chunk_limit + 1)
                            * max(chunk_partition)
                            * compressed_token_per_frame,
                            self.transformer.config.attention_head_dim,
                        ),
                        device=init_latents.device,
                        dtype=init_latents.dtype,
                    ),
                }

            kv_cache_flag = {
                "num_cached_chunks": 0,
                "is_cache_step": False,
            }
        else:
            kv_cache = None
            kv_cache_flag = None

        output = torch.zeros_like(init_latents)

        # setup start sequence
        if context_sequence is not None:
            if "latent" in context_sequence:
                latents = context_sequence["latent"].permute(0, 2, 1, 3, 4)
            else:
                if (context_sequence["raw"].shape[1] - 1) % 4 != 0:
                    raise ValueError(
                        f"`context_sequence['raw']` must have `(num_frames - 1) % 4 == 0`, got "
                        f"num_frames={context_sequence['raw'].shape[1]}."
                    )
                latents = self.vae_encode(context_sequence["raw"])
            current_context_length = latents.shape[1]
            output[:, :current_context_length] = latents
            num_context_chunks = next(
                i + 1 for i in range(len(chunk_partition)) if sum(chunk_partition[: i + 1]) >= current_context_length
            )
        else:
            num_context_chunks = 0

        # Each non-context chunk runs `num_inference_steps` denoising steps that fire
        # callback_on_step_end; context chunks only encode KV cache and never call back.
        self._num_timesteps = (len(chunk_partition) - num_context_chunks) * num_inference_steps

        chunk_iter = range(len(chunk_partition))
        if show_progress:
            chunk_iter = tqdm(chunk_iter)
        for chunk_idx in chunk_iter:
            if chunk_idx >= num_context_chunks:
                pred_latents = self._inference(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    kv_cache=kv_cache,
                    kv_cache_flag=kv_cache_flag,
                    latents=init_latents[:, sum(chunk_partition[:chunk_idx]) : sum(chunk_partition[: chunk_idx + 1])],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    chunk_partition=chunk_partition[: chunk_idx + 1],
                    use_mean_velocity=use_mean_velocity,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
                output[:, sum(chunk_partition[:chunk_idx]) : sum(chunk_partition[: chunk_idx + 1])] = pred_latents

            # Cache the KVs for this chunk so subsequent chunks can attend back to it.
            if chunk_idx < len(chunk_partition) - 1:
                kv_cache = self.encode_kv_cache(
                    kv_cache,
                    kv_cache_flag,
                    chunk_partition=chunk_partition[: chunk_idx + 1],
                    chunk_idx=chunk_idx,
                    output=output,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                )  # noqa: E501

        output = output.permute(0, 2, 1, 3, 4)
        return output

    @torch.no_grad()
    def encode_kv_cache(
        self, kv_cache, kv_cache_flag, chunk_partition, chunk_idx, output, prompt_embeds, negative_prompt_embeds
    ):
        kv_cache_flag["is_cache_step"] = True

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        latents = output[:, : sum(chunk_partition)]
        latent_model_input = (
            torch.cat([latents] * 2).to(self.transformer.dtype)
            if self.do_classifier_free_guidance
            else latents.to(self.transformer.dtype)
        )

        timestep = torch.tensor([0], device=latents.device).expand(latent_model_input.shape[0]).unsqueeze(-1)
        timestep = timestep.repeat((1, latent_model_input.shape[1]))

        r_timestep = torch.tensor([0], device=latents.device).expand(latent_model_input.shape[0]).unsqueeze(-1)
        r_timestep = r_timestep.repeat((1, latent_model_input.shape[1]))

        _, kv_cache = self.transformer(
            hidden_states=latent_model_input,
            chunk_partition=chunk_partition,
            timestep=timestep,
            r_timestep=r_timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            # kv-cache related
            kv_cache=kv_cache,
            kv_cache_flag=copy.deepcopy(kv_cache_flag),
        )

        kv_cache_flag["num_cached_chunks"] += 1
        kv_cache_flag["is_cache_step"] = False

        return kv_cache

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        context_sequence: Optional[Dict[str, torch.Tensor]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        use_mean_velocity: bool = True,
        use_kv_cache: bool = True,
        show_progress: bool = True,
        chunk_partition: Optional[List[int]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, pass `prompt_embeds`
                instead.
            context_sequence (`Dict[str, torch.Tensor]`, *optional*):
                Selects between T2V (`None`), I2V / V2V (`{"raw": <(B, C, T, H, W) tensor in [0, 1]>}` with
                `T = 4n + 1`) and latent-conditioned generation (`{"latent": <(B, C, T_latent, H_latent,
                W_latent) tensor>}` to skip the VAE encode step).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during video generation. Ignored when not using guidance
                (`guidance_scale < 1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated video.
            width (`int`, defaults to `832`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video. Must satisfy `(num_frames - 1) %
                vae_scale_factor_temporal == 0`.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps per chunk. Distilled AnyFlow-FAR checkpoints support any-step
                sampling (1, 2, 4, 8, ...).
            guidance_scale (`float`, defaults to `1.0`):
                Classifier-free guidance scale. The released AnyFlow checkpoints fuse CFG into the weights
                during training; keep at `1.0` unless the checkpoint requires otherwise.
            num_videos_per_prompt (`int`, *optional*, defaults to `1`):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Generator used to seed sampling.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents. If not provided, latents are sampled from the supplied
                `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. If not provided, embeddings are generated from `prompt`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"np"`):
                Output format. One of `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`AnyFlowPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Reserved for future use.
            callback_on_step_end (`Callable`, *optional*):
                A function or [`PipelineCallback`] called at the end of each inference step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*, defaults to `["latents"]`):
                Tensor inputs forwarded to the callback. Must be a subset of `self._callback_tensor_inputs`.
            max_sequence_length (`int`, defaults to `512`):
                The maximum text-encoder sequence length.
            use_mean_velocity (`bool`, defaults to `True`):
                When `True`, condition the flow-map model on both the source timestep `t` and the target
                timestep `r` to predict a mean velocity. Disable to mirror raw Euler stepping.
            use_kv_cache (`bool`, defaults to `True`):
                Reuse the FAR attention KV cache across causal chunks. Disable only for debugging.
            show_progress (`bool`, defaults to `True`):
                Display a per-chunk progress bar.
            chunk_partition (`List[int]`, *optional*):
                Per-chunk frame counts. Defaults to `default_chunk_partition` (matched to the released 81-frame
                checkpoints). When you change `num_frames`, supply a `chunk_partition` that sums to
                `(num_frames - 1) // vae_scale_factor_temporal + 1`.

        Examples:

        Returns:
            [`~AnyFlowPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, an [`AnyFlowPipelineOutput`] is returned, otherwise a `tuple`
                whose first element is the generated video.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._num_timesteps = num_inference_steps

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        init_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        # ``prepare_latents`` returns the standard ``(B, C, T, H, W)`` diffusers layout, which is
        # what the FAR rollout expects on input — no permute needed here.
        init_latents = init_latents.to(transformer_dtype)

        latents = self._denoise_rollout(
            context_sequence=context_sequence,
            num_inference_steps=num_inference_steps,
            latents=init_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            use_mean_velocity=use_mean_velocity,
            use_kv_cache=use_kv_cache,
            show_progress=show_progress,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            chunk_partition=chunk_partition,
        )

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return AnyFlowPipelineOutput(frames=video)
