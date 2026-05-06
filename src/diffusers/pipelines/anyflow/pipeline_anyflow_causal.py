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
from einops import rearrange
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import WanLoraLoaderMixin
from ...models import AnyFlowTransformer3DModel, AutoencoderKLWan
from ...models.autoencoders.vae import DiagonalGaussianDistribution
from ...schedulers import FlowMapEulerDiscreteScheduler
from ...utils import is_ftfy_available, logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import AnyFlowPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class AnyFlowCausalPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Causal (FAR-based) text-to-video / image-to-video / text+video-to-video pipeline for AnyFlow checkpoints.

    The pipeline drives a frame-level autoregressive sampling loop over chunks: each chunk is denoised with
    flow-map steps while attending only to past chunks via block-sparse causal attention, and intermediate
    KV cache is reused across chunks. Set ``task_type`` per call to switch between ``"t2v"``, ``"i2v"``, and
    ``"tv2v"``.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer from [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).
        text_encoder ([`UMT5EncoderModel`]):
            [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) text encoder.
        transformer ([`AnyFlowTransformer3DModel`]):
            Conditional 3D Transformer (must be configured with ``init_far_model=True`` and
            ``init_flowmap_model=True``).
        vae ([`AutoencoderKLWan`]):
            VAE that encodes/decodes videos to and from latent representations.
        scheduler ([`FlowMapEulerDiscreteScheduler`]):
            Flow-map sampler.
        use_mean_velocity (`bool`, defaults to `True`):
            When ``True`` the model output is averaged across two anchor times to reduce discretization error.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: AnyFlowTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMapEulerDiscreteScheduler,
        use_mean_velocity: bool = True,
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
        self.use_mean_velocity = use_mean_velocity

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
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

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
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
        latents = rearrange(latents, "b c t h w -> b t c h w")
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
    def current_timestep(self):
        return self._current_timestep

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
        context_sequence = rearrange(context_sequence, "b c t h w -> b t c h w")
        return context_sequence

    def _normalize_latents(self, latents, latents_mean, latents_std):
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    @torch.no_grad()
    def encode_latents(self, videos, sample=True):
        videos = rearrange(videos, "b t c h w -> b c t h w")
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

    def inference(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        kv_cache=None,
        kv_cache_flag=None,
        grad_timestep=None,
        chunk_partition=None,
    ):
        if negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        def inference_range(latents, timesteps):

            for i, t in enumerate(timesteps[:-1]):
                r = timesteps[i + 1]

                if t == r:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1)
                timestep = timestep.repeat((1, latent_model_input.shape[1]))

                if self.use_mean_velocity:
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
                    # kv-cache related
                    kv_cache=kv_cache,
                    kv_cache_flag=copy.deepcopy(kv_cache_flag),
                )
                if self.do_classifier_free_guidance:
                    noise_uncond, noise_pred = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, latents, t, r)

            return latents

        device = self._execution_device

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if grad_timestep is None:
            x_final_val = inference_range(latents, timesteps)
            return x_final_val

        # 6. Denoising loop
        self._num_timesteps = len(timesteps)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        prev_timestep = [timesteps[0], timesteps[grad_timestep]]
        current_timestep = [timesteps[grad_timestep], timesteps[grad_timestep + 1]]
        post_timestep = [timesteps[grad_timestep + 1], timesteps[-1]]

        # 1. Fast-forward to the target timestep without tracking gradients
        latents = inference_range(latents, prev_timestep)

        # 2. Execute a single differentiable step to anchor the gradient flow
        x_next_grad = inference_range(latents, current_timestep)

        # 3. Complete the rollout to x0 in no_grad mode to save VRAM
        x_final_val = inference_range(x_next_grad, post_timestep)
        return x_final_val

    def training_rollout(
        self,
        context_sequence=None,
        num_inference_steps: int = 50,
        grad_timestep: int = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        use_kv_cache=True,
    ):
        self._guidance_scale = guidance_scale

        latents = rearrange(latents, "b c t h w -> b t c h w")
        batch_size, num_frame, _, height, width = latents.shape

        # 5. Prepare latent variables
        init_latents = latents

        chunk_partition = self.transformer.config.chunk_partition

        assert init_latents.shape[1] == sum(chunk_partition), (
            "please check the chunk_partition equal to num_smaple_frames"
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
                latents = rearrange(context_sequence["latent"], "b c t h w -> b t c h w")
            else:
                assert (context_sequence["raw"].shape[1] - 1) % 4 == 0, "require 4n+1 frames"
                latents = self.vae_encode(context_sequence["raw"])
            current_context_length = latents.shape[1]
            output[:, :current_context_length] = latents
            num_context_chunks = next(
                i + 1 for i in range(len(chunk_partition)) if sum(chunk_partition[: i + 1]) >= current_context_length
            )
        else:
            num_context_chunks = 0

        for chunk_idx in tqdm(range(len(chunk_partition))):
            if chunk_idx >= num_context_chunks:
                pred_latents = self.inference(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    kv_cache=kv_cache,
                    kv_cache_flag=kv_cache_flag,
                    latents=init_latents[:, sum(chunk_partition[:chunk_idx]) : sum(chunk_partition[: chunk_idx + 1])],
                    num_inference_steps=num_inference_steps,
                    grad_timestep=grad_timestep,
                    guidance_scale=guidance_scale,
                    chunk_partition=chunk_partition[: chunk_idx + 1],
                )
                output[:, sum(chunk_partition[:chunk_idx]) : sum(chunk_partition[: chunk_idx + 1])] = pred_latents

            # step1: save to kv cache
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

        output = rearrange(output, "b f c h w -> b c f h w")
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
            torch.cat([latents] * 2).to(torch.bfloat16)
            if self.do_classifier_free_guidance
            else latents.to(torch.bfloat16)
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
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        context_sequence=None,
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
        show_progress=True,
        use_kv_cache=True,
    ):

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
        self._current_timestep = None
        self._interrupt = False

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

        transformer_dtype = torch.bfloat16
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
        init_latents = init_latents.to(transformer_dtype)
        init_latents = rearrange(init_latents, "b f c h w -> b c f h w")

        latents = self.training_rollout(
            context_sequence=context_sequence,
            num_inference_steps=num_inference_steps,
            grad_timestep=None,
            latents=init_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
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
