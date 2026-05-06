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
# Adapted from diffusers.pipelines.wan.pipeline_wan.WanPipeline (v0.35.1) for any-step flow-map sampling.

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


class AnyFlowPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Bidirectional text-to-video generation pipeline for AnyFlow flow-map-distilled checkpoints.

    AnyFlow learns arbitrary-interval transitions :math:`z_t \to z_r` rather than the fixed
    :math:`z_t \to z_0` mapping of consistency models, so a single distilled checkpoint can be evaluated at
    1, 2, 4, 8, 16... NFE without retraining. This pipeline operates over the full video tensor in one
    bidirectional pass; for frame-level autoregressive (causal) generation use ``AnyFlowFARPipeline``.

    The released NVIDIA checkpoints loaded by this pipeline went through a two-stage LoRA distillation:
    (1) forward Flow-Map training with the MeanFlow identity as a stop-grad regression target, and
    (2) on-policy distillation that combines Flow-Map backward simulation with DMD reverse-divergence
    supervision over the student's own rollouts. Sampling at inference is plain Euler in mean-velocity
    form (``z_r = z_t - (t - r) * u``) with no re-noising and no CFG (guidance was fused into the model
    weights during stage 1). See ``_denoise_rollout`` for the rollout entry point reused during DMD
    fine-tuning.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer from [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).
        text_encoder ([`UMT5EncoderModel`]):
            [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) text encoder.
        transformer ([`AnyFlowTransformer3DModel`]):
            Bidirectional flow-map 3D Transformer.
        vae ([`AutoencoderKLWan`]):
            VAE that encodes/decodes videos to and from latent representations.
        scheduler ([`FlowMapEulerDiscreteScheduler`]):
            Flow-map sampler. The pipeline drives ``scheduler.step(..., timestep, r_timestep)`` per inference
            step.
        use_mean_velocity (`bool`, defaults to `True`):
            When ``True`` the model output is averaged across two anchor times to reduce discretization error
            (the default training-time behavior). Disable to mirror raw Euler stepping.
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

    def _denoise_rollout(
        self,
        context_sequence=None,
        num_inference_steps: int = 50,
        grad_timestep: int = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ):
        r"""
        Three-segment Flow-Map backward simulation used as the on-policy rollout for stage-2 DMD
        distillation. Not part of the standard inference path — end users should call ``__call__``.

        When ``grad_timestep`` is ``None`` the method reduces to a plain (no-grad) multi-step rollout.
        When ``grad_timestep`` is set, the rollout is split into three segments (``z_T -> z_t``, the
        gradient anchor ``z_t -> z_r`` step, then ``z_r -> z_0``); every segment contributes to the
        autograd graph, matching Algorithm 2 of the AnyFlow paper. This is the entry point that the
        on-policy trainer composes with a frozen ``real_score`` and a trainable discriminator to compute
        the DMD KL-gradient surrogate.

        Args:
            context_sequence (`torch.Tensor`, *optional*):
                Clean prefix latents to keep fixed during the rollout (used by I2V / TV2V variants).
            num_inference_steps (`int`, defaults to 50):
                Number of inference steps used to discretize the rollout schedule.
            grad_timestep (`int`, *optional*):
                Index into the inference schedule that becomes the gradient anchor. ``None`` disables
                gradients and turns the call into a plain rollout.
            latents (`torch.Tensor`, *optional*):
                Initial Gaussian latents at ``t = T``.
            prompt_embeds, negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text encoder embeddings.
            guidance_scale (`float`, defaults to 1.0):
                CFG scale applied at runtime. Set to 1.0 (default) for distilled checkpoints since CFG
                was fused into the weights during stage 1.

        Returns:
            `torch.Tensor`: the rolled-out latents at the final timestep.
        """
        self._guidance_scale = guidance_scale

        if negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # setup start sequence
        if context_sequence is not None:
            context_length = context_sequence.shape[1]

        def inference_range(latents, timesteps):
            nonlocal prompt_embeds, negative_prompt_embeds

            for i, t in enumerate(tqdm(timesteps[:-1])):
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

                if context_sequence is not None:
                    latent_model_input[:, :context_length, ...] = context_sequence
                    timestep[:, :context_length] = 0

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    r_timestep=r_timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond, noise_pred = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, latents, t, r)

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
        init_latents = init_latents.to(transformer_dtype)

        # setup start sequence
        if context_sequence is not None:
            context_sequence = self.vae_encode(context_sequence)
            context_length = context_sequence.shape[1]

        latents = self._denoise_rollout(
            context_sequence=context_sequence,
            num_inference_steps=num_inference_steps,
            grad_timestep=None,
            latents=init_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        if context_sequence is not None:
            latents[:, :context_length, ...] = context_sequence
        latents = rearrange(latents, "b f c h w -> b c f h w")

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
