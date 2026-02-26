# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import html
import math
from enum import Enum
from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Union

import regex as re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import HeliosLoraLoaderMixin
from ...models import AutoencoderKLWan, HeliosTransformer3DModel
from ...schedulers import HeliosUniPCScheduler, UniPCMultistepScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HeliosPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, HeliosPipeline

        >>> # Available models: BestWishYsh/Helios-Base, BestWishYsh/Helios-Mid, BestWishYsh/Helios-Distilled
        >>> model_id = "BestWishYsh/Helios-Base"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = HeliosPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=384,
        ...     width=640,
        ...     num_frames=132,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=24)
        ```
"""


def optimized_scale(positive_flat, negative_flat):
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    return st_star


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


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def apply_schedule_shift(
    image_seq_len,
    sigmas,
    sigmas_two=None,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    exp_max: float = 7.0,
    is_exponential: bool = False,
    mu: float = None,
    return_mu: bool = False,
):
    if mu is None:
        # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
        mu = calculate_shift(
            image_seq_len,
            base_seq_len,
            max_seq_len,
            base_shift,
            max_shift,
        )
        if is_exponential:
            mu = min(mu, math.log(exp_max))
            mu = math.exp(mu)

    if sigmas_two is not None:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        sigmas_two = (sigmas_two * mu) / (1 + (mu - 1) * sigmas_two)
        if return_mu:
            return sigmas, sigmas_two, mu
        else:
            return sigmas, sigmas_two
    else:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        if return_mu:
            return sigmas, mu
        else:
            return sigmas


def add_noise(original_samples, noise, timestep, sigmas, timesteps):
    sigmas = sigmas.to(noise.device)
    timesteps = timesteps.to(noise.device)
    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    sample = (1 - sigma) * original_samples + sigma * noise
    return sample.type_as(noise)


def convert_flow_pred_to_x0(flow_pred, xt, timestep, sigmas, timesteps):
    # use higher precision for calculations
    original_dtype = flow_pred.dtype
    device = flow_pred.device
    flow_pred, xt, sigmas, timesteps = (x.double().to(device) for x in (flow_pred, xt, sigmas, timesteps))

    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred.to(original_dtype)


class VAEDecodeType(str, Enum):
    DEFAULT = "default"
    DEFAULT_BATCH = "default_batch"


class HeliosPipeline(DiffusionPipeline, HeliosLoraLoaderMixin):
    r"""
    Pipeline for text-to-video / image-to-video / video-to-video generation using Helios.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`HeliosTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler | HeliosUniPCScheduler,
        transformer: HeliosTransformer3DModel,
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

        return prompt_embeds, text_inputs.attention_mask.bool()

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
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
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        negative_prompt_attention_mask = None
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

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

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
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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
        height: int = 384,
        width: int = 640,
        num_frames: int = 33,
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
        return latents

    def prepare_image_latents(
        self,
        image: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        fake_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        if latents is None:
            image = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
            latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            latents = (latents - latents_mean) * latents_std
        if fake_latents is None:
            fake_video = image.repeat(1, 1, 33, 1, 1).to(device=device, dtype=self.vae.dtype)
            fake_latents_full = self.vae.encode(fake_video).latent_dist.sample(generator=generator)
            fake_latents_full = (fake_latents_full - latents_mean) * latents_std
            fake_latents = fake_latents_full[:, :, -1:, :, :]
        return latents.to(device=device, dtype=dtype), fake_latents.to(device=device, dtype=dtype)

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        latent_window_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        video = video.to(device=device, dtype=self.vae.dtype)
        if latents is None:
            num_frames = video.shape[2]
            min_frames = (latent_window_size - 1) * 4 + 1
            num_chunks = num_frames // min_frames
            if num_chunks == 0:
                raise ValueError(
                    f"Video must have at least {min_frames} frames "
                    f"(got {num_frames} frames). "
                    f"Required: (latent_window_size - 1) * 4 + 1 = ({latent_window_size} - 1) * 4 + 1 = {min_frames}"
                )
            total_valid_frames = num_chunks * min_frames
            start_frame = num_frames - total_valid_frames

            first_frame = video[:, :, 0:1, :, :]
            first_frame_latent = self.vae.encode(first_frame).latent_dist.sample(generator=generator)
            first_frame_latent = (first_frame_latent - latents_mean) * latents_std

            latents_chunks = []
            for i in range(num_chunks - 1, -1, -1):
                chunk_start = start_frame + i * min_frames
                chunk_end = chunk_start + min_frames
                video_chunk = video[:, :, chunk_start:chunk_end, :, :]
                chunk_latents = self.vae.encode(video_chunk).latent_dist.sample(generator=generator)
                chunk_latents = (chunk_latents - latents_mean) * latents_std
                latents_chunks.insert(0, chunk_latents)
            latents = torch.cat(latents_chunks, dim=2)
        return first_frame_latent.to(device=device, dtype=dtype), latents.to(device=device, dtype=dtype)

    def interpolate_prompt_embeds(
        self,
        prompt_embeds_1: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        interpolation_steps: int = 4,
    ):
        x = torch.lerp(
            prompt_embeds_1,
            prompt_embeds_2,
            torch.linspace(0, 1, steps=interpolation_steps).unsqueeze(1).unsqueeze(2).to(prompt_embeds_1),
        )
        interpolated_prompt_embeds = list(x.chunk(interpolation_steps, dim=0))
        return interpolated_prompt_embeds

    def sample_block_noise(self, batch_size, channel, num_frames, height, width):
        gamma = self.scheduler.config.gamma
        cov = torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma
        dist = torch.distributions.MultivariateNormal(torch.zeros(4, device=cov.device), covariance_matrix=cov)
        block_number = batch_size * channel * num_frames * (height // 2) * (width // 2)

        noise = dist.sample((block_number,))  # [block number, 4]
        noise = noise.view(batch_size, channel, num_frames, height // 2, width // 2, 2, 2)
        noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)
        return noise

    def stage1_sample(
        self,
        latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        guidance_scale: Optional[float] = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        attention_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        transformer_dtype: torch.dtype = None,
        scheduler_type: str = "unipc",
        use_dynamic_shifting: bool = False,
        generator: Optional[torch.Generator] = None,
        # ------------ CFG Zero ------------
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        # -------------- DMD --------------
        use_dmd: bool = False,
        dmd_sigmas: torch.Tensor = None,
        dmd_timesteps: torch.Tensor = None,
        is_amplify_first_chunk: bool = False,
        # ------------ Callback ------------
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: list = None,
        progress_bar=None,
    ):
        batch_size = latents.shape[0]

        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0])

            latent_model_input = latents.to(transformer_dtype)
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(transformer_dtype),
                    latents_history_mid=latents_history_mid.to(transformer_dtype),
                    latents_history_long=latents_history_long.to(transformer_dtype),
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            if self.do_classifier_free_guidance and not use_dmd:
                with self.transformer.cache_context("uncond"):
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if use_cfg_zero_star:
                    noise_pred_text = noise_pred
                    positive_flat = noise_pred_text.view(batch_size, -1)
                    negative_flat = noise_uncond.view(batch_size, -1)

                    alpha = optimized_scale(positive_flat, negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                    alpha = alpha.to(noise_pred_text.dtype)

                    if (i <= zero_steps) and use_zero_init:
                        noise_pred = noise_pred_text * 0.0
                    else:
                        noise_pred = noise_uncond * alpha + guidance_scale * (noise_pred_text - noise_uncond * alpha)
                else:
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            if use_dmd:
                pred_image_or_video = convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=latent_model_input,
                    timestep=t * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                    sigmas=dmd_sigmas,
                    timesteps=dmd_timesteps,
                )
                if i < len(timesteps) - 1:
                    latents = add_noise(
                        pred_image_or_video,
                        randn_tensor(pred_image_or_video.shape, generator=generator, device=device),
                        timesteps[i + 1] * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                        sigmas=dmd_sigmas,
                        timesteps=dmd_timesteps,
                    )
                else:
                    latents = pred_image_or_video
            else:
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

        return latents

    def stage2_sample(
        self,
        latents: torch.Tensor = None,
        stage2_num_stages: int = None,
        stage2_num_inference_steps_list: List[int] = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        guidance_scale: Optional[float] = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        attention_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        transformer_dtype: torch.dtype = None,
        scheduler_type: str = "unipc",  # unipc, euler
        use_dynamic_shifting: bool = False,
        # ------------ CFG Zero ------------
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        # -------------- DMD --------------
        use_dmd: bool = False,
        is_amplify_first_chunk: bool = False,
        # ------------ Callback ------------
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: list = None,
        progress_bar=None,
    ):
        batch_size, num_channel, num_frmaes, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frmaes, num_channel, height, width)
        for _ in range(stage2_num_stages - 1):
            height //= 2
            width //= 2
            latents = (
                F.interpolate(
                    latents,
                    size=(height, width),
                    mode="bilinear",
                )
                * 2
            )
        latents = latents.reshape(batch_size, num_frmaes, num_channel, height, width).permute(0, 2, 1, 3, 4)

        batch_size = latents.shape[0]
        if use_dmd:
            start_point_list = [latents]

        i = 0
        for i_s in range(stage2_num_stages):
            if use_dmd:
                if is_amplify_first_chunk:
                    self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] * 2 + 1, i_s, device=device)
                else:
                    self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] + 1, i_s, device=device)
                self.scheduler.timesteps = self.scheduler.timesteps[:-1]
                self.scheduler.sigmas = torch.cat([self.scheduler.sigmas[:-2], self.scheduler.sigmas[-1:]])
            else:
                self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s], i_s, device=device)

            if i_s > 0:
                height *= 2
                width *= 2
                num_frames = latents.shape[2]
                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frmaes, num_channel, height // 2, width // 2
                )
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                latents = latents.reshape(batch_size, num_frmaes, num_channel, height, width).permute(0, 2, 1, 3, 4)
                # Fix the stage
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]  # the original coeff of signal
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, channel, num_frames, height, width = latents.shape
                noise = self.sample_block_noise(batch_size, channel, num_frames, height, width)
                noise = noise.to(device=device, dtype=transformer_dtype)
                latents = alpha * latents + beta * noise  # To fix the block artifact

                if use_dmd:
                    start_point_list.append(latents)

            if use_dynamic_shifting:
                patch_size = self.transformer.config.patch_size
                image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                    patch_size[0] * patch_size[1] * patch_size[2]
                )
                temp_sigmas = apply_schedule_shift(
                    image_seq_len,
                    self.scheduler.sigmas,
                    base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
                    max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
                    base_shift=self.scheduler.config.get("base_shift", 0.5),
                    max_shift=self.scheduler.config.get("max_shift", 1.15),
                )
                temp_timesteps = self.scheduler.timesteps_per_stage[i_s].min() + temp_sigmas[:-1] * (
                    self.scheduler.timesteps_per_stage[i_s].max() - self.scheduler.timesteps_per_stage[i_s].min()
                )

                self.scheduler.sigmas = temp_sigmas
                self.scheduler.timesteps = temp_timesteps

            timesteps = self.scheduler.timesteps

            for idx, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(torch.int64)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                    )[0]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("cond_uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latents.to(transformer_dtype),
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=latents_history_short.to(transformer_dtype),
                            latents_history_mid=latents_history_mid.to(transformer_dtype),
                            latents_history_long=latents_history_long.to(transformer_dtype),
                        )[0]

                    if use_cfg_zero_star:
                        noise_pred_text = noise_pred
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_uncond.view(batch_size, -1)

                        alpha = optimized_scale(positive_flat, negative_flat)
                        alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                        alpha = alpha.to(noise_pred_text.dtype)

                        if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                            noise_pred = noise_pred_text * 0.0
                        else:
                            noise_pred = noise_uncond * alpha + guidance_scale * (
                                noise_pred_text - noise_uncond * alpha
                            )
                    else:
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                if use_dmd:
                    pred_image_or_video = convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=latents,
                        timestep=timestep,
                        sigmas=self.scheduler.sigmas,
                        timesteps=self.scheduler.timesteps,
                    )
                    if idx < len(timesteps) - 1:
                        latents = add_noise(
                            pred_image_or_video,
                            start_point_list[i_s],
                            timesteps[idx + 1] * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                            sigmas=self.scheduler.sigmas,
                            timesteps=self.scheduler.timesteps,
                        )
                    else:
                        latents = pred_image_or_video
                else:
                    if scheduler_type == "unipc":
                        latents = self.scheduler.step_unipc(noise_pred.float(), t, latents, return_dict=False)[0]
                    else:
                        latents = self.scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

                i += 1

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
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 384,
        width: int = 640,
        num_frames: int = 132,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
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
        # ------------ I2V ------------
        image: Optional[PipelineImageInput] = None,
        image_latents: Optional[torch.Tensor] = None,
        fake_image_latents: Optional[torch.Tensor] = None,
        add_noise_to_image_latents: bool = True,
        image_noise_sigma_min: float = 0.111,
        image_noise_sigma_max: float = 0.135,
        # ------------ V2V ------------
        video: Optional[PipelineImageInput] = None,
        video_latents: Optional[torch.Tensor] = None,
        add_noise_to_video_latents: bool = True,
        video_noise_sigma_min: float = 0.111,
        video_noise_sigma_max: float = 0.135,
        # ------------ Interactive ------------
        use_interpolate_prompt: bool = False,
        interpolate_time_list: list = [7, 7, 7],
        interpolation_steps: int = 3,
        # ------------ Stage 1 ------------
        history_sizes: list = [16, 2, 1],
        latent_window_size: int = 9,
        use_dynamic_shifting: bool = False,
        is_keep_x0: bool = True,
        # ------------ Stage 2 ------------
        is_enable_stage2: bool = False,
        stage2_num_stages: int = 3,
        stage2_num_inference_steps_list: list = [10, 10, 10],
        scheduler_type: str = "unipc",  # unipc, euler
        # ------------ CFG Zero ------------
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        # ------------ DMD ------------
        use_dmd: bool = False,
        is_skip_first_section: bool = False,
        is_amplify_first_chunk: bool = False,
        # ------------ other ------------
        vae_decode_type: VAEDecodeType = "default",  # "default", "default_batch"
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `384`):
                The height in pixels of the generated image.
            width (`int`, defaults to `640`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `132`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HeliosPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~HeliosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HeliosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if image is not None and video is not None:
            raise ValueError("image and video cannot be provided simultaneously")

        if use_interpolate_prompt:
            assert num_videos_per_prompt == 1, f"num_videos_per_prompt must be 1, got {num_videos_per_prompt}"
            assert isinstance(prompt, list), "prompt must be a list"
            assert len(prompt) == len(interpolate_time_list), (
                f"Length mismatch: {len(prompt)} vs {len(interpolate_time_list)}"
            )
            assert min(interpolate_time_list) > interpolation_steps, (
                f"Minimum value {min(interpolate_time_list)} must be greater than {interpolation_steps}"
            )
            interpolate_interval_idx = None
            interpolate_embeds = None
            interpolate_cumulative_list = list(accumulate(interpolate_time_list))

        history_sizes = sorted(history_sizes, reverse=True)  # From big to small

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

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

        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        vae_dtype = self.vae.dtype

        # 2. Define call parameters
        if use_interpolate_prompt or (prompt is not None and isinstance(prompt, str)):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        all_prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        )

        transformer_dtype = self.transformer.dtype
        all_prompt_embeds = all_prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            if use_interpolate_prompt:
                negative_prompt_embeds = negative_prompt_embeds[0].unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare image
        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image_latents, fake_image_latents = self.prepare_image_latents(
                image,
                latents_mean=latents_mean,
                latents_std=latents_std,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=image_latents,
                fake_latents=fake_image_latents,
            )

        if image_latents is not None and add_noise_to_image_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )
            fake_image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (video_noise_sigma_max - video_noise_sigma_min)
                + video_noise_sigma_min
            )
            fake_image_latents = (
                fake_image_noise_sigma * randn_tensor(fake_image_latents.shape, generator=generator, device=device)
                + (1 - fake_image_noise_sigma) * fake_image_latents
            )

        if video is not None:
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            image_latents, video_latents = self.prepare_video_latents(
                video,
                latents_mean=latents_mean,
                latents_std=latents_std,
                latent_window_size=latent_window_size,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=video_latents,
            )

        if video_latents is not None and add_noise_to_video_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )

            noisy_latents_chunks = []
            num_latent_chunks = video_latents.shape[2] // latent_window_size
            for i in range(num_latent_chunks):
                chunk_start = i * latent_window_size
                chunk_end = chunk_start + latent_window_size
                latent_chunk = video_latents[:, :, chunk_start:chunk_end, :, :]

                chunk_frames = latent_chunk.shape[2]
                frame_sigmas = (
                    torch.rand(chunk_frames, device=device, generator=generator)
                    * (video_noise_sigma_max - video_noise_sigma_min)
                    + video_noise_sigma_min
                )
                frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)

                noisy_chunk = (
                    frame_sigmas * randn_tensor(latent_chunk.shape, generator=generator, device=device)
                    + (1 - frame_sigmas) * latent_chunk
                )
                noisy_latents_chunks.append(noisy_chunk)
            video_latents = torch.cat(noisy_latents_chunks, dim=2)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
        num_latent_sections = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        history_video = None
        total_generated_latent_frames = 0

        if not is_keep_x0:
            history_sizes[-1] = history_sizes[-1] + 1
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents, fake_image_latents], dim=2)
            total_generated_latent_frames += 1
        if video_latents is not None:
            history_frames = history_latents.shape[2]
            video_frames = video_latents.shape[2]
            if video_frames < history_frames:
                keep_frames = history_frames - video_frames
                history_latents = torch.cat([history_latents[:, :, :keep_frames, :, :], video_latents], dim=2)
            else:
                history_latents = video_latents
            total_generated_latent_frames += video_latents.shape[2]

        # 6. Denoising loop
        if use_interpolate_prompt:
            if num_latent_sections < max(interpolate_cumulative_list):
                num_latent_sections = sum(interpolate_cumulative_list)
                print(f"Update num_latent_sections to: {num_latent_sections}")

        for k in range(num_latent_sections):
            if use_interpolate_prompt:
                assert num_latent_sections >= max(interpolate_cumulative_list)

                current_interval_idx = 0
                for idx, cumulative_val in enumerate(interpolate_cumulative_list):
                    if k < cumulative_val:
                        current_interval_idx = idx
                        break

                if current_interval_idx == 0:
                    prompt_embeds = all_prompt_embeds[0].unsqueeze(0)
                else:
                    interval_start = interpolate_cumulative_list[current_interval_idx - 1]
                    position_in_interval = k - interval_start

                    if position_in_interval < interpolation_steps:
                        if interpolate_embeds is None or interpolate_interval_idx != current_interval_idx:
                            interpolate_embeds = self.interpolate_prompt_embeds(
                                prompt_embeds_1=all_prompt_embeds[current_interval_idx - 1].unsqueeze(0),
                                prompt_embeds_2=all_prompt_embeds[current_interval_idx].unsqueeze(0),
                                interpolation_steps=interpolation_steps,
                            )
                            interpolate_interval_idx = current_interval_idx

                        prompt_embeds = interpolate_embeds[position_in_interval]
                    else:
                        prompt_embeds = all_prompt_embeds[current_interval_idx].unsqueeze(0)
            else:
                prompt_embeds = all_prompt_embeds

            is_first_section = k == 0
            is_second_section = k == 1
            if is_keep_x0:
                if is_first_section:
                    history_sizes_first_section = [1] + history_sizes.copy()
                    history_latents_first_section = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_section),
                        height // self.vae_scale_factor_spatial,
                        width // self.vae_scale_factor_spatial,
                        device=device,
                        dtype=torch.float32,
                    )
                    if fake_image_latents is not None:
                        history_latents_first_section = torch.cat(
                            [history_latents_first_section, fake_image_latents], dim=2
                        )
                    if video_latents is not None:
                        history_frames = history_latents_first_section.shape[2]
                        video_frames = video_latents.shape[2]
                        if video_frames < history_frames:
                            keep_frames = history_frames - video_frames
                            history_latents_first_section = torch.cat(
                                [history_latents_first_section[:, :, :keep_frames, :, :], video_latents], dim=2
                            )
                        else:
                            history_latents_first_section = video_latents

                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                        history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                            history_sizes_first_section, dim=2
                        )
                    )
                    if image_latents is not None:
                        latents_prefix = image_latents
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                else:
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix = image_latents
                    latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                indices = torch.arange(0, sum([*history_sizes, latent_window_size]))
                (
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_short,
                    indices_hidden_states,
                ) = indices.split([*history_sizes, latent_window_size], dim=0)
                latents_history_long, latents_history_mid, latents_history_short = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)

            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            if not is_enable_stage2:
                self.scheduler.set_timesteps(num_inference_steps, device=device)

                if use_dynamic_shifting:
                    patch_size = self.transformer.config.patch_size
                    image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                        patch_size[0] * patch_size[1] * patch_size[2]
                    )
                    sigmas = torch.linspace(
                        0.999, 0.0, steps=num_inference_steps + 1, dtype=torch.float32, device=device
                    )[:-1]
                    sigmas = apply_schedule_shift(
                        image_seq_len,
                        sigmas,
                        base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
                        max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
                        base_shift=self.scheduler.config.get("base_shift", 0.5),
                        max_shift=self.scheduler.config.get("max_shift", 1.15),
                    )
                    timesteps = sigmas * 1000.0  # rescale to [0, 1000.0)
                    timesteps = timesteps.to(device)
                    sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
                    self.scheduler.timesteps = timesteps
                    self.scheduler.sigmas = sigmas

                timesteps = self.scheduler.timesteps

                dmd_sigmas = None
                dmd_timesteps = None
                if use_dmd:
                    dmd_sigmas = self.scheduler.sigmas.to(self.transformer.device)
                    dmd_timesteps = self.scheduler.timesteps.to(self.transformer.device)

                self._num_timesteps = len(timesteps)
            else:
                num_inference_steps = (
                    sum(stage2_num_inference_steps_list) * 2
                    if is_amplify_first_chunk and use_dmd and is_first_section
                    else sum(stage2_num_inference_steps_list)
                )

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                if is_enable_stage2:
                    latents = self.stage2_sample(
                        latents=latents,
                        stage2_num_stages=stage2_num_stages,
                        stage2_num_inference_steps_list=stage2_num_inference_steps_list,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        guidance_scale=guidance_scale,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_history_long,
                        attention_kwargs=attention_kwargs,
                        device=device,
                        transformer_dtype=transformer_dtype,
                        scheduler_type=scheduler_type,
                        use_dynamic_shifting=use_dynamic_shifting,
                        # ------------ CFG Zero ------------
                        use_cfg_zero_star=use_cfg_zero_star,
                        use_zero_init=use_zero_init,
                        zero_steps=zero_steps,
                        # -------------- DMD --------------
                        use_dmd=use_dmd,
                        is_amplify_first_chunk=is_amplify_first_chunk and is_first_section,
                        # ------------ Callback ------------
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        progress_bar=progress_bar,
                    )
                else:
                    latents = self.stage1_sample(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        timesteps=timesteps,
                        guidance_scale=guidance_scale,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_history_long,
                        attention_kwargs=attention_kwargs,
                        device=device,
                        transformer_dtype=transformer_dtype,
                        scheduler_type=scheduler_type,
                        use_dynamic_shifting=use_dynamic_shifting,
                        generator=generator,
                        # ------------ CFG Zero ------------
                        use_cfg_zero_star=use_cfg_zero_star,
                        use_zero_init=use_zero_init,
                        zero_steps=zero_steps,
                        # -------------- DMD --------------
                        use_dmd=use_dmd,
                        dmd_sigmas=dmd_sigmas,
                        dmd_timesteps=dmd_timesteps,
                        is_amplify_first_chunk=is_amplify_first_chunk and is_first_section,
                        # ------------ Callback ------------
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        progress_bar=progress_bar,
                    )

                if is_keep_x0 and (
                    (is_first_section and image_latents is None) or (is_skip_first_section and is_second_section)
                ):
                    image_latents = latents[:, :, 0:1, :, :]

                total_generated_latent_frames += latents.shape[2]
                history_latents = torch.cat([history_latents, latents], dim=2)
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
                index_slice = (
                    slice(None),
                    slice(None),
                    slice(-latent_window_size, None),
                )

                if vae_decode_type == "default":
                    current_latents = real_history_latents[index_slice].to(vae_dtype) / latents_std + latents_mean
                    current_video = self.vae.decode(current_latents, return_dict=False)[0]

                    if history_video is None:
                        history_video = current_video
                    else:
                        history_video = torch.cat([history_video, current_video], dim=2)

        self._current_timestep = None

        if output_type != "latent":
            if vae_decode_type == "default_batch":
                total_latent_frames = real_history_latents.shape[2]
                batch_size = real_history_latents.shape[0]
                num_chunks = total_latent_frames // latent_window_size

                chunks = (
                    real_history_latents.reshape(
                        batch_size,
                        -1,
                        num_chunks,
                        latent_window_size,
                        real_history_latents.shape[-2],
                        real_history_latents.shape[-1],
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        batch_size * num_chunks,
                        -1,
                        latent_window_size,
                        real_history_latents.shape[-2],
                        real_history_latents.shape[-1],
                    )
                )

                chunks = chunks.to(vae_dtype) / latents_std + latents_mean
                batch_video = self.vae.decode(chunks, return_dict=False)[0]

                video_frames_per_chunk = batch_video.shape[2]
                history_video = (
                    batch_video.reshape(
                        batch_size,
                        num_chunks,
                        -1,
                        video_frames_per_chunk,
                        batch_video.shape[-2],
                        batch_video.shape[-1],
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        batch_size,
                        -1,
                        num_chunks * video_frames_per_chunk,
                        batch_video.shape[-2],
                        batch_video.shape[-1],
                    )
                )

            generated_frames = history_video.size(2)
            generated_frames = (
                generated_frames - 1
            ) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            history_video = history_video[:, :, :generated_frames]
            video = self.video_processor.postprocess_video(history_video, output_type=output_type)
        else:
            video = real_history_latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HeliosPipelineOutput(frames=video)
