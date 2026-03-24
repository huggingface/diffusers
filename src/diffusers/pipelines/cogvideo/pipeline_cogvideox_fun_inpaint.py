# Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI, Alibaba-PAI and The HuggingFace Team.
# All rights reserved.
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

import inspect
import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor, is_valid_image, is_valid_image_imagelist
from ...loaders import CogVideoXLoraLoaderMixin
from ...models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from ...models.embeddings import get_3d_rotary_pos_embed
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from .pipeline_output import CogVideoXPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoXFunInpaintPipeline, DDIMScheduler
        >>> from diffusers.utils import export_to_video, load_video

        >>> pipe = CogVideoXFunInpaintPipeline.from_pretrained(
        ...     "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        >>> pipe.to("cuda")

        >>> video = load_video(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        ... )
        >>> mask_video = [Image.new("L", frame.size, 255) for frame in video]
        >>> prompt = "A cinematic mountain hike with dramatic lighting."

        >>> output = pipe(prompt=prompt, video=video, mask_video=mask_video, output_type="pt").frames[0]
        >>> export_to_video(output, "output.mp4", fps=8)
        ```
"""


# Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def resize_mask(mask: torch.Tensor, latent: torch.Tensor, process_first_frame_only: bool = True) -> torch.Tensor:
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(mask[:, :, :1], size=target_size, mode="trilinear", align_corners=False)

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:], size=target_size, mode="trilinear", align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(mask, size=target_size, mode="trilinear", align_corners=False)

    return resized_mask


def add_noise_to_reference_video(image: torch.Tensor, ratio: float | None = None) -> torch.Tensor:
    if ratio is None:
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],), device=image.device)
        sigma = torch.exp(sigma).to(image.dtype)
    else:
        sigma = torch.ones((image.shape[0],), device=image.device, dtype=image.dtype) * ratio

    image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    image_noise = torch.where(image == -1, torch.zeros_like(image_noise), image_noise)
    return image + image_noise


class CogVideoXFunInpaintPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for CogVideoX Fun inpainting and video editing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        text_encoder (`T5EncoderModel`):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        vae (`AutoencoderKLCogVideoX`):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer (`CogVideoXTransformer3DModel`):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler (`SchedulerMixin`):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->vae->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.vae_scaling_factor_image = self.vae.config.scaling_factor if getattr(self, "vae", None) else 0.7

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._get_t5_prompt_embeds
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
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.encode_prompt
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

    def _preprocess_mask_video(self, mask_video, height: int, width: int) -> torch.Tensor:
        if isinstance(mask_video, (np.ndarray, torch.Tensor)) and mask_video.ndim == 5:
            mask_video = list(mask_video)
        elif isinstance(mask_video, list) and (is_valid_image(mask_video[0]) or is_valid_image_imagelist(mask_video)):
            mask_video = [mask_video]
        elif isinstance(mask_video, list) and is_valid_image_imagelist(mask_video[0]):
            pass
        else:
            raise ValueError(
                "Input `mask_video` is in incorrect format. Supported formats are PIL frames, list of videos, "
                "and 5D tensor/ndarray."
            )

        mask_video = torch.stack(
            [self.mask_processor.preprocess(img, height=height, width=width) for img in mask_video], dim=0
        )
        mask_video = mask_video.permute(0, 2, 1, 3, 4)
        return mask_video.to(dtype=torch.float32)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        video=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_video_latents=False,
    ):
        shape = (
            batch_size,
            (video_length - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if return_video_latents or (latents is None and not is_strength_max):
            video = video.to(device=device, dtype=self.vae.dtype)

            videos = []
            for i in range(video.size(0)):
                video_bs = video[i : i + 1]
                video_bs = self.vae.encode(video_bs)[0]
                video_bs = video_bs.sample()
                videos.append(video_bs)
            video = torch.cat(videos, dim=0)
            video = video * self.vae_scaling_factor_image

            video_latents = video.repeat(batch_size // video.shape[0], 1, 1, 1, 1)
            video_latents = video_latents.to(device=device, dtype=dtype)
            video_latents = video_latents.permute(0, 2, 1, 3, 4)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = noise if is_strength_max else self.scheduler.add_noise(video_latents, noise, timestep)
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_video_latents:
            outputs += (video_latents,)

        return outputs

    def prepare_mask_latents(
        self,
        mask: torch.Tensor | None = None,
        masked_image: torch.Tensor | None = None,
        device: torch.device | None = None,
        noise_aug_strength: float | None = None,
    ):
        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            masks = []
            for i in range(mask.size(0)):
                current_mask = mask[i].unsqueeze(0)
                current_mask = self.vae.encode(current_mask)[0]
                current_mask = current_mask.mode()
                masks.append(current_mask)
            mask = torch.cat(masks, dim=0)
            mask = mask * self.vae_scaling_factor_image

        if masked_image is not None:
            if getattr(self.transformer.config, "add_noise_in_inpaint_model", False):
                masked_image = add_noise_to_reference_video(masked_image, ratio=noise_aug_strength)

            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            mask_pixel_values = []
            for i in range(masked_image.size(0)):
                mask_pixel_value = masked_image[i].unsqueeze(0)
                mask_pixel_value = self.vae.encode(mask_pixel_value)[0]
                mask_pixel_value = mask_pixel_value.mode()
                mask_pixel_values.append(mask_pixel_value)
            masked_image_latents = torch.cat(mask_pixel_values, dim=0)
            masked_image_latents = masked_image_latents * self.vae_scaling_factor_image
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.decode_latents
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4) # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents
        return self.vae.decode(latents).sample

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        strength,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        video=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should be in [0.0, 1.0] but is {strength}")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if video is not None and latents is not None:
            raise ValueError("Only one of `video` or `latents` should be provided.")

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]
        
        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        video: list[Image.Image] | list[list[Image.Image]] | torch.Tensor | np.ndarray | None = None,
        mask_video: list[Image.Image] | list[list[Image.Image]] | torch.Tensor | np.ndarray | None = None,
        masked_video_latents: list[Image.Image] | list[list[Image.Image]] | torch.Tensor | np.ndarray | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        strength: float = 1.0,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 226,
        noise_aug_strength: float = 0.0563,
    ) -> CogVideoXPipelineOutput | tuple:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            video (`list[PIL.Image.Image]`):
                The video to condition the generation on. Must be a list of images/frames of the video. If not
                provided, `masked_video_latents` must be provided.
            mask_video (`list[PIL.Image.Image]`):
                The mask video to condition the generation on. Must be a list of images/frames of the video. If not
                provided, `masked_video_latents` must be provided.
            masked_video_latents (`list[PIL.Image.Image]`):
                The masked video latents to condition the generation on. Must be a list of images/frames of the video. If not
                provided, `video` must be provided.
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_width * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, *optional*, defaults to self.transformer.config.sample_frames):
                The number of frames in the video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`list[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            strength (`float`, *optional*, defaults to 1.0):
                The strength of the inpainting.
            guidance_scale (`float`, *optional*, defaults to 6.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            use_dynamic_cfg (`bool`, *optional*, defaults to False):
                Whether to use dynamic cfg or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                The eta value to use for the denoising process.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.
            noise_aug_strength (`float`, *optional*, defaults to 0.0563):
                The strength of the noise augmentation.
                
        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames
        
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            strength=strength,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            video=video,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        
        if XLA_AVAILABLE:
            timestep_device = "cpu"
        else:
            timestep_device = device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timestep_device, timesteps
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, timesteps, strength, device)
        self._num_timesteps = len(timesteps)

        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        is_strength_max = strength == 1.0

        # 5. Prepare latents
        init_video = None
        if video is not None:
            init_video = self.video_processor.preprocess_video(video, height=height, width=width).to(dtype=torch.float32)

        video_length = init_video.shape[2] if init_video is not None else num_frames

        local_latent_length = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        patch_size_t = self.transformer.config.patch_size_t
        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
            num_frames = max(num_frames, 1)

        if video_length > num_frames:
            logger.warning(
                "The input video length is clipped so latent frames are divisible by `patch_size_t` for this model."
            )
            video_length = num_frames
            init_video = init_video[:, :, :video_length] if init_video is not None else None

        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels
        return_image_latents = num_channels_transformer == num_channels_latents

        latents_outputs = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            video=init_video,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_video_latents=return_image_latents,
        )
        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        inpaint_latents = None
        if mask_video is not None:
            mask_condition = self._preprocess_mask_video(mask_video, height=height, width=width).to(device=device)
            if mask_condition.shape[2] > video_length:
                mask_condition = mask_condition[:, :, :video_length]

            if (mask_condition == 1).all():
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(device, latents.dtype)
                masked_video_latents_cond = torch.zeros_like(latents).to(device, latents.dtype)
                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents_cond] * 2)
                    if do_classifier_free_guidance
                    else masked_video_latents_cond
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            elif num_channels_transformer != num_channels_latents:
                mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                if masked_video_latents is None:
                    if init_video is None:
                        raise ValueError(
                            "`video` is required when `mask_video` is not fully masked and "
                            "`masked_video_latents` is not provided."
                        )
                    masked_video = (
                        init_video * (mask_condition_tile < 0.5)
                        + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                    )
                else:
                    masked_video = self.video_processor.preprocess_video(
                        masked_video_latents, height=height, width=width
                    ).to(dtype=torch.float32)
                    if masked_video.shape[2] > video_length:
                        masked_video = masked_video[:, :, :video_length]

                _, masked_video_latents_cond = self.prepare_mask_latents(
                    None,
                    masked_video,
                    batch_size,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                    noise_aug_strength=noise_aug_strength,
                )
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents_cond)
                mask_latents = mask_latents.to(masked_video_latents_cond.device) * self.vae_scaling_factor_image

                mask_input = mask_latents.permute(0, 2, 1, 3, 4)
                masked_video_latents_input = masked_video_latents_cond.permute(0, 2, 1, 3, 4)
                if do_classifier_free_guidance:
                    mask_input = torch.cat([mask_input] * 2)
                    masked_video_latents_input = torch.cat([masked_video_latents_input] * 2)

                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                inpaint_latents = None
        elif num_channels_transformer != num_channels_latents:
            mask_latents = torch.zeros_like(latents)[:, :, :1].to(device, latents.dtype)
            masked_video_latents_cond = torch.zeros_like(latents).to(device, latents.dtype)
            mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
            masked_video_latents_input = (
                torch.cat([masked_video_latents_cond] * 2) if do_classifier_free_guidance else masked_video_latents_cond
            )
            inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)

        if num_channels_transformer != num_channels_latents and inpaint_latents is not None:
            num_channels_inpaint = inpaint_latents.shape[2]
            if num_channels_latents + num_channels_inpaint != self.transformer.config.in_channels:
                raise ValueError(
                    f"Incorrect channel configuration. Transformer expects `{self.transformer.config.in_channels}` "
                    f"channels, but got `{num_channels_latents + num_channels_inpaint}` from latents + inpaint latents."
                )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)        
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if inpaint_latents is not None:
                    latent_model_input = torch.cat([latent_model_input, inpaint_latents], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                # predict noise model_output
                with self.transformer.cache_context("cond_uncond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type != "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
