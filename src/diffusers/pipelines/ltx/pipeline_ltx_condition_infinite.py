# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5TokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTXVideo
from ...models.transformers import LTXVideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline
from .pipeline_output import LTXPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        # TODO(aryan)
        ```
"""


# Copied from diffusers.pipelines.ltx.pipeline_ltx_condition.linear_quadratic_schedule
def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
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
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class LTXConditionInfinitePipeline(DiffusionPipeline, FromSingleFileMixin, LTXVideoLoraLoaderMixin):
    r"""
    Pipeline for long text/image/video-to-video generation.

    Reference: https://github.com/Lightricks/LTX-Video

    Args:
        transformer ([`LTXVideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLLTXVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTXVideo,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: LTXVideoTransformer3DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer") is not None else 1
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 128
        )

        self.default_height = 512
        self.default_width = 704
        self.default_frames = 121

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
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
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        temporal_tile_size=None,
        temporal_overlap=None,
        horizontal_tiles=None,
        vertical_tiles=None,
        spatial_overlap=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

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
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        if temporal_tile_size < 24 or temporal_tile_size > 1000 or temporal_tile_size % 8 != 0:
            raise ValueError(
                f"`temporal_tile_size` must be in [24, 1000] and divisible by 8 but is {temporal_tile_size}."
            )
        if temporal_overlap < 16 or temporal_overlap > 80 or temporal_overlap % 8 != 0:
            raise ValueError(f"`temporal_overlap` must be in [16, 80] and divisible by 8 but is {temporal_overlap}.")
        if not (1 <= horizontal_tiles <= 6):
            raise ValueError(f"`horizontal_tiles` must be between 1 and 6 but is {horizontal_tiles}.")
        if not (1 <= vertical_tiles <= 6):
            raise ValueError(f"`vertical_tiles` must be between 1 and 6 but is {vertical_tiles}.")
        if not (1 <= spatial_overlap <= 8):
            raise ValueError(f"`spatial_overlap` must be between 1 and 8 but is {spatial_overlap}.")

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx_condition.LTXConditionPipeline._prepare_video_ids
    def _prepare_video_ids(
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, num_frames, patch_size_t, device=device),
            torch.arange(0, height, patch_size, device=device),
            torch.arange(0, width, patch_size, device=device),
            indexing="ij",
        )
        latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)

        return latent_coords

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx_condition.LTXConditionPipeline._scale_video_ids
    def _scale_video_ids(
        video_ids: torch.Tensor,
        scale_factor: int = 32,
        scale_factor_t: int = 8,
        frame_index: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        scaled_latent_coords = (
            video_ids
            * torch.tensor([scale_factor_t, scale_factor, scale_factor], device=video_ids.device)[None, :, None]
        )
        scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
        scaled_latent_coords[:, 0] += frame_index

        return scaled_latent_coords

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def _extract_spatial_tile(self, latents, v_start, v_end, h_start, h_end):
        """Extract spatial tiles from all inputs for a given spatial region."""
        tile_latents = latents[:, :, :, v_start:v_end, h_start:h_end]
        return tile_latents

    def _select_latents(self, latents: torch.Tensor, start_index: int, end_index: int) -> torch.Tensor:
        num_frames = latents.shape[2]
        start_idx = num_frames + start_index if start_index < 0 else start_index
        end_idx = num_frames + end_index if end_index < 0 else end_index
        start_idx = max(0, min(start_idx, num_frames - 1))
        end_idx = max(0, min(end_idx, num_frames - 1))
        if start_idx > end_idx:
            start_idx = min(start_idx, end_idx)
        latents = latents[:, :, start_idx : end_idx + 1, :, :].clone()
        return latents

    @staticmethod
    def _create_spatial_weights(latents, v, h, horizontal_tiles, vertical_tiles, spatial_overlap):
        """Create blending weights for spatial tiles."""
        tile_weights = torch.ones_like(latents)

        # Apply horizontal blending weights
        if h > 0:  # Left overlap
            h_blend = torch.linspace(0, 1, spatial_overlap, device=latents.device)
            tile_weights[:, :, :, :, :spatial_overlap] *= h_blend.view(1, 1, 1, 1, -1)
        if h < horizontal_tiles - 1:  # Right overlap
            h_blend = torch.linspace(1, 0, spatial_overlap, device=latents.device)
            tile_weights[:, :, :, :, -spatial_overlap:] *= h_blend.view(1, 1, 1, 1, -1)

        # Apply vertical blending weights
        if v > 0:  # Top overlap
            v_blend = torch.linspace(0, 1, spatial_overlap, device=latents.device)
            tile_weights[:, :, :, :spatial_overlap, :] *= v_blend.view(1, 1, 1, -1, 1)
        if v < vertical_tiles - 1:  # Bottom overlap
            v_blend = torch.linspace(1, 0, spatial_overlap, device=latents.device)
            tile_weights[:, :, :, -spatial_overlap:, :] *= v_blend.view(1, 1, 1, -1, 1)

        return tile_weights

    # Copied from diffusers.pipelines.ltx.pipeline_ltx_condition.LTXConditionPipeline.trim_conditioning_sequence
    def trim_conditioning_sequence(self, start_frame: int, sequence_num_frames: int, target_num_frames: int):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.
        Returns:
            int: updated sequence length
        """
        scale_factor = self.vae_temporal_compression_ratio
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx_condition.LTXConditionPipeline.add_noise_to_image_conditioning_latents
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if latents is not None:
            if latents.shape != shape:
                raise ValueError(
                    f"Latents shape {latents.shape} does not match expected shape {shape}. Please check the input."
                )
            latents = latents.to(device=device, dtype=dtype)
        else:
            latents = noise

        video_ids = self._prepare_video_ids(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=device,
        )
        video_ids = self._scale_video_ids(
            video_ids, self.vae_spatial_compression_ratio, self.vae_temporal_compression_ratio, 0, device
        )

        return latents, video_ids

    # Copied from diffusers.pipelines.ltx.pipeline_ltx_condition.LTXConditionPipeline.get_timesteps
    def get_timesteps(self, sigmas, timesteps, num_inference_steps, strength):
        num_steps = min(int(num_inference_steps * strength), num_inference_steps)
        start_index = max(num_inference_steps - num_steps, 0)
        sigmas = sigmas[start_index:]
        timesteps = timesteps[start_index:]
        return sigmas, timesteps, num_inference_steps - start_index

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

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
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 25,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        guidance_rescale: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        adain_factor: float = 0.0,
        tone_map_compression_ratio: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        temporal_tile_size: int = 80,
        temporal_overlap: int = 24,
        temporal_overlap_cond_strength: float = 0.5,
        horizontal_tiles: int = 1,
        vertical_tiles: int = 1,
        spatial_overlap: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, defaults to `704`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `3 `):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `128 `):
                Maximum sequence length to use with the `prompt`.
            temporal_tile_size (`int`, defaults to `80`):
                The size of the temporal tile to use for the sampling, in pixel frames, in addition to the overlapping
                region.
            temporal_overlap (`int`, defaults to `24`):
                The overlap between the temporal tiles, in pixel frames.
            horizontal_tiles (`int`, defaults to `1`):
                Number of horizontal spatial tiles to use for the sampling.
            vertical_tiles (`int`, defaults to `1`):
                Number of vertical spatial tiles to use for the sampling.
            spatial_overlap (`int`, defaults to `1`):
                Number of overlapping spatial tiles to use for sampling.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if horizontal_tiles > 1 or vertical_tiles > 1:
            raise ValueError(
                "Setting `horizontal_tiles` or `vertical_tiles` to a value greater than 0 is not supported yet."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            temporal_tile_size=temporal_tile_size,
            temporal_overlap=temporal_overlap,
            horizontal_tiles=horizontal_tiles,
            vertical_tiles=vertical_tiles,
            spatial_overlap=spatial_overlap,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        temporal_tile_size = temporal_tile_size // self.vae_temporal_compression_ratio
        temporal_overlap = temporal_overlap // self.vae_temporal_compression_ratio
        base_tile_height = (latent_height + (vertical_tiles - 1) * spatial_overlap) // vertical_tiles
        base_tile_width = (latent_width + (horizontal_tiles - 1) * spatial_overlap) // horizontal_tiles
        temporal_range_max = latent_num_frames + temporal_tile_size - temporal_overlap
        temporal_range_step = temporal_tile_size - temporal_overlap

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        vae_dtype = self.vae.dtype

        # 3. Prepare text embeddings & conditioning image/video
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare timesteps
        if timesteps is None:
            sigmas = linear_quadratic_schedule(num_inference_steps)
            timesteps = sigmas * 1000

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, _ = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            latents=latents,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        final_latents = torch.zeros_like(latents)
        weights = torch.zeros_like(latents)

        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                # Calculate tile boundaries
                h_start = h * (base_tile_width - spatial_overlap)
                v_start = v * (base_tile_height - spatial_overlap)

                # Adjust end positions for edge tiles
                h_end = min(h_start + base_tile_width, latent_width) if h < horizontal_tiles - 1 else latent_width
                v_end = min(v_start + base_tile_height, latent_height) if v < vertical_tiles - 1 else latent_height

                # Calculate actual tile dimensions
                latent_tile_height = v_end - v_start
                latent_tile_width = h_end - h_start

                # Extract spatial tiles from all inputs
                tile_latents = self._extract_spatial_tile(latents, v_start, v_end, h_start, h_end)

                tile_out_latents = None
                first_tile_out_latents = None

                for index_temporal_tile, (start_index, end_index) in enumerate(
                    zip(
                        range(0, temporal_range_max, temporal_range_step),
                        range(temporal_tile_size, temporal_range_max, temporal_range_step),
                    )
                ):
                    latent_chunk = self._select_latents(
                        tile_latents, start_index, min(end_index - 1, tile_latents.shape[2] - 1)
                    )
                    latent_tile_num_frames = latent_chunk.shape[2]

                    if start_index > 0:
                        last_latent_chunk = self._select_latents(tile_out_latents, -temporal_overlap, -1)
                        last_latent_tile_num_frames = last_latent_chunk.shape[2]
                        latent_chunk = torch.cat([last_latent_chunk, latent_chunk], dim=2)
                        total_latent_num_frames = last_latent_tile_num_frames + latent_tile_num_frames
                        last_latent_chunk = self._pack_latents(
                            last_latent_chunk,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )
                        last_latent_chunk_num_tokens = last_latent_chunk.shape[1]
                        if self.do_classifier_free_guidance:
                            last_latent_chunk = torch.cat([last_latent_chunk, last_latent_chunk], dim=0)

                        conditioning_mask = torch.zeros(
                            (batch_size, total_latent_num_frames),
                            dtype=torch.float32,
                            device=device,
                        )
                        # conditioning_mask[:, :last_latent_tile_num_frames] = temporal_overlap_cond_strength
                        conditioning_mask[:, :last_latent_tile_num_frames] = 1.0
                    else:
                        total_latent_num_frames = latent_tile_num_frames

                    latent_chunk = self._pack_latents(
                        latent_chunk,
                        self.transformer_spatial_patch_size,
                        self.transformer_temporal_patch_size,
                    )

                    video_ids = self._prepare_video_ids(
                        batch_size,
                        total_latent_num_frames,
                        latent_tile_height,
                        latent_tile_width,
                        patch_size_t=self.transformer_temporal_patch_size,
                        patch_size=self.transformer_spatial_patch_size,
                        device=device,
                    )

                    if start_index > 0:
                        conditioning_mask = conditioning_mask.gather(1, video_ids[:, 0])
                        conditioning_mask_model_input = (
                            torch.cat([conditioning_mask, conditioning_mask])
                            if self.do_classifier_free_guidance
                            else conditioning_mask
                        )

                    video_ids = self._scale_video_ids(
                        video_ids,
                        scale_factor=self.vae_spatial_compression_ratio,
                        scale_factor_t=self.vae_temporal_compression_ratio,
                        frame_index=0,
                        device=device,
                    )
                    video_ids = video_ids.float()
                    video_ids[:, 0] = video_ids[:, 0] * (1.0 / frame_rate)
                    if self.do_classifier_free_guidance:
                        video_ids = torch.cat([video_ids, video_ids], dim=0)

                    # Set timesteps
                    inner_timesteps, inner_num_inference_steps = retrieve_timesteps(
                        self.scheduler, num_inference_steps, device, timesteps
                    )
                    sigmas = self.scheduler.sigmas
                    num_warmup_steps = max(len(inner_timesteps) - inner_num_inference_steps * self.scheduler.order, 0)
                    self._num_timesteps = len(inner_timesteps)

                    with self.progress_bar(total=inner_num_inference_steps) as progress_bar:
                        for i, t in enumerate(inner_timesteps):
                            if self.interrupt:
                                continue

                            self._current_timestep = t
                            latent_model_input = (
                                torch.cat([latent_chunk] * 2) if self.do_classifier_free_guidance else latent_chunk
                            )
                            latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                            # Create timestep tensor that has prod(latent_model_input.shape) elements
                            if start_index == 0:
                                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1)
                            else:
                                timestep = t.view(1, 1).expand((latent_model_input.shape[:-1])).clone()
                                timestep[:, :last_latent_chunk_num_tokens] = 0.0

                            timestep = timestep.float()
                            # timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
                            # if start_index > 0:
                            #     timestep = torch.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

                            with self.transformer.cache_context("cond_uncond"):
                                noise_pred = self.transformer(
                                    hidden_states=latent_model_input,
                                    encoder_hidden_states=prompt_embeds,
                                    timestep=timestep,
                                    encoder_attention_mask=prompt_attention_mask,
                                    video_coords=video_ids,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]

                            if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + self.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                                timestep, _ = timestep.chunk(2)

                                if self.guidance_rescale > 0:
                                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                                    noise_pred = rescale_noise_cfg(
                                        noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                                    )

                            denoised_latent_chunk = self.scheduler.step(
                                -noise_pred, t, latent_chunk, per_token_timesteps=timestep, return_dict=False
                            )[0]
                            if start_index == 0:
                                latent_chunk = denoised_latent_chunk
                            else:
                                latent_chunk = torch.cat(
                                    [last_latent_chunk, denoised_latent_chunk[:, last_latent_chunk_num_tokens:]], dim=1
                                )
                                # tokens_to_denoise_mask = (t / 1000 - 1e-6 < (1.0 - conditioning_mask)).unsqueeze(-1)
                                # latent_chunk = torch.where(tokens_to_denoise_mask, denoised_latent_chunk, latent_chunk)

                            if callback_on_step_end is not None:
                                callback_kwargs = {}
                                for k in callback_on_step_end_tensor_inputs:
                                    callback_kwargs[k] = locals()[k]
                                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                                latent_chunk = callback_outputs.pop("latents", latent_chunk)
                                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                            # call the callback, if provided
                            if i == len(inner_timesteps) - 1 or (
                                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                            ):
                                progress_bar.update()

                            if XLA_AVAILABLE:
                                xm.mark_step()

                    latent_chunk = self._unpack_latents(
                        latent_chunk,
                        total_latent_num_frames,
                        latent_tile_height,
                        latent_tile_width,
                        self.transformer_spatial_patch_size,
                        self.transformer_temporal_patch_size,
                    )

                    if start_index == 0:
                        first_tile_out_latents = latent_chunk.clone()
                    else:
                        latent_chunk = latent_chunk[:, :, last_latent_tile_num_frames:-1, :, :]
                        latent_chunk = LTXLatentUpsamplePipeline.adain_filter_latent(
                            latent_chunk, first_tile_out_latents, adain_factor
                        )

                        alpha = torch.linspace(1, 0, temporal_overlap + 1, device=latent_chunk.device)[1:-1]
                        alpha = alpha.view(1, 1, -1, 1, 1)

                        # Combine samples
                        t_minus_one = temporal_overlap - 1
                        parts = [
                            tile_out_latents[:, :, :-t_minus_one],
                            alpha * tile_out_latents[:, :, -t_minus_one:]
                            + (1 - alpha) * latent_chunk[:, :, :t_minus_one],
                            latent_chunk[:, :, t_minus_one:],
                        ]
                        latent_chunk = torch.cat(parts, dim=2)

                    tile_out_latents = latent_chunk.clone()

                tile_weights = self._create_spatial_weights(
                    tile_out_latents, v, h, horizontal_tiles, vertical_tiles, spatial_overlap
                )
                final_latents[:, :, :, v_start:v_end, h_start:h_end] += latent_chunk * tile_weights
                weights[:, :, :, v_start:v_end, h_start:h_end] += tile_weights

        eps = 1e-8
        latents = final_latents / (weights + eps)
        latents = LTXLatentUpsamplePipeline.tone_map_latents(latents, tone_map_compression_ratio)

        if output_type == "latent":
            video = latents
        else:
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
