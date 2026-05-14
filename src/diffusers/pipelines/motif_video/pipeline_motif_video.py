# Copyright 2026 Motif Technologies, Inc. and The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch

# NOTE: This pipeline requires transformers>=5.1.0 for T5Gemma2Encoder support.
# The T5Gemma2Encoder class is only available in transformers 5.1.0 and later.
from transformers import BatchEncoding, PreTrainedTokenizerBase, SiglipImageProcessor, T5Gemma2Encoder

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...guiders import BaseGuidance
from ...models import AutoencoderKLWan
from ...models.transformers import MotifVideoTransformer3DModel
from ...schedulers import SchedulerMixin
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import MotifVideoPipelineOutput


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
        >>> from diffusers import MotifVideoPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Load the Motif-Video pipeline
        >>> motif_video_model_id = "Motif-Technologies/Motif-Video-2B"
        >>> pipe = MotifVideoPipeline.from_pretrained(motif_video_model_id, torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
        >>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        >>> video = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     width=1280,
        ...     height=736,
        ...     num_frames=121,
        ...     num_inference_steps=50,
        ... ).frames[0]
        >>> export_to_video(video, "output.mp4", fps=24)
        ```
"""


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


class MotifVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Motif-Video.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`MotifVideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents. Should be an
            instance of a class inheriting from `SchedulerMixin`, such as [`DPMSolverMultistepScheduler`]. If not
            provided, uses the scheduler attached to the pretrained model.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5Gemma2Encoder`]):
            Primary text encoder for encoding text prompts into embeddings.
        tokenizer ([`PreTrainedTokenizerBase`]):
            Tokenizer corresponding to the primary text encoder.
        guider ([`BaseGuidance`]):
            The guidance method to use. Should be an instance of a class inheriting from `BaseGuidance`, such as
            [`ClassifierFreeGuidance`], [`AdaptiveProjectedGuidance`], or [`SkipLayerGuidance`]. If not provided,
            defaults to `ClassifierFreeGuidance`.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: SchedulerMixin,
        vae: AutoencoderKLWan,
        text_encoder: T5Gemma2Encoder,
        tokenizer: PreTrainedTokenizerBase,
        transformer: MotifVideoTransformer3DModel,
        guider: BaseGuidance,
        feature_extractor: Optional[SiglipImageProcessor] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            guider=guider,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 2
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer", None) is not None else 1
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 512
        )

    def _get_prompt_embeds(
        self,
        text_encoder: T5Gemma2Encoder,
        tokenizer: PreTrainedTokenizerBase,
        prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_inputs = BatchEncoding(
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in text_inputs.items()}
        )

        prompt_embeds = text_encoder(**text_inputs)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, text_inputs.attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] | None = None,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to be encoded.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            max_sequence_length (`int`, defaults to 512):
                Maximum sequence length for the tokenizer.
            device (`torch.device`, *optional*):
                Device to place tensors on.
            dtype (`torch.dtype`, *optional*):
                Data type for tensors.

        Returns:
            `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`:
                A tuple containing:
                - `prompt_embeds`: The text embeddings for the positive prompt
                - `negative_prompt_embeds`: The text embeddings for the negative prompt (None if not using guidance)
                - `prompt_attention_mask`: The attention mask for the positive prompt
                - `negative_prompt_attention_mask`: The attention mask for the negative prompt (None if not using
                  guidance)
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.bool()
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_videos_per_prompt, dim=0)

        # Compute negative embeddings if needed
        if negative_prompt_embeds is None and negative_prompt is not None:
            # Prepare negative_prompt to match batch_size
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            else:
                negative_prompt = list(negative_prompt)

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.bool()
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(batch_size, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat_interleave(
                num_videos_per_prompt, dim=0
            )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        batch_size,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % self.vae_scale_factor_spatial != 0 or width % self.vae_scale_factor_spatial != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor_spatial} but are {height} and {width}."
            )

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

        if negative_prompt is not None:
            if not isinstance(negative_prompt, (str, list)):
                raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
            if isinstance(negative_prompt, list) and len(negative_prompt) != batch_size:
                raise ValueError(
                    f"`negative_prompt` list length ({len(negative_prompt)}) must match batch_size ({batch_size})."
                )

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

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = torch.tensor(latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std
        return latents

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = torch.tensor(latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean
        return latents

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 736,
        width: int = 1280,
        num_frames: int = 121,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is None:
            shape = (
                batch_size,
                num_channels_latents,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents

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
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 736,
        width: int = 1280,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        vae_batch_size: int | None = None,
    ):
        r"""
        The call function to the pipeline for text-to-video generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance.
            height (`int`, defaults to `736`):
                The height in pixels of the generated video.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `121`):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                PyTorch Generator object(s) for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video. Choose between `"pil"`, `"np"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~MotifVideoPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Arguments passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function or subclass of `PipelineCallback` or `MultiPipelineCallbacks` called at the end of each
                denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, defaults to `512`):
                Maximum sequence length for the tokenizer.
            vae_batch_size (`int`, *optional*):
                Batch size for VAE decoding. If provided and latents batch size is larger, VAE decoding will be done in
                chunks.

        Examples:

        Returns:
            [`~MotifVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~MotifVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list of generated video frames.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 2. Check inputs
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            batch_size=batch_size,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        device = self._execution_device

        # 3. Prepare text embeddings
        # Ensure negative prompt is provided for multi-condition guiders
        if self.guider.num_conditions > 1 and negative_prompt_embeds is None and negative_prompt is None:
            negative_prompt = ""

        prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        )

        # 4. Prepare latents
        num_channels_latents = self.vae.config.z_dim
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            self.transformer.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        packed_latent_height = latent_height // self.transformer_spatial_patch_size
        packed_latent_width = latent_width // self.transformer_spatial_patch_size
        packed_latent_num_frames = latent_num_frames // self.transformer_temporal_patch_size
        video_sequence_length = packed_latent_num_frames * packed_latent_height * packed_latent_width

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )

        # Prepare conditioning tensors (T2V mode: no first-frame conditioning)
        batch_size, latent_channels, latent_num_frames, latent_height, latent_width = latents.shape
        latent_condition = torch.zeros(
            batch_size,
            latent_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            device=latents.device,
            dtype=latents.dtype,
        )
        latent_mask = torch.zeros(
            batch_size,
            1,
            latent_num_frames,
            latent_height,
            latent_width,
            device=latents.device,
            dtype=latents.dtype,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Concatenate current latents with conditioning: [latents | latent_condition | latent_mask]
                hidden_states = torch.cat([latents, latent_condition, latent_mask], dim=1)

                timestep = t.expand(latents.shape[0])

                # Guider: collect model inputs
                if self.guider.num_conditions == 1:
                    guider_inputs = {
                        "encoder_hidden_states": (prompt_embeds,),
                        "encoder_attention_mask": (prompt_attention_mask,),
                    }
                else:
                    guider_inputs = {
                        "encoder_hidden_states": (prompt_embeds, negative_prompt_embeds),
                        "encoder_attention_mask": (
                            prompt_attention_mask,
                            negative_prompt_attention_mask,
                        ),
                    }

                self.guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)
                guider_state = self.guider.prepare_inputs(guider_inputs)

                for guider_state_batch in guider_state:
                    self.guider.prepare_models(self.transformer)

                    cond_kwargs = {
                        input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()
                    }

                    context_name = getattr(guider_state_batch, self.guider._identifier_key)
                    with self.transformer.cache_context(context_name):
                        noise_pred = self.transformer(
                            hidden_states=hidden_states,
                            timestep=timestep,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                            **cond_kwargs,
                        )[0].clone()

                        guider_state_batch.noise_pred = noise_pred
                    self.guider.cleanup_models(self.transformer)

                noise_pred = self.guider(guider_state)[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    if "negative_prompt_embeds" in callback_outputs:
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds")

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            video = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = self._denormalize_latents(latents, self.vae.config.latents_mean, self.vae.config.latents_std)
            if vae_batch_size is not None and latents.shape[0] > vae_batch_size:
                video_chunks = []
                for i in range(0, latents.shape[0], vae_batch_size):
                    chunk = latents[i : i + vae_batch_size]
                    video_chunks.append(self.vae.decode(chunk, return_dict=False)[0])
                video = torch.cat(video_chunks, dim=0)
                del video_chunks
            else:
                video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return MotifVideoPipelineOutput(frames=video)
