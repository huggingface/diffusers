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

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer, GemmaTokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import FromSingleFileMixin, LTX2LoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...models.transformers import LTX2VideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .connectors import LTX2TextConnectors
from .image_processor import LTX2VideoHDRProcessor
from .pipeline_output import LTX2PipelineOutput
from .vocoder import LTX2Vocoder, LTX2VocoderWithBWE


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class LTX2HDRReferenceCondition:
    r"""
    A reference video condition for HDR IC-LoRA conditioning.

    The reference video is encoded into latent tokens and concatenated to the noisy latent sequence during denoising,
    allowing the HDR IC-LoRA adapter to condition the generation on the reference video content.

    Matches the `(video_path, strength)` tuples consumed by the reference `HDRICLoraPipeline`'s `video_conditioning`
    argument.

    Attributes:
        frames (`PIL.Image.Image` or `List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            The reference video frames. Accepts any type handled by `VideoProcessor.preprocess_video`.
        strength (`float`, defaults to `1.0`):
            Controls how "clean" the reference tokens appear to the model. A value of `1.0` means fully clean
            (per-token timestep=0), `0.0` means fully noisy.
    """

    frames: PIL.Image.Image | list[PIL.Image.Image] | np.ndarray | torch.Tensor
    strength: float = 1.0


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from safetensors import safe_open
        >>> from diffusers import LTX2HDRPipeline
        >>> from diffusers.pipelines.ltx2.pipeline_ltx2_hdr_lora import LTX2HDRReferenceCondition
        >>> from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        >>> from diffusers.pipelines.ltx2.export_utils import encode_hdr_tensor_to_mp4
        >>> from diffusers.utils import load_video

        >>> pipe = LTX2HDRPipeline.from_pretrained("dg845/LTX-2.3-Distilled-Diffusers", torch_dtype=torch.bfloat16)
        >>> pipe.enable_sequential_cpu_offload(device="cuda")
        >>> pipe.load_lora_weights(
        ...     "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
        ...     adapter_name="hdr_lora",
        ...     weight_name="ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
        ... )
        >>> pipe.set_adapters("hdr_lora", 1.0)

        >>> reference_video = load_video("/path/to/reference.mp4")
        >>> ref_cond = LTX2HDRReferenceCondition(frames=reference_video, strength=1.0)

        >>> # Load pre-computed HDR LoRA connector embeddings.
        >>> with safe_open("/path/to/connector/embeds.safetensors", framework="pt", device="cuda") as f:
        ...     connector_video_embeds = f.get_tensor("video_context")
        ...     connector_audio_embeds = f.get_tensor("audio_context")

        >>> # `hdr_video` is a linear HDR tensor of shape (batch, frames, H, W, C).
        >>> hdr_video = pipe(
        ...     reference_conditions=[ref_cond],
        ...     connector_video_embeds=connector_video_embeds,
        ...     connector_audio_embeds=connector_audio_embeds,
        ...     width=768,
        ...     height=512,
        ...     num_frames=121,
        ...     frame_rate=24.0,
        ...     num_inference_steps=8,
        ...     sigmas=DISTILLED_SIGMA_VALUES,
        ...     guidance_scale=1.0,
        ...     output_type="pt",
        ...     return_dict=False,
        ... )[0]

        >>> # Convert the HDR video to a SDR sRGB-tonemapped `.mp4` video.
        >>> # A custom tone-mapper can be specified via the `tone_mapping_fn` argument.
        >>> encode_hdr_tensor_to_mp4(hdr_video[0], "ltx2_hdr_lora_output.mp4", frame_rate=24.0)
        ```
"""


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


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


class LTX2HDRPipeline(DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin):
    r"""
    Pipeline for LTX-2.X HDR video generation with reference video conditioning.

    The pipeline accepts a reference SDR ("normal") video and generates a linear HDR output with values in `[0, ∞)` via
    a LogC3 inverse transform which has the same content as the reference video. The motivating use case for this
    pipeline is to support LTX-2.X HDR IC-LoRAs, but it should support any LTX-2.X-like model that operates on HDR
    inputs as above.

    Compared to [`LTX2InContextPipeline`], the HDR pipeline has the following differences:

    - Video-only (no audio output). The transformer's audio branch is still run since the diffusers transformer API
      requires audio inputs, but the decoded audio is discarded and audio-specific guidance scales are fixed to no-op
      values to avoid wasted compute.
    - No frame-level keyframe conditioning (the reference HDR pipeline does not support this).

    Two-stage inference is supported through separate calls to `__call__`:

    - **Stage 1**: generate video latents at target resolution with HDR IC-LoRA conditioning (`output_type="latent"`).
    - **Stage 2**: upsample via [`LTX2LatentUpsamplePipeline`] and refine with this same pipeline (or [`LTX2Pipeline`])
      by passing `latents=upsampled_latents`. The reference HDR stage-2 additionally supports spatial/temporal tiling
      of the refinement pass — that optimization is not yet implemented here.

    Reference: https://github.com/Lightricks/LTX-2 Paper: https://huggingface.co/papers/2604.11788

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler used in the denoising loop.
        vae ([`AutoencoderKLLTX2Video`]):
            Video VAE.
        audio_vae ([`AutoencoderKLLTX2Audio`]):
            Audio VAE. Required for transformer compatibility; its outputs are discarded.
        text_encoder ([`transformers.Gemma3ForConditionalGeneration`]):
            Text encoder.
        tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`):
            Tokenizer for the text encoder.
        connectors ([`LTX2TextConnectors`]):
            Text connector stack for the transformer.
        transformer ([`LTX2VideoTransformer3DModel`]):
            Transformer backbone.
        vocoder ([`LTX2Vocoder`] or [`LTX2VocoderWithBWE`]):
            Vocoder. Required for transformer compatibility; its outputs are discarded.
        hdr_transform (`str`, *optional*, defaults to `"logc3"`):
            HDR transform identifier applied during postprocessing. Currently only `"logc3"` is supported.
    """

    model_cpu_offload_seq = "text_encoder->connectors->transformer->vae->audio_vae->vocoder"
    _optional_components = ["audio_scheduler"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Gemma3ForConditionalGeneration,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        connectors: LTX2TextConnectors,
        transformer: LTX2VideoTransformer3DModel,
        vocoder: LTX2Vocoder | LTX2VocoderWithBWE,
        audio_scheduler: FlowMatchEulerDiscreteScheduler | None = None,
        hdr_transform: str = "logc3",
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=transformer,
            vocoder=vocoder,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer") is not None else 1
        )

        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        )
        self.audio_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        self.audio_latent_channels = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        )

        self.hdr_video_processor = LTX2VideoHDRProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio,
            hdr_transform=hdr_transform,
        )

        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 1024
        )
        tokenizer_padding_side = "left"
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
        self.tokenizer_padding_side = tokenizer_padding_side

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._get_gemma_prompt_embeds
    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`str` or `torch.device`):
                torch device to place the resulting embeddings on
            dtype: (`torch.dtype`):
                torch dtype to cast the prompt embeds to
            max_sequence_length (`int`, defaults to 1024): Maximum sequence length to use for the prompt.
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if getattr(self, "tokenizer", None) is not None:
            # Gemma expects left padding for chat-style prompts
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
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
        text_input_ids = text_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(dtype=dtype)  # Pack to 3D

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
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
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
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

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
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
        connector_video_embeds=None,
        connector_audio_embeds=None,
        latents=None,
        spatio_temporal_guidance_blocks=None,
        stg_scale=None,
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
            if connector_video_embeds is None or connector_audio_embeds is None:
                raise ValueError(
                    "Provide a `prompt`, `prompt_embeds` or `connector_video_embeds` and `connector_audio_embeds`"
                )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if latents is not None and latents.ndim != 5:
            raise ValueError(
                f"Only unpacked (5D) video latents of shape `[batch_size, latent_channels, latent_frames,"
                f" latent_height, latent_width] are supported, but got {latents.ndim} dims."
            )

        if (stg_scale is not None and stg_scale > 0.0) and not spatio_temporal_guidance_blocks:
            raise ValueError(
                "Spatio-Temporal Guidance (STG) is specified but no STG blocks are supplied. Please supply a list of"
                " block indices at which to apply STG in `spatio_temporal_guidance_blocks`"
            )

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._pack_latents
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
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_latents
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
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_image2video.LTX2ImageToVideoPipeline._normalize_latents
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._create_noised_state
    def _create_noised_state(
        latents: torch.Tensor, noise_scale: float | torch.Tensor, generator: torch.Generator | None = None
    ):
        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        noised_latents = noise_scale * noise + (1 - noise_scale) * latents
        return noised_latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._normalize_audio_latents
    def _normalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) / latents_std

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_audio_latents
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._pack_audio_latents
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
    ) -> torch.Tensor:
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        if patch_size is not None and patch_size_t is not None:
            # Packs the latents into a patch sequence of shape [B, L // p_t * M // p, C * p_t * p] (a ndim=3 tnesor).
            # dim=1 is the effective audio sequence length and dim=2 is the effective audio input feature size.
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length / patch_size_t
            post_patch_mel_bins = latent_mel_bins / patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        else:
            # Packs the latents into a patch sequence of shape [B, L, C * M]. This implicitly assumes a (mel)
            # patch_size of M (all mel bins constitutes a single patch) and a patch_size_t of 1.
            latents = latents.transpose(1, 2).flatten(2, 3)  # [B, C, L, M] --> [B, L, C * M]
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_audio_latents
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> torch.Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    def prepare_latents(
        self,
        reference_conditions: list[LTX2HDRReferenceCondition] | None = None,
        reference_downscale_factor: int = 1,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, torch.Tensor | None]:
        r"""
        Prepare noisy video latents, applying HDR IC-LoRA reference-video conditioning.

        Builds a packed latent sequence in the order `[base | reference]`:
          - Base: either fresh noise (Stage 1, `latents=None`) or pre-existing upsampled latents (Stage 2).
          - Reference: HDR-encoded reference-video tokens appended with per-token `conditioning_mask = strength`,
            following the same pattern as [`LTX2InContextPipeline.prepare_latents`]. (HDR LoRA does not currently take
            per-frame `conditions`, so there is no first-frame / keyframe block in between.)

        Returns a 6-tuple matching [`LTX2InContextPipeline.prepare_latents`]:
            - `latents`: packed noisy latents `(B, base + n_ref, C)`.
            - `conditioning_mask`: `(B, seq_len, 1)` with `strength` at reference positions, `0` elsewhere.
            - `clean_latents`: clean reference values at reference positions (zeros elsewhere); same shape as
              `latents`.
            - `appended_coords`: `[1, 3, n_ref, 2]` reference coordinates to concat onto `video_coords`, or `None` when
              no reference conditions are provided.
            - `num_ref_tokens`: count of reference tokens at the END of `latents`.
            - `ref_cross_mask`: always `None` for HDR LoRA (no cross-attention masking support).
        """
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective"
                    f" batch size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

        # Build the base noisy latents at the maximum sigma (zeros for Stage 1 fresh noise; normalized provided latents
        # for Stage 2). The noise mixing at the bottom converts these into the right partial-denoise state.
        if latents is not None:
            if latents.ndim == 5:
                latents = self._normalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )
            if latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is [batch_size,"
                    f" num_seq, num_features]."
                )
        else:
            shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
            latents = torch.zeros(shape, device=device, dtype=dtype)
            latents = self._pack_latents(
                latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )
        latents = latents.to(device=device, dtype=dtype)

        # Build conditioning_mask and clean_latents over the base token sequence (zeros — base is unconditioned).
        base_seq_len = latents.shape[1]
        conditioning_mask = torch.zeros((batch_size, base_seq_len, 1), device=device, dtype=dtype)
        clean_latents = torch.zeros_like(latents)

        # Append reference tokens (if any) as a contiguous block at the end of the sequence with per-token
        # `conditioning_mask = strength` and `clean_latents = encoded_ref`.
        ref_coords: torch.Tensor | None = None
        num_ref_tokens = 0
        if reference_conditions is not None and len(reference_conditions) > 0:
            ref_latents_packed, ref_coords, _ = self._encode_reference_conditions(
                reference_conditions=reference_conditions,
                num_frames=num_frames,
                height=height,
                width=width,
                reference_downscale_factor=reference_downscale_factor,
                frame_rate=frame_rate,
                dtype=dtype,
                device=device,
                generator=generator[0] if isinstance(generator, list) else generator,
            )
            num_ref_tokens = ref_latents_packed.shape[1]

            # All reference videos preprocess to the same shape, so split tokens evenly across conditions.
            n_per_ref = num_ref_tokens // len(reference_conditions)
            ref_mask_chunks = [
                torch.full(
                    (batch_size, n_per_ref, 1),
                    float(ref_cond.strength),
                    device=device,
                    dtype=conditioning_mask.dtype,
                )
                for ref_cond in reference_conditions
            ]
            ref_mask_full = torch.cat(ref_mask_chunks, dim=1)

            ref_latents_packed_b = ref_latents_packed.expand(batch_size, -1, -1)
            latents = torch.cat([latents, ref_latents_packed_b], dim=1)
            conditioning_mask = torch.cat([conditioning_mask, ref_mask_full], dim=1)
            clean_latents = torch.cat([clean_latents, ref_latents_packed_b], dim=1)

        # HDR LoRA has no keyframe conditions, so the only appended tokens are reference tokens.
        appended_coords = ref_coords

        # The conditioning_mask values have the following semantics:
        #   - mask=0: fully noise tokens (e.g. noisy latents)
        #   - mask=1: keep fully clean (e.g. I2V first-frame condition, conditions with strength=1)
        #   - mask in (0, 1): use intermediate noise level mask * sigma_i (noise_scale == sigma_0)
        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        scaled_mask = (1.0 - conditioning_mask) * noise_scale  # noise to initial noise level `noise_scale`
        latents = noise * scaled_mask + latents * (1 - scaled_mask)

        return latents, conditioning_mask, clean_latents, appended_coords, num_ref_tokens, None

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.prepare_audio_latents
    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,  # 1 is just a dummy value
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            # latents expected to be unpacked (4D) with shape [B, C, L, M]
            latents = self._pack_audio_latents(latents)
            latents = self._normalize_audio_latents(latents, self.audio_vae.latents_mean, self.audio_vae.latents_std)
            latents = self._create_noised_state(latents, noise_scale, generator)
            return latents.to(device=device, dtype=dtype)

        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Sample in packed shape (B, L, C * M), following the original LTX-2.X code
        packed_shape = (batch_size, audio_latent_length, num_channels_latents * latent_mel_bins)
        latents = randn_tensor(packed_shape, generator=generator, device=device, dtype=dtype)
        return latents

    def _encode_reference_conditions(
        self,
        reference_conditions: list[LTX2HDRReferenceCondition],
        height: int,
        width: int,
        num_frames: int,
        reference_downscale_factor: int = 1,
        frame_rate: float = 24.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode HDR IC-LoRA reference videos into `(reference_latents, reference_coords, reference_cross_mask)`.

        Shared encoding core used by both `prepare_latents` (which folds reference tokens into the main noisy sequence)
        and the back-compat shim `prepare_reference_latents`. HDR LoRA does not currently support cross-attention
        masking for reference tokens, so the third return is always `None`.
        """
        ref_height = height // reference_downscale_factor
        ref_width = width // reference_downscale_factor

        if reference_downscale_factor != 1 and (
            height % reference_downscale_factor != 0 or width % reference_downscale_factor != 0
        ):
            raise ValueError(
                f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor "
                f"({reference_downscale_factor})."
            )

        all_ref_latents = []
        all_ref_coords = []

        for ref_cond in reference_conditions:
            if isinstance(ref_cond.frames, PIL.Image.Image):
                video_like = [ref_cond.frames]
            elif isinstance(ref_cond.frames, np.ndarray) and ref_cond.frames.ndim == 3:
                video_like = np.expand_dims(ref_cond.frames, axis=0)
            elif isinstance(ref_cond.frames, torch.Tensor) and ref_cond.frames.ndim == 3:
                video_like = ref_cond.frames.unsqueeze(0)
            else:
                video_like = ref_cond.frames

            # HDR-specific preprocessing: reflect-pad resize (vs center-crop in the standard IC-LoRA pipeline).
            # For LDR reference videos the numerical output of `preprocess_reference_video_hdr` is identical to the
            # standard [-1, 1] normalization since LogC3's `compress_ldr` is an identity clamp.
            ref_pixels = self.hdr_video_processor.preprocess_reference_video_hdr(video_like, ref_height, ref_width)
            ref_pixels = ref_pixels[:, :, :num_frames]
            ref_pixels = ref_pixels.to(dtype=self.vae.dtype, device=device)

            ref_latent = retrieve_latents(self.vae.encode(ref_pixels), generator=generator, sample_mode="argmax")
            ref_latent = self._normalize_latents(ref_latent, self.vae.latents_mean, self.vae.latents_std).to(
                device=device, dtype=dtype
            )

            _, _, ref_latent_frames, ref_latent_height, ref_latent_width = ref_latent.shape

            ref_latent_packed = self._pack_latents(
                ref_latent, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )

            ref_coords = self.transformer.rope.prepare_video_coords(
                batch_size=1,
                num_frames=ref_latent_frames,
                height=ref_latent_height,
                width=ref_latent_width,
                device=device,
                fps=frame_rate,
            )
            if reference_downscale_factor != 1:
                ref_coords[:, 1, :, :] = ref_coords[:, 1, :, :] * reference_downscale_factor
                ref_coords[:, 2, :, :] = ref_coords[:, 2, :, :] * reference_downscale_factor

            all_ref_latents.append(ref_latent_packed)
            all_ref_coords.append(ref_coords)

        reference_latents = torch.cat(all_ref_latents, dim=1)
        reference_coords = torch.cat(all_ref_coords, dim=2)

        return reference_latents, reference_coords, None

    def prepare_reference_latents(
        self,
        reference_conditions: list[LTX2HDRReferenceCondition],
        height: int,
        width: int,
        num_frames: int,
        reference_downscale_factor: int = 1,
        frame_rate: float = 24.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encode reference videos with HDR preprocessing into packed latent tokens and compute positional coordinates.

        Each reference video is preprocessed via [`LTX2VideoHDRProcessor.preprocess_reference_video_hdr`] (reflect-pad
        resize at the reference resolution), VAE-encoded, packed into tokens, and paired with positional coordinates
        computed at the reference latent dimensions and scaled by `reference_downscale_factor`.

        Returns a 3-tuple `(reference_latents, reference_coords, reference_denoise_factors)` with the same shapes as
        [`LTX2InContextPipeline.prepare_reference_latents`].
        """
        reference_latents, reference_coords, _ = self._encode_reference_conditions(
            reference_conditions=reference_conditions,
            height=height,
            width=width,
            num_frames=num_frames,
            reference_downscale_factor=reference_downscale_factor,
            frame_rate=frame_rate,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Materialize per-token denoise factors for callers that still expect the 3-tuple. Each ref video has
        # `1 - strength` for all of its tokens; we rebuild this from the per-video token counts. All ref videos
        # preprocess to the same shape, so total token count divides equally across them.
        n_total = reference_latents.shape[1]
        n_per_ref = n_total // max(len(reference_conditions), 1)
        denoise_chunks = [
            torch.full((1, n_per_ref), 1.0 - ref_cond.strength, device=reference_latents.device, dtype=torch.float32)
            for ref_cond in reference_conditions
        ]
        reference_denoise_factors = (
            torch.cat(denoise_chunks, dim=1) if denoise_chunks else reference_latents.new_zeros((1, 0))
        )
        return reference_latents, reference_coords, reference_denoise_factors

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.convert_velocity_to_x0
    def convert_velocity_to_x0(
        self, sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler: Any | None = None
    ) -> torch.Tensor:
        if scheduler is None:
            scheduler = self.scheduler

        sample_x0 = sample - denoised_output * scheduler.sigmas[step_idx]
        return sample_x0

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.convert_x0_to_velocity
    def convert_x0_to_velocity(
        self, sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler: Any | None = None
    ) -> torch.Tensor:
        if scheduler is None:
            scheduler = self.scheduler

        sample_v = (sample - denoised_output) / scheduler.sigmas[step_idx]
        return sample_v

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def stg_scale(self):
        return self._stg_scale

    @property
    def modality_scale(self):
        return self._modality_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def do_spatio_temporal_guidance(self):
        return self._stg_scale > 0.0

    @property
    def do_modality_isolation_guidance(self):
        return self._modality_scale > 1.0

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
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] | None = None,
        reference_conditions: LTX2HDRReferenceCondition | list[LTX2HDRReferenceCondition] | None = None,
        reference_downscale_factor: int = 1,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 8,
        sigmas: list[float] | None = None,
        timesteps: list[float] | None = None,
        guidance_scale: float = 1.0,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        noise_scale: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        connector_video_embeds: torch.Tensor | None = None,
        connector_audio_embeds: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        use_cross_timestep: bool = False,
        output_type: str = "pt",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ):
        r"""
        Run HDR IC-LoRA video generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt(s) to guide generation. Either `prompt` or `prompt_embeds` must be supplied.
            negative_prompt (`str` or `List[str]`, *optional*):
                The negative prompt(s). Ignored when `guidance_scale <= 1.0`.
            reference_conditions (`LTX2HDRReferenceCondition` or `List[LTX2HDRReferenceCondition]`, *optional*):
                Reference video conditions for HDR IC-LoRA conditioning.
            reference_downscale_factor (`int`, *optional*, defaults to `1`):
                Ratio between target and reference video resolutions. IC-LoRA models trained with downscaled reference
                videos store this factor in their safetensors metadata.
            height (`int`, *optional*, defaults to `512`):
                Output video height in pixels. Must be divisible by 32.
            width (`int`, *optional*, defaults to `768`):
                Output video width in pixels. Must be divisible by 32.
            num_frames (`int`, *optional*, defaults to `121`):
                Number of frames to generate. Must satisfy `(n - 1) % 8 == 0`.
            frame_rate (`float`, *optional*, defaults to `24.0`):
                Output frame rate (used for temporal positional encoding).
            num_inference_steps (`int`, *optional*, defaults to `8`):
                Number of denoising steps. Default matches the distilled model schedule.
            sigmas (`List[float]`, *optional*):
                Custom sigma schedule. Overrides `num_inference_steps` when set.
            timesteps (`List[float]`, *optional*):
                Custom timesteps schedule. Overrides `num_inference_steps` when set.
            guidance_scale (`float`, *optional*, defaults to `1.0`):
                Classifier-Free Guidance scale for video. Default `1.0` disables CFG (matches the distilled model).
            stg_scale (`float`, *optional*, defaults to `0.0`):
                Spatio-Temporal Guidance scale for video.
            modality_scale (`float`, *optional*, defaults to `1.0`):
                Modality isolation guidance scale for video.
            guidance_rescale (`float`, *optional*, defaults to `0.0`):
                Video guidance rescale factor.
            spatio_temporal_guidance_blocks (`list[int]`, *optional*):
                Transformer block indices at which to apply STG.
            noise_scale (`float`, *optional*):
                Noise scale used when preparing the initial latents. Inferred from the sigma schedule when unset.
            num_videos_per_prompt (`int`, *optional*, defaults to `1`):
                Number of videos to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator(s) for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video latents. Pass output from [`LTX2LatentUpsamplePipeline`] here for Stage 2.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Bypasses `prompt`/`tokenizer`/`text_encoder` if supplied.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for `negative_prompt_embeds`.
            connector_video_embeds (`torch.Tensor`, *optional*):
                Optional pre-computed connector outputs for the video modality. Used by the HDR LoRA pipeline; if
                supplied, will override any `prompt`/`prompt_embeds`.
            connector_audio_embeds (`torch.Tensor`, *optional*):
                Optional pre-computed connector outputs for the audio modality. Used by the HDR LoRA pipeline; if
                supplied, will override any `prompt`/`prompt_embeds`.
            decode_timestep, decode_noise_scale:
                VAE-decode timestep conditioning (only used by VAE configs with `timestep_conditioning=True`).
            use_cross_timestep (`bool`, *optional*, defaults to `False`):
                Whether to use cross-modality sigma for cross-attention modulation.
            output_type (`str`, *optional*, defaults to `"pt"`):
                One of `"pt"`, `"np"`, or `"latent"`. `"pt"` returns a linear HDR torch tensor in `[0, ∞)` of shape
                `(batch_size, num_frames, height, width, channels)`; `"np"` returns the equivalent `float32` NumPy
                array; `"latent"` returns the raw denoised latents (skip the HDR decode).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`LTX2PipelineOutput`] instead of a plain tuple.
            attention_kwargs, callback_on_step_end, callback_on_step_end_tensor_inputs, max_sequence_length:
                Standard hooks and arguments, same as [`LTX2InContextPipeline`].

        Examples:

        Returns:
            [`LTX2PipelineOutput`] or `tuple`. When `return_dict=False`, returns `(frames, None)` — the audio slot is
            always `None` since this pipeline is video-only.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            connector_video_embeds=connector_video_embeds,
            connector_audio_embeds=connector_audio_embeds,
            latents=latents,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
            stg_scale=stg_scale,
        )

        # Video-only guidance state.
        self._guidance_scale = guidance_scale
        self._stg_scale = stg_scale
        self._modality_scale = modality_scale
        self._guidance_rescale = guidance_rescale

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = connector_video_embeds.shape[0]

        if reference_conditions is not None and not isinstance(reference_conditions, list):
            reference_conditions = [reference_conditions]

        if noise_scale is None:
            noise_scale = sigmas[0] if sigmas is not None else 1.0

        device = self._execution_device

        # 3. Prepare text embeddings
        if connector_video_embeds is None or connector_audio_embeds is None:
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

            connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
                prompt_embeds, prompt_attention_mask, padding_side=self.tokenizer_padding_side
            )
        else:
            connector_prompt_embeds = connector_video_embeds.to(device=device, dtype=self.transformer.dtype)
            connector_audio_prompt_embeds = connector_audio_embeds.to(device=device, dtype=self.transformer.dtype)
            connector_attention_mask = None

        # 4. Prepare video latents
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None:
            logger.info(
                "Got pre-supplied latents of shape %s; `latent_num_frames`, `latent_height`, and `latent_width` will"
                " be inferred.",
                tuple(latents.shape),
            )
            _, _, latent_num_frames, latent_height, latent_width = latents.shape

        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, clean_latents, appended_coords, num_ref_tokens, _ = self.prepare_latents(
            reference_conditions=reference_conditions,
            reference_downscale_factor=reference_downscale_factor,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        # Track the base (non-reference) token count so we can trim the appended reference tokens off
        # `latents` before unpack/decode at the end.
        base_token_count = latents.shape[1] - num_ref_tokens
        if self.do_classifier_free_guidance and num_ref_tokens > 0:
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])

        # 5. Prepare audio latents. Audio is discarded at the end, but the transformer's audio branch still runs so
        # we need well-formed audio inputs. Audio guidance is fixed so no extra audio-only forward passes fire.
        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)

        audio_latents = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents=self.audio_latent_channels,
            audio_latent_length=audio_num_frames,
            num_mel_bins=self.audio_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
        )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        if self.audio_scheduler is not None:
            audio_scheduler = self.audio_scheduler
        else:
            audio_scheduler = copy.deepcopy(self.scheduler)
        audio_timesteps, _ = retrieve_timesteps(
            audio_scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 7. Prepare positional coordinates
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate
        )
        if appended_coords is not None:
            # Expand appended_coords to effective batch size (to [B, 3, num_extra_tokens, 2])
            appended_coords = appended_coords.expand(latents.shape[0], -1, -1, -1)
            video_coords = torch.cat([video_coords, appended_coords], dim=2)
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device
        )
        if self.do_classifier_free_guidance:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        # 8. Denoising loop
        video_seq_len = latents.shape[1]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(connector_prompt_embeds.dtype)
                audio_latent_model_input = (
                    torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                )
                audio_latent_model_input = audio_latent_model_input.to(connector_prompt_embeds.dtype)

                timestep_scalar = t.expand(latent_model_input.shape[0])
                if num_ref_tokens > 0:
                    video_timestep = timestep_scalar.unsqueeze(-1) * (1 - conditioning_mask.squeeze(-1))
                else:
                    video_timestep = timestep_scalar.unsqueeze(-1).expand(-1, video_seq_len)

                t_audio = audio_timesteps[i]
                audio_timestep = t_audio.expand(latent_model_input.shape[0])

                # --- Main forward pass (cond + uncond for CFG) ---
                with self.transformer.cache_context("cond_uncond"):
                    noise_pred_video, noise_pred_audio = self.transformer(
                        hidden_states=latent_model_input,
                        audio_hidden_states=audio_latent_model_input,
                        encoder_hidden_states=connector_prompt_embeds,
                        audio_encoder_hidden_states=connector_audio_prompt_embeds,
                        timestep=video_timestep,
                        audio_timestep=audio_timestep,
                        sigma=timestep_scalar,  # Used by LTX-2.3
                        audio_sigma=audio_timestep,
                        encoder_attention_mask=connector_attention_mask,
                        audio_encoder_attention_mask=connector_attention_mask,
                        video_self_attention_mask=None,
                        num_frames=latent_num_frames,
                        height=latent_height,
                        width=latent_width,
                        fps=frame_rate,
                        audio_num_frames=audio_num_frames,
                        video_coords=video_coords,
                        audio_coords=audio_coords,
                        isolate_modalities=False,
                        spatio_temporal_guidance_blocks=None,
                        perturbation_mask=None,
                        use_cross_timestep=use_cross_timestep,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )
                noise_pred_video = noise_pred_video.float()

                if self.do_classifier_free_guidance:
                    noise_pred_video_uncond_text, noise_pred_video = noise_pred_video.chunk(2)
                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)
                    noise_pred_video_uncond_text = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_text, i, self.scheduler
                    )
                    video_cfg_delta = (self.guidance_scale - 1) * (noise_pred_video - noise_pred_video_uncond_text)

                    if self.do_spatio_temporal_guidance or self.do_modality_isolation_guidance:
                        if i == 0:
                            video_prompt_embeds = connector_prompt_embeds.chunk(2, dim=0)[1]
                            audio_prompt_embeds = connector_audio_prompt_embeds.chunk(2, dim=0)[1]
                            prompt_attn_mask = connector_attention_mask.chunk(2, dim=0)[1]
                            video_pos_ids = video_coords.chunk(2, dim=0)[0]
                            audio_pos_ids = audio_coords.chunk(2, dim=0)[0]
                        timestep_scalar_single = timestep_scalar.chunk(2, dim=0)[0]
                        if num_ref_tokens > 0:
                            video_timestep_single = video_timestep.chunk(2, dim=0)[0]
                        else:
                            video_timestep_single = timestep_scalar_single.unsqueeze(-1).expand(-1, video_seq_len)
                        audio_timestep_single = audio_timestep.chunk(2, dim=0)[0]
                else:
                    video_cfg_delta = 0

                    video_prompt_embeds = connector_prompt_embeds
                    audio_prompt_embeds = connector_audio_prompt_embeds
                    prompt_attn_mask = connector_attention_mask
                    video_pos_ids = video_coords
                    audio_pos_ids = audio_coords

                    timestep_scalar_single = timestep_scalar
                    if num_ref_tokens > 0:
                        video_timestep_single = video_timestep
                    else:
                        video_timestep_single = timestep_scalar.unsqueeze(-1).expand(-1, video_seq_len)
                    audio_timestep_single = audio_timestep

                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)

                # --- STG forward pass (video only — audio output discarded) ---
                if self.do_spatio_temporal_guidance:
                    with self.transformer.cache_context("uncond_stg"):
                        noise_pred_video_uncond_stg, noise_pred_audio_uncond_stg = self.transformer(
                            hidden_states=latents.to(dtype=connector_prompt_embeds.dtype),
                            audio_hidden_states=audio_latents.to(dtype=connector_prompt_embeds.dtype),
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep_single,
                            audio_timestep=audio_timestep_single,
                            sigma=timestep_scalar_single,  # Used by LTX-2.3
                            audio_sigma=audio_timestep_single,
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            video_self_attention_mask=None,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_pos_ids,
                            audio_coords=audio_pos_ids,
                            isolate_modalities=False,
                            # Use STG at given blocks to perturb model
                            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                            perturbation_mask=None,
                            use_cross_timestep=use_cross_timestep,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )
                    noise_pred_video_uncond_stg = noise_pred_video_uncond_stg.float()
                    noise_pred_video_uncond_stg = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_stg, i, self.scheduler
                    )
                    video_stg_delta = self.stg_scale * (noise_pred_video - noise_pred_video_uncond_stg)
                else:
                    video_stg_delta = 0

                # --- Modality isolation guidance forward pass ---
                if self.do_modality_isolation_guidance:
                    with self.transformer.cache_context("uncond_modality"):
                        noise_pred_video_uncond_mod, noise_pred_audio_uncond_mod = self.transformer(
                            hidden_states=latents.to(dtype=connector_prompt_embeds.dtype),
                            audio_hidden_states=audio_latents.to(dtype=connector_prompt_embeds.dtype),
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep_single,
                            audio_timestep=audio_timestep_single,
                            sigma=timestep_scalar_single,  # Used by LTX-2.3
                            audio_sigma=audio_timestep_single,
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            video_self_attention_mask=None,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_pos_ids,
                            audio_coords=audio_pos_ids,
                            # Turn off A2V and V2A cross attn to isolate video and audio modalities
                            isolate_modalities=True,
                            spatio_temporal_guidance_blocks=None,
                            perturbation_mask=None,
                            use_cross_timestep=use_cross_timestep,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )
                    noise_pred_video_uncond_mod = noise_pred_video_uncond_mod.float()
                    noise_pred_video_uncond_mod = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_mod, i, self.scheduler
                    )
                    video_modality_delta = (self.modality_scale - 1) * (noise_pred_video - noise_pred_video_uncond_mod)
                else:
                    video_modality_delta = 0

                noise_pred_video_g = noise_pred_video + video_cfg_delta + video_stg_delta + video_modality_delta

                if self.guidance_rescale > 0:
                    noise_pred_video = rescale_noise_cfg(
                        noise_pred_video_g, noise_pred_video, guidance_rescale=self.guidance_rescale
                    )
                else:
                    noise_pred_video = noise_pred_video_g

                # Apply the conditioning mask to apply the reference conditions at the specified strength.
                if num_ref_tokens > 0:
                    bsz = noise_pred_video.size(0)
                    denoised_sample_cond = (
                        noise_pred_video * (1 - conditioning_mask[:bsz])
                        + clean_latents.float() * conditioning_mask[:bsz]
                    ).to(noise_pred_video.dtype)
                    noise_pred_video = denoised_sample_cond

                noise_pred_video = self.convert_x0_to_velocity(latents, noise_pred_video, i, self.scheduler)

                latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
                # Step the audio scheduler so its internal state stays in sync with the video scheduler (audio
                # output is discarded at the end, but keeping schedulers aligned avoids surprising behavior if the
                # scheduler writes internal indices during `.step()`).
                _ = audio_scheduler.step(torch.zeros_like(audio_latents), t, audio_latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 9. Decode
        # Trim any appended reference tokens from the latents to recover the generated video only.
        latents = latents[:, :base_token_count]
        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if output_type == "latent":
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            video = latents
        else:
            latents = latents.to(connector_prompt_embeds.dtype)

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

            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            latents = latents.to(self.vae.dtype)

            # VAE decode returns a video tensor in the VAE's native range ([-1, 1]).
            decoded = self.vae.decode(latents, timestep, return_dict=False)[0]
            # HDR postprocess: LogC3 decompress → linear HDR [0, ∞). Always float32 for HDR fidelity.
            video = self.hdr_video_processor.postprocess_hdr_video(decoded, output_type=output_type)

        # Audio is always None for this video-only pipeline.
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, None)

        return LTX2PipelineOutput(frames=video, audio=None)
