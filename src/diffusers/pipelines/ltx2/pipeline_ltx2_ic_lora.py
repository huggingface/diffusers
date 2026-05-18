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
import math
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
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .connectors import LTX2TextConnectors
from .pipeline_ltx2_condition import LTX2VideoCondition
from .pipeline_output import LTX2PipelineOutput
from .vocoder import LTX2Vocoder, LTX2VocoderWithBWE


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class LTX2ReferenceCondition:
    """
    A reference video condition for IC-LoRA (In-Context LoRA) conditioning.

    The reference video is encoded into latent tokens and concatenated to the noisy latent sequence during denoising.
    The transformer attends to these extra tokens, allowing the IC-LoRA adapter to condition the generation on the
    reference video content (e.g. style, structure, depth, pose).

    Attributes:
        frames (`PIL.Image.Image` or `List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            The reference video frames. Accepts any type handled by `VideoProcessor.preprocess_video`.
        strength (`float`, defaults to `1.0`):
            Controls how "clean" the reference tokens appear to the model. A value of `1.0` means fully clean
            (timestep=0 for reference tokens), `0.0` means fully noisy (same as denoising tokens).
    """

    frames: PIL.Image.Image | list[PIL.Image.Image] | np.ndarray | torch.Tensor
    strength: float = 1.0


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import LTX2InContextPipeline
        >>> from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
        >>> from diffusers.pipelines.ltx2.export_utils import encode_video
        >>> from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
        >>> from diffusers.utils import load_video

        >>> pipe = LTX2InContextPipeline.from_pretrained("dg845/LTX-2.3-Diffusers", torch_dtype=torch.bfloat16)
        >>> pipe.enable_sequential_cpu_offload(device="cuda")
        >>> pipe.load_lora_weights(
        ...     "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
        ...     adapter_name="ic_lora",
        ...     weight_name="ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        ... )
        >>> pipe.set_adapters("ic_lora", 1.0)

        >>> # If the IC LoRA uses reference conditions, you can specify them as follows:
        >>> # reference_video = load_video("reference.mp4")
        >>> # ref_cond = LTX2ReferenceCondition(frames=reference_video, strength=1.0)

        >>> prompt = "A flowing river in a forest"
        >>> frame_rate = 24.0
        >>> video, audio = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        ...     # reference_conditions=[ref_cond],
        ...     width=768,
        ...     height=512,
        ...     num_frames=121,
        ...     frame_rate=frame_rate,
        ...     num_inference_steps=30,
        ...     guidance_scale=3.0,
        ...     output_type="np",
        ...     return_dict=False,
        ... )

        >>> encode_video(
        ...     video[0],
        ...     fps=frame_rate,
        ...     audio=audio[0].float().cpu(),
        ...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        ...     output_path="ic_lora_output.mp4",
        ... )
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
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


class LTX2InContextPipeline(DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin):
    r"""
    Pipeline for LTX-2.X models with in-context (IC) conditioning. Also supports frame-level image conditions like
    `LTX2ConditionPipeline`; both frame and reference conditions can be used together.

    In-context conditioning works by conditioning the generation on a reference video by encoding it into latent tokens
    and concatenating them to the noisy latent tokens during denoising. The motivating use case is to support LTX-2.X
    IC LoRAs, which may use reference conditions (e.g. a pose video for pose control) to guide generation, but this
    pipeline is designed to work with any LTX-2.X-like model trained with in-context reference conditions.

    Two-stage inference is supported through separate calls to `__call__`:
    - **Stage 1**: Generate at target resolution with IC-LoRA conditioning (`output_type="latent"`).
    - **Stage 2**: Upsample via [`LTX2LatentUpsamplePipeline`], then refine with a distilled LoRA (no IC-LoRA reference
      conditioning needed for Stage 2).

    Reference: https://github.com/Lightricks/LTX-Video

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLLTX2Video`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        audio_vae ([`AutoencoderKLLTX2Audio`]):
            Audio VAE to encode and decode audio spectrograms.
        text_encoder ([`Gemma3ForConditionalGeneration`]):
            Text encoder model.
        tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`):
            Tokenizer for the text encoder.
        connectors ([`LTX2TextConnectors`]):
            Text connector stack used to adapt text encoder hidden states for the video and audio branches.
        transformer ([`LTX2VideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        vocoder ([`LTX2Vocoder`] or [`LTX2VocoderWithBWE`]):
            Vocoder to convert mel spectrograms to audio waveforms.
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

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, resample="bilinear")

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
        latents=None,
        audio_latents=None,
        spatio_temporal_guidance_blocks=None,
        stg_scale=None,
        audio_stg_scale=None,
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

        if latents is not None and latents.ndim != 5:
            raise ValueError(
                f"Only unpacked (5D) video latents of shape `[batch_size, latent_channels, latent_frames,"
                f" latent_height, latent_width] are supported, but got {latents.ndim} dims. If you have packed (3D)"
                f" latents, please unpack them (e.g. using the `_unpack_latents` method)."
            )
        if audio_latents is not None and audio_latents.ndim != 4:
            raise ValueError(
                f"Only unpacked (4D) audio latents of shape `[batch_size, num_channels, audio_length, mel_bins] are"
                f" supported, but got {audio_latents.ndim} dims. If you have packed (3D) latents, please unpack them"
                f" (e.g. using the `_unpack_audio_latents` method)."
            )

        if ((stg_scale > 0.0) or (audio_stg_scale > 0.0)) and not spatio_temporal_guidance_blocks:
            raise ValueError(
                "Spatio-Temporal Guidance (STG) is specified but no STG blocks are supplied. Please supply a list of"
                "block indices at which to apply STG in `spatio_temporal_guidance_blocks`"
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

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.trim_conditioning_sequence
    def trim_conditioning_sequence(self, start_frame: int, sequence_num_frames: int, target_num_frames: int) -> int:
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

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.preprocess_conditions
    def preprocess_conditions(
        self,
        conditions: LTX2VideoCondition | list[LTX2VideoCondition] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        device: torch.device | None = None,
    ) -> tuple[list[torch.Tensor], list[float], list[int], list[int]]:
        """
        Preprocesses the condition images/videos to torch tensors.

        Args:
            conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*, defaults to `None`):
                A list of image/video condition instances.
            height (`int`, *optional*, defaults to `512`):
                The desired height in pixels.
            width (`int`, *optional*, defaults to `768`):
                The desired width in pixels.
            num_frames (`int`, *optional*, defaults to `121`):
                The desired number of frames in the generated video.
            device (`torch.device`, *optional*, defaults to `None`):
                The device on which to put the preprocessed image/video tensors.

        Returns:
            `Tuple[List[torch.Tensor], List[float], List[int], List[int]]`:
                Returns a 4-tuple of lists of length `len(conditions)` as follows:
                    1. The first list is a list of preprocessed video tensors of shape [batch_size=1, num_channels,
                       num_frames, height, width].
                    2. The second list is a list of conditioning strengths.
                    3. The third list is a list of latent-space indices for each condition.
                    4. The fourth list is a list of (trimmed) pixel-space frame counts per condition. This is needed
                       for keyframe coord semantics (single-pixel-frame keyframes have a clamped temporal extent).
        """
        conditioning_frames, conditioning_strengths, conditioning_indices, conditioning_pixel_frames = [], [], [], []

        if conditions is None:
            conditions = []
        if isinstance(conditions, LTX2VideoCondition):
            conditions = [conditions]

        frame_scale_factor = self.vae_temporal_compression_ratio
        latent_num_frames = (num_frames - 1) // frame_scale_factor + 1
        for i, condition in enumerate(conditions):
            # Create a channels-last video-like array of shape (F, H, W, C) in preparation for resizing.
            if isinstance(condition.frames, PIL.Image.Image):
                arr = np.array(condition.frames.convert("RGB"))[None]  # (1, H, W, 3)
            elif isinstance(condition.frames, list) and all(isinstance(f, PIL.Image.Image) for f in condition.frames):
                arr = np.stack([np.array(f.convert("RGB")) for f in condition.frames])  # (F, H, W, 3)
            elif isinstance(condition.frames, np.ndarray):
                arr = condition.frames if condition.frames.ndim == 4 else condition.frames[None]
            elif isinstance(condition.frames, torch.Tensor):
                t = condition.frames if condition.frames.ndim == 4 else condition.frames.unsqueeze(0)
                # Reference layout for video tensors is (F, C, H, W); convert to (F, H, W, C) for the
                # resize logic, which expects channels-last.
                arr = t.detach().cpu().permute(0, 2, 3, 1).numpy()
            else:
                raise TypeError(f"Unsupported `frames` type for condition {i}: {type(condition.frames)}")

            src_h, src_w = arr.shape[1], arr.shape[2]
            num_cond_frames = arr.shape[0]
            # Convert the NumPy array to a channels-first tensor of shape (1, C, F, H, W)
            pixels = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
            pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, C, F, H, W)

            # Resize so the longer side fills the target, then center-crop to exact (height, width).
            scale = max(height / src_h, width / src_w)
            new_h = math.ceil(src_h * scale)
            new_w = math.ceil(src_w * scale)
            # Flatten (B, C, F, H, W) → (B*F, C, H, W) for the per-frame interpolation
            pixels = pixels.permute(0, 2, 1, 3, 4).reshape(num_cond_frames, 3, src_h, src_w)
            # NOTE: we avoid using VideoProcessor.preprocess_video here because it uses PIL.Image.resize under the
            # hood, which will apply an anti-aliasing pre-filter when downsampling. The original LTX-2.X code simply
            # uses F.interpolate, which is reproduced here.
            pixels = torch.nn.functional.interpolate(pixels, size=(new_h, new_w), mode="bilinear", align_corners=False)
            top = (new_h - height) // 2
            left = (new_w - width) // 2
            pixels = pixels[:, :, top : top + height, left : left + width]
            pixels = pixels.reshape(1, num_cond_frames, 3, height, width).permute(0, 2, 1, 3, 4)

            # Map [0, 255] → [-1, 1] (VAE input convention).
            condition_pixels = pixels / 127.5 - 1.0

            # Interpret the index as a latent index, following the original LTX-2 code.
            latent_start_idx = condition.index
            # Support negative latent indices (e.g. -1 for the last latent index)
            if latent_start_idx < 0:
                # latent_start_idx will be positive because latent_num_frames is positive
                latent_start_idx = latent_start_idx % latent_num_frames
            if latent_start_idx >= latent_num_frames:
                logger.warning(
                    f"The starting latent index {latent_start_idx} of condition {i} is too big for the specified number"
                    f" of latent frames {latent_num_frames}. This condition will be skipped."
                )
                continue

            cond_num_frames = condition_pixels.size(2)
            start_idx = max((latent_start_idx - 1) * frame_scale_factor + 1, 0)
            truncated_cond_frames = self.trim_conditioning_sequence(start_idx, cond_num_frames, num_frames)
            condition_pixels = condition_pixels[:, :, :truncated_cond_frames]

            conditioning_frames.append(condition_pixels.to(dtype=self.vae.dtype, device=device))
            conditioning_strengths.append(condition.strength)
            conditioning_indices.append(latent_start_idx)
            conditioning_pixel_frames.append(truncated_cond_frames)

        return conditioning_frames, conditioning_strengths, conditioning_indices, conditioning_pixel_frames

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline.apply_first_frame_conditioning
    def apply_first_frame_conditioning(
        self,
        latents: torch.Tensor,
        conditioning_mask: torch.Tensor,
        condition_latents: list[torch.Tensor],
        condition_strengths: list[float],
        condition_indices: list[int],
        latent_height: int,
        latent_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply first-frame visual conditioning by overwriting tokens at the first-frame positions.

        Only conditions with `latent_idx == 0` are applied here (matching `VideoConditionByLatentIndex` in the
        reference implementation). Conditions at non-zero latent indices are appended as separate keyframe tokens via
        `prepare_keyframe_extras` (matching `VideoConditionByKeyframeIndex`) and are skipped here.

        Args:
            latents (`torch.Tensor`):
                Initial packed (patchified) latents of shape [batch_size, patch_seq_len, hidden_dim].
            conditioning_mask (`torch.Tensor`):
                Initial packed (patchified) conditioning mask of shape [batch_size, patch_seq_len, 1] with values in
                [0, 1] where 0 means the denoising model output will be fully used and 1 means the condition will be
                fully used.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:
                Returns a 3-tuple of tensors where:
                    1. The packed video latents with first-frame conditions applied.
                    2. The packed conditioning mask with first-frame strengths applied.
                    3. The clean conditioning latents at first-frame positions (zeros elsewhere).
        """
        clean_latents = torch.zeros_like(latents)
        for cond, strength, latent_idx in zip(condition_latents, condition_strengths, condition_indices):
            if latent_idx != 0:
                # Non-first-frame conditions are handled as keyframe extras (appended tokens) instead.
                continue
            num_cond_tokens = cond.size(1)
            start_token_idx = latent_idx * latent_height * latent_width
            end_token_idx = start_token_idx + num_cond_tokens

            latents[:, start_token_idx:end_token_idx] = cond
            conditioning_mask[:, start_token_idx:end_token_idx] = strength
            clean_latents[:, start_token_idx:end_token_idx] = cond

        return latents, conditioning_mask, clean_latents

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline._prepare_keyframe_coords
    def _prepare_keyframe_coords(
        self,
        keyframe_latent_num_frames: int,
        keyframe_latent_height: int,
        keyframe_latent_width: int,
        pixel_frame_idx: int,
        num_pixel_frames: int,
        fps: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute positional coordinates for a keyframe condition being appended as extra tokens.

        Mirrors `VideoConditionByKeyframeIndex.apply_to` in the reference implementation:
        - Latent coords scaled to pixel space *without* the causal fix (since non-zero-index keyframes don't need the
          first-frame causal adjustment).
        - Temporal axis offset by `pixel_frame_idx` (the pixel-space index at which the keyframe appears).
        - For single-pixel-frame keyframes, the per-patch temporal extent is clamped to `[idx, idx + 1)` so the
          keyframe occupies a single pixel timestep rather than the VAE-scaled range.
        - Temporal coords divided by `fps` to produce seconds.
        """
        patch_size = self.transformer_spatial_patch_size
        patch_size_t = self.transformer_temporal_patch_size
        scale_factors = (
            self.vae_temporal_compression_ratio,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )

        grid_f = torch.arange(
            start=0, end=keyframe_latent_num_frames, step=patch_size_t, dtype=torch.float32, device=device
        )
        grid_h = torch.arange(start=0, end=keyframe_latent_height, step=patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=keyframe_latent_width, step=patch_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        latent_coords = torch.stack([grid, patch_ends], dim=-1)  # [3, N_F, N_H, N_W, 2]
        latent_coords = latent_coords.flatten(1, 3)  # [3, num_patches, 2]
        latent_coords = latent_coords.unsqueeze(0)  # [1, 3, num_patches, 2]

        scale_tensor = torch.tensor(scale_factors, device=device, dtype=latent_coords.dtype)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        # No causal fix: keyframe coords place the keyframe at `pixel_frame_idx` without the first-frame adjustment.
        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] + pixel_frame_idx

        if num_pixel_frames == 1:
            # Single-pixel-frame keyframe: clamp temporal extent to [idx, idx + 1).
            pixel_coords[:, 0, :, 1:] = pixel_coords[:, 0, :, :1] + 1

        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

        return pixel_coords

    def prepare_latents(
        self,
        conditions: LTX2VideoCondition | list[LTX2VideoCondition] | None = None,
        reference_conditions: list[LTX2ReferenceCondition] | None = None,
        reference_downscale_factor: int = 1,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        noise_scale: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, torch.Tensor | None]:
        """
        Prepare noisy video latents, applying frame and reference-video conditioning.

        Conditioning sources are unified into a single packed sequence in the order `[base | keyframe | reference]`:

        - First-frame conditions (`conditions` with `latent_idx == 0`) overwrite tokens at the first-frame positions
          (`VideoConditionByLatentIndex` semantics).
        - Non-first-frame conditions (`conditions` with `latent_idx > 0`) are concatenated onto the main latent
          sequence with per-token `conditioning_mask = strength` (`VideoConditionByKeyframeIndex` semantics).
        - IC-LoRA `reference_conditions` (if any) are encoded by the VAE and appended after the keyframes with
          per-token `conditioning_mask = strength` (matching the reference repo's `VideoConditionByReferenceLatent`
          semantics).

        For all appended tokens the noise mixing below blends them to noise level `(1 - strength) * sigma_max`, and the
        existing per-token timestep formula `t * (1 - conditioning_mask)` and the post-process blend `denoised * (1 -
        cond_mask) + clean * cond_mask` drive them through the loop.

        Returns a 6-tuple:
            - `latents`: packed noisy latents `(B, base + n_keyframe + n_ref, C)`.
            - `conditioning_mask`: `(B, seq_len, 1)` with values in `[0, 1]` — `1` at first-frame positions, `strength`
              at keyframe / reference positions, `0` elsewhere.
            - `clean_latents`: clean condition values at conditioned positions (zeros elsewhere); same shape as
              `latents`.
            - `appended_coords`: `[1, 3, n_keyframe + n_ref, 2]` positional coordinates to concat onto `video_coords`,
              or `None` if no keyframe/reference conditions are provided.
            - `num_ref_tokens`: count of reference tokens at the END of `latents` (used by the call site to build the
              unified self-attention mask).
            - `ref_cross_mask`: `[1, num_ref_tokens]` per-reference-token cross-attention strengths in `[0, 1]`, or
              `None` when `conditioning_attention_strength == 1.0` and no pixel-space mask is provided (in which case
              attention is uniform).
        """
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
        mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)

        if latents is not None:
            # Latents are expected to be unpacked (5D) with shape [B, F, C, H, W]
            latents = self._normalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
        else:
            # NOTE: we set the initial latents to zeros rather a sample from the standard Gaussian prior because we
            # will sample from the prior later once we have calculated the conditioning mask
            latents = torch.zeros(shape, device=device, dtype=dtype)

        conditioning_mask = latents.new_zeros(mask_shape)
        latents = self._pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )
        conditioning_mask = self._pack_latents(
            conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )  # [B, seq_len, 1]

        if latents.ndim != 3 or latents.shape[:2] != conditioning_mask.shape[:2]:
            raise ValueError(
                f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is {conditioning_mask.shape[:2] + (num_channels_latents,)}."
            )

        if isinstance(generator, list):
            logger.warning(
                f"{self.__class__.__name__} does not support using a list of generators. The first generator in the"
                f" list will be used for all (pseudo-)random operations."
            )

        condition_frames, condition_strengths, condition_indices, condition_pixel_frames = self.preprocess_conditions(
            conditions, height, width, num_frames, device=device
        )
        # Encode each condition through the VAE. We keep both the 5D latent (for coord computation) and the packed
        # 3D latent (for first-frame replacement or keyframe append).
        condition_latents_5d = []
        condition_latents_packed = []
        for condition_tensor in condition_frames:
            condition_latent_5d = retrieve_latents(
                self.vae.encode(condition_tensor),
                generator=generator[0] if isinstance(generator, list) else generator,
                sample_mode="argmax",
            )
            condition_latent_5d = self._normalize_latents(
                condition_latent_5d, self.vae.latents_mean, self.vae.latents_std
            ).to(device=device, dtype=dtype)
            condition_latent_packed = self._pack_latents(
                condition_latent_5d, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )
            condition_latents_5d.append(condition_latent_5d)
            condition_latents_packed.append(condition_latent_packed)

        # First-frame conditions (latent_idx == 0): replace tokens at the first-frame positions.
        # NOTE: following the I2V pipeline, we return a conditioning mask. The original LTX 2 code uses a denoising
        # mask, which is the inverse of the conditioning mask (`denoise_mask = 1 - conditioning_mask`).
        latents, conditioning_mask, clean_latents = self.apply_first_frame_conditioning(
            latents,
            conditioning_mask,
            condition_latents_packed,
            condition_strengths,
            condition_indices,
            latent_height=latent_height,
            latent_width=latent_width,
        )

        # Non-first-frame ("keyframe") conditions (latent_idx > 0): append as extra latent tokens to the noisy latent.
        # Each condition gets a all-`strength` conditioning mask and pos ids, which are also appended to those of the
        # noisy latent. At each denoising step i, the keyframe conditions get an effective noise level of
        # (1 - conditioning_strength) * sigma_i.
        frame_scale_factor = self.vae_temporal_compression_ratio
        kf_tokens_list, kf_coords_list, kf_mask_list, kf_clean_list = [], [], [], []
        for cond_5d, cond_packed, strength, latent_idx, num_pixel_frames in zip(
            condition_latents_5d,
            condition_latents_packed,
            condition_strengths,
            condition_indices,
            condition_pixel_frames,
        ):
            if latent_idx == 0:
                continue

            _, _, kf_latent_frames, kf_latent_height, kf_latent_width = cond_5d.shape
            pixel_frame_idx = (latent_idx - 1) * frame_scale_factor + 1

            coords = self._prepare_keyframe_coords(
                keyframe_latent_num_frames=kf_latent_frames,
                keyframe_latent_height=kf_latent_height,
                keyframe_latent_width=kf_latent_width,
                pixel_frame_idx=pixel_frame_idx,
                num_pixel_frames=num_pixel_frames,
                fps=frame_rate,
                device=device,
            )

            num_tokens = cond_packed.shape[1]
            kf_mask = torch.full(
                (cond_packed.shape[0], num_tokens, 1),
                float(strength),
                device=device,
                dtype=conditioning_mask.dtype,
            )

            kf_tokens_list.append(cond_packed)
            kf_clean_list.append(cond_packed)
            kf_mask_list.append(kf_mask)
            kf_coords_list.append(coords)

        if kf_tokens_list:
            keyframe_coords = torch.cat(kf_coords_list, dim=2)
            latents = torch.cat([latents, torch.cat(kf_tokens_list, dim=1)], dim=1)
            conditioning_mask = torch.cat([conditioning_mask, torch.cat(kf_mask_list, dim=1)], dim=1)
            clean_latents = torch.cat([clean_latents, torch.cat(kf_clean_list, dim=1)], dim=1)
        else:
            keyframe_coords = None

        # IC-LoRA reference-video conditions: encode each reference video, then append it to the main packed
        # sequence with per-token `conditioning_mask = strength`. This is the same architectural pattern as
        # for non-first-frame conditions above, but we need to keep keyframe and reference conditions separate
        # for attention masking.
        ref_cross_mask: torch.Tensor | None = None
        ref_coords: torch.Tensor | None = None
        num_ref_tokens = 0
        if reference_conditions is not None and len(reference_conditions) > 0:
            ref_latents_packed, ref_coords, ref_cross_mask = self._encode_reference_conditions(
                reference_conditions=reference_conditions,
                num_frames=num_frames,
                height=height,
                width=width,
                reference_downscale_factor=reference_downscale_factor,
                frame_rate=frame_rate,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
                dtype=dtype,
                device=device,
                generator=generator[0] if isinstance(generator, list) else generator,
            )
            num_ref_tokens = ref_latents_packed.shape[1]

            # All reference videos preprocess to the same (ref_height, ref_width, num_frames), so their packed
            # token counts are identical. Split `num_ref_tokens` evenly across the conditions and materialize
            # the per-token strength mask in `reference_conditions` order, matching the layout the encoder
            # emitted.
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

        # Combine keyframe + reference appended-coords into a single block to concat onto `video_coords` at
        # the call site.
        if keyframe_coords is not None and ref_coords is not None:
            appended_coords = torch.cat([keyframe_coords, ref_coords], dim=2)
        elif keyframe_coords is not None:
            appended_coords = keyframe_coords
        elif ref_coords is not None:
            appended_coords = ref_coords
        else:
            appended_coords = None

        # The conditioning_mask values have the following semantics:
        #   - mask=0: fully noise tokens (e.g. noisy latents)
        #   - mask=1: keep fully clean (e.g. I2V first-frame condition, conditions with strength=1)
        #   - mask in (0, 1): use intermediate noise level mask * sigma_i (noise_scale == sigma_0)
        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        scaled_mask = (1.0 - conditioning_mask) * noise_scale  # noise to initial noise level `noise_scale`
        latents = noise * scaled_mask + latents * (1 - scaled_mask)

        return latents, conditioning_mask, clean_latents, appended_coords, num_ref_tokens, ref_cross_mask

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
        reference_conditions: list[LTX2ReferenceCondition],
        height: int,
        width: int,
        num_frames: int,
        reference_downscale_factor: int = 1,
        frame_rate: float = 24.0,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode IC-LoRA reference videos into `(reference_latents, reference_coords, reference_cross_mask)`.

        This is the shared encoding core used by both `prepare_latents` (which folds reference tokens into the main
        noisy sequence) and the back-compat shim `prepare_reference_latents` (which exposes the legacy 4-tuple output).
        See `prepare_reference_latents` for parameter documentation.
        """
        ref_height = height // reference_downscale_factor
        ref_width = width // reference_downscale_factor

        mask_needed = conditioning_attention_strength < 1.0 or conditioning_attention_mask is not None

        all_ref_latents = []
        all_ref_coords = []
        all_ref_cross_masks = []

        for ref_cond in reference_conditions:
            # Preprocess reference video frames to the (possibly downscaled) resolution
            if isinstance(ref_cond.frames, PIL.Image.Image):
                video_like = [ref_cond.frames]
            elif isinstance(ref_cond.frames, np.ndarray) and ref_cond.frames.ndim == 3:
                video_like = np.expand_dims(ref_cond.frames, axis=0)
            elif isinstance(ref_cond.frames, torch.Tensor) and ref_cond.frames.ndim == 3:
                video_like = ref_cond.frames.unsqueeze(0)
            else:
                video_like = ref_cond.frames

            ref_pixels = self.video_processor.preprocess_video(video_like, ref_height, ref_width, resize_mode="crop")
            # Trim to num_frames
            ref_pixels = ref_pixels[:, :, :num_frames]
            ref_pixels = ref_pixels.to(dtype=self.vae.dtype, device=device)

            # Encode through VAE
            ref_latent = retrieve_latents(self.vae.encode(ref_pixels), generator=generator, sample_mode="argmax")
            ref_latent = self._normalize_latents(ref_latent, self.vae.latents_mean, self.vae.latents_std).to(
                device=device, dtype=dtype
            )

            # Get latent dimensions for coordinate computation
            _, _, ref_latent_frames, ref_latent_height, ref_latent_width = ref_latent.shape

            # Pack into tokens
            ref_latent_packed = self._pack_latents(
                ref_latent, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )

            # Compute positional coordinates for the reference tokens. We use the transformer's
            # prepare_video_coords at the reference video's latent dimensions, then scale spatial coords
            # by downscale_factor so they map to the target coordinate space.
            ref_coords = self.transformer.rope.prepare_video_coords(
                batch_size=1,
                num_frames=ref_latent_frames,
                height=ref_latent_height,
                width=ref_latent_width,
                device=device,
                fps=frame_rate,
            )
            if reference_downscale_factor != 1:
                # Scale spatial coordinates (height=axis 1, width=axis 2) to match target space
                ref_coords[:, 1, :, :] = ref_coords[:, 1, :, :] * reference_downscale_factor
                ref_coords[:, 2, :, :] = ref_coords[:, 2, :, :] * reference_downscale_factor

            num_tokens = ref_latent_packed.shape[1]

            all_ref_latents.append(ref_latent_packed)
            all_ref_coords.append(ref_coords)

            if mask_needed:
                # Per-reference cross-attention mask. Start from either a downsampled pixel-space mask or a full-1
                # tensor, then scale by conditioning_attention_strength.
                if conditioning_attention_mask is not None:
                    ref_cross = self._downsample_mask_to_latent(
                        mask=conditioning_attention_mask,
                        latent_num_frames=ref_latent_frames,
                        latent_height=ref_latent_height,
                        latent_width=ref_latent_width,
                    ).to(device=device, dtype=torch.float32)
                else:
                    ref_cross = torch.ones((1, num_tokens), device=device, dtype=torch.float32)
                ref_cross = ref_cross * conditioning_attention_strength
                all_ref_cross_masks.append(ref_cross)

        # Concatenate all reference tokens into a single sequence
        reference_latents = torch.cat(all_ref_latents, dim=1)  # [1, total_ref_tokens, D]
        reference_coords = torch.cat(all_ref_coords, dim=2)  # [1, 3, total_ref_tokens, 2]
        reference_cross_mask = torch.cat(all_ref_cross_masks, dim=1) if mask_needed else None

        return reference_latents, reference_coords, reference_cross_mask

    def prepare_reference_latents(
        self,
        reference_conditions: list[LTX2ReferenceCondition],
        height: int,
        width: int,
        num_frames: int,
        reference_downscale_factor: int = 1,
        frame_rate: float = 24.0,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Encode reference videos into packed latent tokens and compute their positional coordinates.

        Each reference video is independently encoded by the VAE, packed into tokens, and its positional coordinates
        are computed with spatial scaling by `reference_downscale_factor` to match the target coordinate space.

        All reference tokens are concatenated into a single sequence. When `conditioning_attention_strength < 1.0` or
        `conditioning_attention_mask` is provided, a per-token cross-attention mask is also computed for each reference
        video (downsampled to the reference video's latent dimensions) and returned so callers can build a
        self-attention mask over the full video sequence.

        Args:
            reference_conditions (`list[LTX2ReferenceCondition]`):
                The reference video conditions.
            height (`int`):
                Target video height in pixels (used to determine reference video preprocessing size with
                `reference_downscale_factor`).
            width (`int`):
                Target video width in pixels.
            num_frames (`int`):
                Number of target video frames.
            reference_downscale_factor (`int`, defaults to `1`):
                Ratio between target and reference resolutions. A factor of 2 means the reference video is preprocessed
                at half the target resolution. Spatial positional coordinates are scaled by this factor to map
                reference tokens into the target coordinate space.
            frame_rate (`float`, defaults to `24.0`):
                Video frame rate (used for temporal coordinate computation).
            conditioning_attention_strength (`float`, defaults to `1.0`):
                Scalar in `[0, 1]` controlling how strongly reference tokens attend to noisy tokens (and vice versa) in
                the self-attention mask. `1.0` means full attention (no masking), `0.0` means reference tokens are
                effectively ignored by the noisy tokens.
            conditioning_attention_mask (`torch.Tensor`, *optional*):
                Optional pixel-space mask of shape `(1, 1, F_pix, H_pix, W_pix)` with values in `[0, 1]` that provides
                spatially-varying attention strength. Downsampled to latent space per reference video and multiplied by
                `conditioning_attention_strength`.
            dtype (`torch.dtype`, *optional*):
                Data type for the latents.
            device (`torch.device`, *optional*):
                Device for the latents.
            generator (`torch.Generator`, *optional*):
                Random generator for VAE encoding.

        Returns:
            A 4-tuple of `(reference_latents, reference_coords, reference_denoise_factors, reference_cross_mask)`:
                - `reference_latents`: `[1, total_ref_tokens, hidden_dim]`
                - `reference_coords`: `[1, 3, total_ref_tokens, 2]`
                - `reference_denoise_factors`: `[1, total_ref_tokens]` — per-token `(1 - strength)` factors
                - `reference_cross_mask`: `[1, total_ref_tokens]` per-token noisy↔reference attention strengths in `[0,
                  1]`, or `None` when `conditioning_attention_strength == 1.0` and no pixel-space mask is provided (in
                  which case attention is unmasked).
        """
        reference_latents, reference_coords, reference_cross_mask = self._encode_reference_conditions(
            reference_conditions=reference_conditions,
            height=height,
            width=width,
            num_frames=num_frames,
            reference_downscale_factor=reference_downscale_factor,
            frame_rate=frame_rate,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Materialize per-token denoise factors for callers that still expect the 4-tuple. Each ref video has
        # `1 - strength` for all of its tokens; we rebuild this from the per-video token counts which we can
        # back out from `reference_latents.shape[1]` and the input `reference_conditions` order.
        ref_denoise_chunks: list[torch.Tensor] = []
        idx = 0
        # Walk the encoded ref tokens video-by-video. Each ref's token count is fixed by the ref video's latent
        # shape, which equals (num_frames -> ref_latent_frames) * ref_latent_h * ref_latent_w. Computing it here
        # would duplicate the encoding math; instead we rely on the shape match across all refs being identical
        # (same `num_frames`, same downscaled height/width) so we can split equally.
        n_total = reference_latents.shape[1]
        n_per_ref = n_total // max(len(reference_conditions), 1)
        for ref_cond in reference_conditions:
            ref_denoise_chunks.append(
                torch.full(
                    (1, n_per_ref), 1.0 - ref_cond.strength, device=reference_latents.device, dtype=torch.float32
                )
            )
            idx += n_per_ref
        reference_denoise_factors = (
            torch.cat(ref_denoise_chunks, dim=1) if ref_denoise_chunks else reference_latents.new_zeros((1, 0))
        )

        return reference_latents, reference_coords, reference_denoise_factors, reference_cross_mask

    @staticmethod
    def _downsample_mask_to_latent(
        mask: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """
        Downsample a pixel-space attention mask to a flattened per-token latent-space mask. Uses causal temporal
        downsampling (the first frame is kept as-is).

        Args:
            mask (`torch.Tensor`):
                Pixel-space mask of shape `(B, 1, F_pix, H_pix, W_pix)` with values in `[0, 1]`.
            latent_num_frames (`int`), latent_height (`int`), latent_width (`int`):
                Target latent dimensions.

        Returns:
            Flattened latent-space mask of shape `(B, latent_num_frames * latent_height * latent_width)`.
        """
        if mask.ndim != 5 or mask.shape[1] != 1:
            raise ValueError(
                f"Expected `conditioning_attention_mask` of shape (B, 1, F, H, W), got {tuple(mask.shape)}."
            )
        b, _, f_pix, _, _ = mask.shape

        # 1. Spatial downsampling (area interpolation per frame).
        mask_2d = mask.reshape(b * f_pix, 1, mask.shape[-2], mask.shape[-1])
        spatial_down = torch.nn.functional.interpolate(mask_2d, size=(latent_height, latent_width), mode="area")
        spatial_down = spatial_down.reshape(b, 1, f_pix, latent_height, latent_width)

        # 2. Causal temporal downsampling.
        first_frame = spatial_down[:, :, :1, :, :]  # (B, 1, 1, H_lat, W_lat)
        if f_pix > 1 and latent_num_frames > 1:
            t = (f_pix - 1) // (latent_num_frames - 1)
            if (f_pix - 1) % (latent_num_frames - 1) != 0:
                raise ValueError(
                    f"Pixel frames ({f_pix}) not compatible with latent frames ({latent_num_frames}): "
                    f"(f_pix - 1) must be divisible by (latent_num_frames - 1)."
                )
            rest = spatial_down[:, :, 1:, :, :]
            rest = rest.reshape(b, 1, latent_num_frames - 1, t, latent_height, latent_width).mean(dim=3)
            latent_mask = torch.cat([first_frame, rest], dim=2)
        else:
            latent_mask = first_frame

        # 3. Flatten to token order (f, h, w).
        return latent_mask.reshape(b, latent_num_frames * latent_height * latent_width)

    @staticmethod
    def _build_video_self_attention_mask(
        num_noisy_tokens: int,
        extras_cross_masks: list[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Build the `(1, T_video, T_video)` self-attention mask over `noisy + extras` tokens, where `extras` is a
        concatenation of one or more conditioning groups (e.g. keyframes, IC-LoRA references).

        Block structure (mirrors the reference `update_attention_mask` / `ConditioningItemAttentionStrengthWrapper`):
            - noisy ↔ noisy: 1.0 (full attention)
            - noisy ↔ group_i: `extras_cross_masks[i]` broadcast across the noisy-token axis
            - group_i ↔ noisy: `extras_cross_masks[i]` broadcast across the noisy-token axis (symmetric)
            - group_i ↔ group_i: 1.0 (tokens in a group fully attend to themselves)
            - group_i ↔ group_j (i != j): 0.0 (different conditioning groups don't cross-attend)

        Args:
            num_noisy_tokens (`int`):
                Number of noisy video tokens.
            extras_cross_masks (`list[torch.Tensor]`):
                List of per-token cross-attention strengths, one per conditioning group. Each entry has shape `(1,
                num_tokens_in_group)` with values in `[0, 1]`. Groups must appear in the same order as their tokens in
                the extras block.
            device, dtype:
                Tensor device and dtype.

        Returns:
            Multiplicative self-attention mask of shape `(1, num_noisy_tokens + sum(group_sizes), num_noisy_tokens +
            sum(group_sizes))` with values in `[0, 1]`.
        """
        total_extras = sum(m.shape[1] for m in extras_cross_masks)
        total = num_noisy_tokens + total_extras

        # Initialize to 0 so that between-group blocks remain masked without explicit assignment.
        attn_mask = torch.zeros((1, total, total), device=device, dtype=dtype)
        attn_mask[:, :num_noisy_tokens, :num_noisy_tokens] = 1.0  # noisy ↔ noisy

        offset = num_noisy_tokens
        for cross_mask in extras_cross_masks:
            n = cross_mask.shape[1]
            cross = cross_mask.to(device=device, dtype=dtype)
            # noisy (rows) ↔ this group (cols)
            attn_mask[:, :num_noisy_tokens, offset : offset + n] = cross.unsqueeze(1)
            # this group (rows) ↔ noisy (cols)
            attn_mask[:, offset : offset + n, :num_noisy_tokens] = cross.unsqueeze(2)
            # this group ↔ this group (self-attention within the group)
            attn_mask[:, offset : offset + n, offset : offset + n] = 1.0
            offset += n
        return attn_mask

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
    def audio_guidance_scale(self):
        return self._audio_guidance_scale

    @property
    def audio_guidance_rescale(self):
        return self._audio_guidance_rescale

    @property
    def audio_stg_scale(self):
        return self._audio_stg_scale

    @property
    def audio_modality_scale(self):
        return self._audio_modality_scale

    @property
    def do_classifier_free_guidance(self):
        return (self._guidance_scale > 1.0) or (self._audio_guidance_scale > 1.0)

    @property
    def do_spatio_temporal_guidance(self):
        return (self._stg_scale > 0.0) or (self._audio_stg_scale > 0.0)

    @property
    def do_modality_isolation_guidance(self):
        return (self._modality_scale > 1.0) or (self._audio_modality_scale > 1.0)

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
        reference_conditions: LTX2ReferenceCondition | list[LTX2ReferenceCondition] | None = None,
        conditions: LTX2VideoCondition | list[LTX2VideoCondition] | None = None,
        reference_downscale_factor: int = 1,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 30,
        sigmas: list[float] | None = None,
        timesteps: list[float] | None = None,
        guidance_scale: float = 3.0,
        stg_scale: float = 1.0,
        modality_scale: float = 3.0,
        guidance_rescale: float = 0.7,
        audio_guidance_scale: float | None = 7.0,
        audio_stg_scale: float | None = 1.0,
        audio_modality_scale: float | None = 3.0,
        audio_guidance_rescale: float | None = 0.7,
        spatio_temporal_guidance_blocks: list[int] | None = [28],
        noise_scale: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        use_cross_timestep: bool = True,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide video generation. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            reference_conditions (`LTX2ReferenceCondition` or `List[LTX2ReferenceCondition]`, *optional*):
                Reference video conditions for IC-LoRA conditioning. Each reference video is encoded into latent tokens
                and concatenated to the noisy latent sequence during denoising, allowing the IC-LoRA adapter to
                condition the generation on the reference video content.
            conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*):
                Frame-level conditioning (same as [`LTX2ConditionPipeline`]). Conditions are inserted at specific
                latent positions and blended with the denoised output during each denoising step.
            reference_downscale_factor (`int`, *optional*, defaults to `1`):
                Ratio between target and reference video resolutions. IC-LoRA models trained with downscaled reference
                videos store this factor in their safetensors metadata (`reference_downscale_factor` key). A factor of
                `2` means the reference video is preprocessed at half the target resolution and spatial positional
                coordinates are scaled accordingly.
            conditioning_attention_strength (`float`, *optional*, defaults to `1.0`):
                Scalar in `[0, 1]` controlling how strongly noisy tokens and appended reference tokens attend to each
                other in the video self-attention. `1.0` = full attention (no masking, same as the base IC-LoRA
                behavior). `0.0` = reference tokens are fully masked out of the noisy-token attention (and vice versa).
                Only takes effect when `reference_conditions` is provided.
            conditioning_attention_mask (`torch.Tensor`, *optional*):
                Optional pixel-space spatial attention mask of shape `(1, 1, F_pix, H_pix, W_pix)` with values in `[0,
                1]` that provides per-region attention strength. The mask's spatial-temporal dimensions must match the
                reference video's pixel dimensions. Downsampled to latent space using VAE scale factors (with causal
                temporal handling for the first frame) and multiplied by `conditioning_attention_strength` to form the
                final cross-attention mask between noisy and reference tokens. Only takes effect when
                `reference_conditions` is provided.
            height (`int`, *optional*, defaults to `512`):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to `768`):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to `121`):
                The number of video frames to generate.
            frame_rate (`float`, *optional*, defaults to `24.0`):
                The frames per second (FPS) of the generated video.
            num_inference_steps (`int`, *optional*, defaults to 40):
                The number of denoising steps.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to `4.0`):
                Classifier-Free Guidance scale for video.
            stg_scale (`float`, *optional*, defaults to `0.0`):
                Spatio-Temporal Guidance scale for video. `0.0` disables STG.
            modality_scale (`float`, *optional*, defaults to `1.0`):
                Modality isolation guidance scale for video. `1.0` disables modality guidance.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor for video.
            audio_guidance_scale (`float`, *optional*, defaults to `None`):
                CFG scale for audio. If `None`, defaults to `guidance_scale`.
            audio_stg_scale (`float`, *optional*, defaults to `None`):
                STG scale for audio. If `None`, defaults to `stg_scale`.
            audio_modality_scale (`float`, *optional*, defaults to `None`):
                Modality guidance scale for audio. If `None`, defaults to `modality_scale`.
            audio_guidance_rescale (`float`, *optional*, defaults to `None`):
                Guidance rescale for audio. If `None`, defaults to `guidance_rescale`.
            spatio_temporal_guidance_blocks (`list[int]`, *optional*):
                Transformer block indices at which to apply STG.
            noise_scale (`float`, *optional*):
                Noise scale for latent initialization. If not set, inferred from the sigma schedule.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator(s) for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video latents (5D unpacked).
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio latents (4D unpacked).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                Noise scale at decode time.
            use_cross_timestep (`bool`, *optional*, defaults to `False`):
                Whether to use cross-modality sigma for cross attention modulation. `True` for LTX-2.3+.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format. Choose `"pil"`, `"np"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`LTX2PipelineOutput`] or a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Additional kwargs passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*, defaults to `["latents"]`):
                Tensor inputs for the callback function.
            max_sequence_length (`int`, *optional*, defaults to `1024`):
                Maximum sequence length for the text prompt.

        Examples:

        Returns:
            [`LTX2PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`LTX2PipelineOutput`] is returned, otherwise a `tuple` of `(video, audio)`
                is returned.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        audio_guidance_scale = audio_guidance_scale or guidance_scale
        audio_stg_scale = audio_stg_scale or stg_scale
        audio_modality_scale = audio_modality_scale or modality_scale
        audio_guidance_rescale = audio_guidance_rescale or guidance_rescale

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
            latents=latents,
            audio_latents=audio_latents,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
            stg_scale=stg_scale,
            audio_stg_scale=audio_stg_scale,
        )

        # Per-modality guidance scales
        self._guidance_scale = guidance_scale
        self._stg_scale = stg_scale
        self._modality_scale = modality_scale
        self._guidance_rescale = guidance_rescale
        self._audio_guidance_scale = audio_guidance_scale
        self._audio_stg_scale = audio_stg_scale
        self._audio_modality_scale = audio_modality_scale
        self._audio_guidance_rescale = audio_guidance_rescale

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if conditions is not None and not isinstance(conditions, list):
            conditions = [conditions]
        if reference_conditions is not None and not isinstance(reference_conditions, list):
            reference_conditions = [reference_conditions]

        # Infer noise scale from sigma schedule if not provided
        if noise_scale is None:
            noise_scale = sigmas[0] if sigmas is not None else 1.0

        device = self._execution_device

        # 3. Prepare text embeddings
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

        # 4. Prepare latent variables
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None:
            logger.info(
                "Got latents of shape [batch_size, latent_dim, latent_frames, latent_height, latent_width],"
                " `latent_num_frames`, `latent_height`, `latent_width` will be inferred."
            )
            _, _, latent_num_frames, latent_height, latent_width = latents.shape

        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, clean_latents, appended_coords, num_ref_tokens, ref_cross_mask = (
            self.prepare_latents(
                conditions=conditions,
                reference_conditions=reference_conditions,
                reference_downscale_factor=reference_downscale_factor,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
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
        )
        # Track the base token count in the generated video, excluding any appended keyframe and reference-video
        # condition tokens.
        base_token_count = latents.shape[1] - (appended_coords.shape[2] if appended_coords is not None else 0)

        has_conditions = conditions is not None and len(conditions) > 0
        has_appended_tokens = appended_coords is not None
        if self.do_classifier_free_guidance and (has_conditions or num_ref_tokens > 0):
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])

        # Build a video self-attention mask over three groups: (1) the noisy latents (2) keyframe conditions, if any
        # and (3) reference conditions, if any. Tokens are attend to each other across groups as follows:
        #   - TODO
        video_self_attention_mask: torch.Tensor | None = None
        if ref_cross_mask is not None:
            num_noisy_tokens = latents.shape[1] - num_ref_tokens
            video_self_attention_mask = self._build_video_self_attention_mask(
                num_noisy_tokens=num_noisy_tokens,
                extras_cross_masks=[ref_cross_mask],
                device=device,
            )

        # 5. Prepare audio latents
        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        if audio_latents is not None:
            logger.info(
                "Got audio_latents of shape [batch_size, num_channels, audio_num_frames, mel_bins],"
                " `audio_num_frames` will be inferred."
            )
            _, _, audio_num_frames, _ = audio_latents.shape

        latent_mel_bins = self.audio_mel_bins // self.audio_vae_mel_compression_ratio
        audio_latents = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents=self.audio_latent_channels,
            audio_latent_length=audio_num_frames,
            num_mel_bins=self.audio_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
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
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                audio_latent_model_input = (
                    torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                )
                audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

                timestep_scalar = t.expand(latent_model_input.shape[0])

                if has_conditions or num_ref_tokens > 0:
                    video_timestep = timestep_scalar.unsqueeze(-1) * (1 - conditioning_mask.squeeze(-1))
                else:
                    video_timestep = timestep_scalar.unsqueeze(-1).expand(-1, video_seq_len)

                t_audio = audio_timesteps[i]
                audio_timestep = t_audio.expand(latent_model_input.shape[0])

                # --- Main transformer forward pass (conditional + unconditional for CFG) ---
                if video_self_attention_mask is not None:
                    video_self_attention_mask = video_self_attention_mask.expand(latent_model_input.shape[0], -1, -1)
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
                        video_self_attention_mask=video_self_attention_mask,
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
                noise_pred_audio = noise_pred_audio.float()

                if self.do_classifier_free_guidance:
                    noise_pred_video_uncond_text, noise_pred_video = noise_pred_video.chunk(2)
                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)
                    noise_pred_video_uncond_text = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_text, i, self.scheduler
                    )
                    video_cfg_delta = (self.guidance_scale - 1) * (noise_pred_video - noise_pred_video_uncond_text)

                    noise_pred_audio_uncond_text, noise_pred_audio = noise_pred_audio.chunk(2)
                    noise_pred_audio = self.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)
                    noise_pred_audio_uncond_text = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_text, i, audio_scheduler
                    )
                    audio_cfg_delta = (self.audio_guidance_scale - 1) * (
                        noise_pred_audio - noise_pred_audio_uncond_text
                    )

                    if self.do_spatio_temporal_guidance or self.do_modality_isolation_guidance:
                        if i == 0:
                            video_prompt_embeds = connector_prompt_embeds.chunk(2, dim=0)[1]
                            audio_prompt_embeds = connector_audio_prompt_embeds.chunk(2, dim=0)[1]
                            prompt_attn_mask = connector_attention_mask.chunk(2, dim=0)[1]

                            video_pos_ids = video_coords.chunk(2, dim=0)[0]
                            audio_pos_ids = audio_coords.chunk(2, dim=0)[0]

                        timestep_scalar_single = timestep_scalar.chunk(2, dim=0)[0]
                        if has_conditions or num_ref_tokens > 0:
                            video_timestep_single = video_timestep.chunk(2, dim=0)[0]
                        else:
                            video_timestep_single = timestep_scalar_single.unsqueeze(-1).expand(-1, video_seq_len)
                        audio_timestep_single = audio_timestep.chunk(2, dim=0)[0]
                else:
                    video_cfg_delta = audio_cfg_delta = 0

                    video_prompt_embeds = connector_prompt_embeds
                    audio_prompt_embeds = connector_audio_prompt_embeds
                    prompt_attn_mask = connector_attention_mask

                    video_pos_ids = video_coords
                    audio_pos_ids = audio_coords

                    timestep_scalar_single = timestep_scalar
                    if has_conditions or num_ref_tokens > 0:
                        video_timestep_single = video_timestep
                    else:
                        video_timestep_single = timestep_scalar.unsqueeze(-1).expand(-1, video_seq_len)
                    audio_timestep_single = audio_timestep

                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)
                    noise_pred_audio = self.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)

                # --- STG forward pass ---
                if self.do_spatio_temporal_guidance:
                    if video_self_attention_mask is not None:
                        video_self_attention_mask = video_self_attention_mask.expand(latents.shape[0], -1, -1)
                    with self.transformer.cache_context("uncond_stg"):
                        noise_pred_video_uncond_stg, noise_pred_audio_uncond_stg = self.transformer(
                            hidden_states=latents.to(dtype=prompt_embeds.dtype),
                            audio_hidden_states=audio_latents.to(dtype=prompt_embeds.dtype),
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep_single,
                            audio_timestep=audio_timestep_single,
                            sigma=timestep_scalar_single,  # Used by LTX-2.3
                            audio_sigma=audio_timestep_single,
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            video_self_attention_mask=video_self_attention_mask,
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
                    noise_pred_audio_uncond_stg = noise_pred_audio_uncond_stg.float()
                    noise_pred_video_uncond_stg = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_stg, i, self.scheduler
                    )
                    noise_pred_audio_uncond_stg = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_stg, i, audio_scheduler
                    )

                    video_stg_delta = self.stg_scale * (noise_pred_video - noise_pred_video_uncond_stg)
                    audio_stg_delta = self.audio_stg_scale * (noise_pred_audio - noise_pred_audio_uncond_stg)
                else:
                    video_stg_delta = audio_stg_delta = 0

                # --- Modality isolation guidance forward pass ---
                if self.do_modality_isolation_guidance:
                    if video_self_attention_mask is not None:
                        video_self_attention_mask = video_self_attention_mask.expand(latents.shape[0], -1, -1)
                    with self.transformer.cache_context("uncond_modality"):
                        noise_pred_video_uncond_mod, noise_pred_audio_uncond_mod = self.transformer(
                            hidden_states=latents.to(dtype=prompt_embeds.dtype),
                            audio_hidden_states=audio_latents.to(dtype=prompt_embeds.dtype),
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep_single,
                            audio_timestep=audio_timestep_single,
                            sigma=timestep_scalar_single,  # Used by LTX-2.3
                            audio_sigma=audio_timestep_single,
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            video_self_attention_mask=video_self_attention_mask,
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
                    noise_pred_audio_uncond_mod = noise_pred_audio_uncond_mod.float()
                    noise_pred_video_uncond_mod = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_mod, i, self.scheduler
                    )
                    noise_pred_audio_uncond_mod = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_mod, i, audio_scheduler
                    )

                    video_modality_delta = (self.modality_scale - 1) * (noise_pred_video - noise_pred_video_uncond_mod)
                    audio_modality_delta = (self.audio_modality_scale - 1) * (
                        noise_pred_audio - noise_pred_audio_uncond_mod
                    )
                else:
                    video_modality_delta = audio_modality_delta = 0

                # Apply all guidance terms
                noise_pred_video_g = noise_pred_video + video_cfg_delta + video_stg_delta + video_modality_delta
                noise_pred_audio_g = noise_pred_audio + audio_cfg_delta + audio_stg_delta + audio_modality_delta

                # Apply guidance rescaling
                if self.guidance_rescale > 0:
                    noise_pred_video = rescale_noise_cfg(
                        noise_pred_video_g, noise_pred_video, guidance_rescale=self.guidance_rescale
                    )
                else:
                    noise_pred_video = noise_pred_video_g

                if self.audio_guidance_rescale > 0:
                    noise_pred_audio = rescale_noise_cfg(
                        noise_pred_audio_g, noise_pred_audio, guidance_rescale=self.audio_guidance_rescale
                    )
                else:
                    noise_pred_audio = noise_pred_audio_g

                # Apply frame conditioning mask: blend denoised x0 with clean condition latents
                if has_conditions:
                    bsz = noise_pred_video.size(0)
                    denoised_sample_cond = (
                        noise_pred_video * (1 - conditioning_mask[:bsz])
                        + clean_latents.float() * conditioning_mask[:bsz]
                    ).to(noise_pred_video.dtype)
                    noise_pred_video = denoised_sample_cond

                # Convert back to velocity for scheduler
                noise_pred_video = self.convert_x0_to_velocity(latents, noise_pred_video, i, self.scheduler)
                noise_pred_audio = self.convert_x0_to_velocity(audio_latents, noise_pred_audio, i, audio_scheduler)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
                audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

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
        # Trim any appended keyframe or reference tokens from the latents to recover the generated video only.
        latents = latents[:, :base_token_count]
        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)

        if output_type == "latent":
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            video = latents
            audio = audio_latents
        else:
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

            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )

            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            audio_latents = audio_latents.to(self.audio_vae.dtype)
            generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = self.vocoder(generated_mel_spectrograms)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)

        return LTX2PipelineOutput(frames=video, audio=audio)
