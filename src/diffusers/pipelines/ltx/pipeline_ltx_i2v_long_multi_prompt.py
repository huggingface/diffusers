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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import T5EncoderModel, T5TokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTXVideo
from ...models.transformers import LTXVideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler, LTXEulerAncestralRFScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
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
        >>> import torch
        >>> from diffusers import LTXEulerAncestralRFScheduler, LTXI2VLongMultiPromptPipeline

        >>> pipe = LTXI2VLongMultiPromptPipeline.from_pretrained("LTX-Video-0.9.8-13B-distilled")
        >>> # For ComfyUI parity, swap in the RF scheduler (keeps the original config).
        >>> pipe.scheduler = LTXEulerAncestralRFScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda").to(dtype=torch.bfloat16)
        >>> # Example A: get decoded frames (PIL)
        >>> out = pipe(
        ...     prompt="a chimpanzee walks | a chimpanzee eats",
        ...     num_frames=161,
        ...     height=512,
        ...     width=704,
        ...     temporal_tile_size=80,
        ...     temporal_overlap=24,
        ...     output_type="pil",
        ...     return_dict=True,
        ... )
        >>> frames = out.frames[0]  # list of PIL.Image.Image
        >>> # Example B: get latent video and decode later (saves VRAM during sampling)
        >>> out_latent = pipe(prompt="a chimpanzee walking", output_type="latent", return_dict=True).frames
        >>> frames = pipe.vae_decode_tiled(out_latent, output_type="pil")[0]
        ```
"""


def get_latent_coords(
    latent_num_frames, latent_height, latent_width, batch_size, device, rope_interpolation_scale, latent_idx
):
    """
    Compute latent patch top-left coordinates in (t, y, x) order.

    Args:
      latent_num_frames: int. Number of latent frames (T_lat).
      latent_height: int. Latent height (H_lat).
      latent_width: int. Latent width (W_lat).
      batch_size: int. Batch dimension (B).
      device: torch.device for the resulting tensor.
      rope_interpolation_scale:
          tuple[int|float, int|float, int|float]. Scale per (t, y, x) latent step to pixel coords.
      latent_idx: Optional[int]. When not None, shifts the time coordinate to align segments:
        - <= 0 uses step multiples of rope_interpolation_scale[0]
        - > 0 starts at 1 then increments by rope_interpolation_scale[0]

    Returns:
      Tensor of shape [B, 3, T_lat * H_lat * W_lat] containing top-left coordinates per latent patch, repeated for each
      batch element.
    """
    latent_sample_coords = torch.meshgrid(
        torch.arange(0, latent_num_frames, 1, device=device),
        torch.arange(0, latent_height, 1, device=device),
        torch.arange(0, latent_width, 1, device=device),
        indexing="ij",
    )
    latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
    latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    latent_coords = latent_coords.flatten(2)
    pixel_coords = latent_coords * torch.tensor(rope_interpolation_scale, device=latent_coords.device)[None, :, None]
    if latent_idx is not None:
        if latent_idx <= 0:
            frame_idx = latent_idx * rope_interpolation_scale[0]
        else:
            frame_idx = 1 + (latent_idx - 1) * rope_interpolation_scale[0]
        if frame_idx == 0:
            pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - rope_interpolation_scale[0]).clamp(min=0)
        pixel_coords[:, 0] += frame_idx
    return pixel_coords


# Copied from diffusers.pipelines.ltx.pipeline_ltx.rescale_noise_cfg
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


def adain_normalize_latents(
    curr_latents: torch.Tensor, ref_latents: Optional[torch.Tensor], factor: float
) -> torch.Tensor:
    """
    Optional AdaIN normalization: channel-wise mean/variance matching of curr_latents to ref_latents, controlled by
    factor.

    Args:
      curr_latents: Tensor [B, C, T, H, W]. Current window latents.
      ref_latents:
          Optional[Tensor] [B, C, T_ref, H, W]. Reference latents (e.g., first window) used to compute target stats.
      factor: float in [0, 1]. 0 keeps current stats; 1 matches reference stats.

    Returns:
      Tensor with per-channel mean/std blended towards the reference.
    """
    if ref_latents is None or factor is None or factor <= 0:
        return curr_latents

    eps = torch.tensor(1e-6, device=curr_latents.device, dtype=curr_latents.dtype)

    # Compute per-channel means/stds for current and reference over (T, H, W)
    mu_curr = curr_latents.mean(dim=(2, 3, 4), keepdim=True)
    sigma_curr = curr_latents.std(dim=(2, 3, 4), keepdim=True)

    mu_ref = ref_latents.mean(dim=(2, 3, 4), keepdim=True).to(device=curr_latents.device, dtype=curr_latents.dtype)
    sigma_ref = ref_latents.std(dim=(2, 3, 4), keepdim=True).to(device=curr_latents.device, dtype=curr_latents.dtype)

    # Blend target statistics
    mu_blend = (1.0 - float(factor)) * mu_curr + float(factor) * mu_ref
    sigma_blend = (1.0 - float(factor)) * sigma_curr + float(factor) * sigma_ref
    sigma_blend = torch.clamp(sigma_blend, min=float(eps))

    # Apply AdaIN
    curr_norm = (curr_latents - mu_curr) / (sigma_curr + eps)
    return curr_norm * sigma_blend + mu_blend


def split_into_temporal_windows(
    latent_len: int, temporal_tile_size: int, temporal_overlap: int, compression: int
) -> List[Tuple[int, int]]:
    """
    Split latent frames into sliding windows.

    Args:
      latent_len: int. Number of latent frames (T_lat).
      temporal_tile_size: int. Window size in latent frames (> 0).
      temporal_overlap: int. Overlap between windows in latent frames (>= 0).
      compression: int. VAE temporal compression ratio (unused here; kept for parity).

    Returns:
      list[tuple[int, int]]: inclusive-exclusive (start, end) indices per window.
    """
    if temporal_tile_size <= 0:
        raise ValueError("temporal_tile_size must be > 0")
    stride = max(temporal_tile_size - temporal_overlap, 1)
    windows = []
    start = 0
    while start < latent_len:
        end = min(start + temporal_tile_size, latent_len)
        windows.append((start, end))
        if end == latent_len:
            break
        start = start + stride
    return windows


def linear_overlap_fuse(prev: torch.Tensor, new: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Temporal linear crossfade between two latent clips over the overlap region.

    Args:
      prev: Tensor [B, C, F, H, W]. Previous output segment.
      new: Tensor [B, C, F, H, W]. New segment to be appended.
      overlap: int. Number of frames to crossfade (overlap <= 1 concatenates without blend).

    Returns:
      Tensor [B, C, F_prev + F_new - overlap, H, W] after crossfade at the seam.
    """
    if overlap <= 1:
        return torch.cat([prev, new], dim=2)
    alpha = torch.linspace(1, 0, overlap + 2, device=prev.device, dtype=prev.dtype)[1:-1]
    shape = [1] * prev.ndim
    shape[2] = alpha.size(0)
    alpha = alpha.reshape(shape)
    blended = alpha * prev[:, :, -overlap:] + (1 - alpha) * new[:, :, :overlap]
    return torch.cat([prev[:, :, :-overlap], blended, new[:, :, overlap:]], dim=2)


def inject_prev_tail_latents(
    window_latents: torch.Tensor,
    prev_tail_latents: Optional[torch.Tensor],
    window_cond_mask_5d: torch.Tensor,
    overlap_lat: int,
    strength: Optional[float],
    prev_overlap_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Inject the tail latents from the previous window at the beginning of the current window (first k frames), where k =
    min(overlap_lat, T_curr, T_prev_tail).

    Args:
      window_latents: Tensor [B, C, T, H, W]. Current window latents.
      prev_tail_latents: Optional[Tensor] [B, C, T_prev, H, W]. Tail segment from the previous window.
      window_cond_mask_5d: Tensor [B, 1, T, H, W]. Per-token conditioning mask (1 = free, 0 = hard condition).
      overlap_lat: int. Number of latent frames to inject from the previous tail.
      strength: Optional[float] in [0, 1]. Blend strength; 1.0 replaces, 0.0 keeps original.
      prev_overlap_len: int. Accumulated overlap length so far (used for trimming later).

    Returns:
      Tuple[Tensor, Tensor, int]: (updated_window_latents, updated_cond_mask, updated_prev_overlap_len)
    """
    if prev_tail_latents is None or overlap_lat <= 0 or strength is None or strength <= 0:
        return window_latents, window_cond_mask_5d, prev_overlap_len

    # Expected shape: [B, C, T, H, W]
    T = int(window_latents.shape[2])
    k = min(int(overlap_lat), T, int(prev_tail_latents.shape[2]))
    if k <= 0:
        return window_latents, window_cond_mask_5d, prev_overlap_len

    tail = prev_tail_latents[:, :, -k:]
    mask = torch.full(
        (window_cond_mask_5d.shape[0], 1, tail.shape[2], window_cond_mask_5d.shape[3], window_cond_mask_5d.shape[4]),
        1.0 - strength,
        dtype=window_cond_mask_5d.dtype,
        device=window_cond_mask_5d.device,
    )

    window_latents = torch.cat([window_latents, tail], dim=2)
    window_cond_mask_5d = torch.cat([window_cond_mask_5d, mask], dim=2)
    return window_latents, window_cond_mask_5d, prev_overlap_len + k


def build_video_coords_for_window(
    latents: torch.Tensor,
    overlap_len: int,
    guiding_len: int,
    negative_len: int,
    rope_interpolation_scale: torch.Tensor,
    frame_rate: int,
) -> torch.Tensor:
    """
    Build video_coords: [B, 3, S] with order [t, y, x].

    Args:
      latents: Tensor [B, C, T, H, W]. Current window latents (before any trimming).
      overlap_len: int. Number of frames from previous tail injected at the head.
      guiding_len: int. Number of guidance frames appended at the head.
      negative_len: int. Number of negative-index frames appended at the head (typically 1 or 0).
      rope_interpolation_scale: tuple[int|float, int|float, int|float]. Scale for (t, y, x).
      frame_rate: int. Used to convert time indices into seconds (t /= frame_rate).

    Returns:
      Tensor [B, 3, T*H*W] of fractional pixel coordinates per latent patch.
    """

    b, c, f, h, w = latents.shape
    pixel_coords = get_latent_coords(f, h, w, b, latents.device, rope_interpolation_scale, 0)
    replace_corrds = []
    if overlap_len > 0:
        replace_corrds.append(get_latent_coords(overlap_len, h, w, b, latents.device, rope_interpolation_scale, 0))
    if guiding_len > 0:
        replace_corrds.append(
            get_latent_coords(guiding_len, h, w, b, latents.device, rope_interpolation_scale, overlap_len)
        )
    if negative_len > 0:
        replace_corrds.append(get_latent_coords(negative_len, h, w, b, latents.device, rope_interpolation_scale, -1))
    if len(replace_corrds) > 0:
        replace_corrds = torch.cat(replace_corrds, axis=2)
        pixel_coords[:, :, -replace_corrds.shape[2] :] = replace_corrds
    fractional_coords = pixel_coords.to(torch.float32)
    fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)
    return fractional_coords


def parse_prompt_segments(prompt: Union[str, List[str]], prompt_segments: Optional[List[Dict[str, Any]]]) -> List[str]:
    """
    Return a list of positive prompts per window index.

    Args:
      prompt: str | list[str]. If str contains '|', parts are split by bars and trimmed.
      prompt_segments:
          list[dict], optional. Each dict with {"start_window", "end_window", "text"} overrides prompts per window.

    Returns:
      list[str] containing the positive prompt for each window index.
    """
    if prompt is None:
        return []
    if prompt_segments:
        max_w = 0
        for seg in prompt_segments:
            max_w = max(max_w, int(seg.get("end_window", 0)))
        texts = [""] * (max_w + 1)
        for seg in prompt_segments:
            s = int(seg.get("start_window", 0))
            e = int(seg.get("end_window", s))
            txt = seg.get("text", "")
            for w in range(s, e + 1):
                texts[w] = txt
        # fill empty by last non-empty
        last = ""
        for i in range(len(texts)):
            if texts[i] == "":
                texts[i] = last
            else:
                last = texts[i]
        return texts

    # bar-split mode
    if isinstance(prompt, str):
        parts = [p.strip() for p in prompt.split("|")]
    else:
        parts = prompt
    parts = [p for p in parts if p is not None]
    return parts


def batch_normalize(latents, reference, factor):
    """
    Batch AdaIN-like normalization for latents in dict format (ComfyUI-compatible).

    Args:
        latents: dict containing "samples" shaped [B, C, F, H, W]
        reference: dict containing "samples" used to compute target stats
        factor: float in [0, 1]; 0 = no change, 1 = full match to reference
    Returns:
        Tuple[dict]: a single-element tuple with the updated latents dict.
    """
    latents_copy = copy.deepcopy(latents)
    t = latents_copy["samples"]  #  B x C x F x H x W

    for i in range(t.size(0)):  # batch
        for c in range(t.size(1)):  # channel
            r_sd, r_mean = torch.std_mean(reference["samples"][i, c], dim=None)  # index by original dim order
            i_sd, i_mean = torch.std_mean(t[i, c], dim=None)

            t[i, c] = ((t[i, c] - i_mean) / i_sd) * r_sd + r_mean

    latents_copy["samples"] = torch.lerp(latents["samples"], t, factor)
    return (latents_copy,)


class LTXI2VLongMultiPromptPipeline(DiffusionPipeline, FromSingleFileMixin, LTXVideoLoraLoaderMixin):
    r"""
    Long-duration I2V (image-to-video) multi-prompt pipeline with ComfyUI parity.

    Key features:
    - Temporal sliding-window sampling only (no spatial H/W sharding); autoregressive fusion across windows.
    - Multi-prompt segmentation per window with smooth transitions at window heads.
    - First-frame hard conditioning via per-token mask for I2V.
    - VRAM control via temporal windowing and VAE tiled decoding.

    Reference: https://github.com/Lightricks/LTX-Video

    Args:
        transformer ([`LTXVideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`] or [`LTXEulerAncestralRFScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLLTXVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
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
        if not isinstance(scheduler, LTXEulerAncestralRFScheduler):
            logger.warning(
                "For ComfyUI parity, `LTXI2VLongMultiPromptPipeline` is typically run with "
                "`LTXEulerAncestralRFScheduler`. Got %s.",
                scheduler.__class__.__name__,
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
            self.transformer.config.patch_size_t if getattr(self, "transformer", None) is not None else 1
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 128
        )

        self.default_height = 512
        self.default_width = 704
        self.default_frames = 121
        self._current_tile_T = None

    @property
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.guidance_scale
    def guidance_scale(self):
        return self._guidance_scale

    @property
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.guidance_rescale
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.do_classifier_free_guidance
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.num_timesteps
    def num_timesteps(self):
        return self._num_timesteps

    @property
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.current_timestep
    def current_timestep(self):
        return self._current_timestep

    @property
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.attention_kwargs
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.interrupt
    def interrupt(self):
        return self._interrupt

    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 128,
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

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.encode_prompt
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
        max_sequence_length: int = 128,
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

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        generator: Optional[torch.Generator],
        dtype: torch.dtype = torch.float32,
        latents: Optional[torch.Tensor] = None,
        cond_latents: Optional[torch.Tensor] = None,
        cond_strength: float = 0.0,
        negative_index_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, int]:
        """
        Prepare base latents and optionally inject first-frame conditioning latents.

        Returns:
          latents, negative_index_latents, latent_num_frames, latent_height, latent_width
        """
        if latents is None:
            latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
            latent_height = height // self.vae_spatial_compression_ratio
            latent_width = width // self.vae_spatial_compression_ratio
            latents = torch.zeros(
                (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width),
                device=device,
                dtype=dtype,
            )
        else:
            latent_num_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]
            latents = latents.to(device=device, dtype=dtype)

        if cond_latents is not None and cond_strength > 0:
            if negative_index_latents is None:
                negative_index_latents = cond_latents
            latents[:, :, :1, :, :] = cond_latents

        return latents, negative_index_latents, latent_num_frames, latent_height, latent_width

    # TODO: refactor this out
    @torch.no_grad()
    def vae_decode_tiled(
        self,
        latents: torch.Tensor,
        decode_timestep: Optional[float] = None,
        decode_noise_scale: Optional[float] = None,
        horizontal_tiles: int = 4,
        vertical_tiles: int = 4,
        overlap: int = 3,
        last_frame_fix: bool = True,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pt",
        auto_denormalize: bool = True,
        compute_dtype: torch.dtype = torch.float32,
        enable_vae_tiling: bool = False,
    ) -> Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]:
        """
        VAE-based spatial tiled decoding (ComfyUI parity) implemented in Diffusers style.
        - Linearly feather and blend overlapping tiles to avoid seams.
        - Optional last_frame_fix: duplicate the last latent frame before decoding, then drop time_scale_factor frames
          at the end.
        - Supports timestep_conditioning and decode_noise_scale injection.
        - By default, "normalized latents" (the denoising output) are de-normalized internally (auto_denormalize=True).
        - Tile fusion is computed in compute_dtype (float32 by default) to reduce blur and color shifts.

        Args:
          latents: [B, C_latent, F_latent, H_latent, W_latent]
          decode_timestep: Optional decode timestep (effective only if VAE supports timestep_conditioning)
          decode_noise_scale:
              Optional decode noise interpolation (effective only if VAE supports timestep_conditioning)
          horizontal_tiles, vertical_tiles: Number of tiles horizontally/vertically (>= 1)
          overlap: Overlap in latent space (in latent pixels, >= 0)
          last_frame_fix: Whether to enable the "repeat last frame" fix
          generator: Random generator (used for decode_noise_scale noise)
          output_type: "latent" | "pt" | "np" | "pil"
            - "latent": return latents unchanged (useful for downstream processing)
            - "pt": return tensor in VAE output space
            - "np"/"pil": post-processed outputs via VideoProcessor.postprocess_video
          auto_denormalize: If True, apply LTX de-normalization to `latents` internally (recommended)
          compute_dtype: Precision used during tile fusion (float32 default; significantly reduces seam blur)
          enable_vae_tiling: If True, delegate tiling to VAE's built-in `tiled_decode` (sets `vae.use_tiling`).

        Returns:
          - If output_type="latent": returns input `latents` unchanged
          - If output_type="pt": returns [B, C, F, H, W] (values roughly in [-1, 1])
          - If output_type="np"/"pil": returns post-processed outputs via postprocess_video
        """
        if output_type == "latent":
            return latents
        if horizontal_tiles < 1 or vertical_tiles < 1:
            raise ValueError("horizontal_tiles and vertical_tiles must be >= 1")
        overlap = max(int(overlap), 0)

        # Device and precision
        device = self._execution_device
        latents = latents.to(device=device, dtype=compute_dtype)

        # De-normalize to VAE space (avoid color artifacts)
        if auto_denormalize:
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
        # dtype required for VAE forward pass
        latents = latents.to(dtype=self.vae.dtype)

        # Temporal/spatial upscaling ratios (parity with ComfyUI's downscale_index_formula)
        tsf = int(self.vae_temporal_compression_ratio)
        sf = int(self.vae_spatial_compression_ratio)

        # Optional: last_frame_fix (repeat last latent frame)
        if last_frame_fix:
            latents = torch.cat([latents, latents[:, :, -1:].contiguous()], dim=2)

        b, c_lat, f_lat, h_lat, w_lat = latents.shape
        f_out = 1 + (f_lat - 1) * tsf
        h_out = h_lat * sf
        w_out = w_lat * sf

        # timestep_conditioning + decode-time noise injection (aligned with pipeline)
        if getattr(self.vae.config, "timestep_conditioning", False):
            dt = float(decode_timestep) if decode_timestep is not None else 0.0
            vt = torch.tensor([dt], device=device, dtype=latents.dtype)
            if decode_noise_scale is not None:
                dns = torch.tensor([float(decode_noise_scale)], device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                latents = (1 - dns) * latents + dns * noise
        else:
            vt = None

        if enable_vae_tiling and hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
            decoded = self.vae.decode(latents, vt, return_dict=False)[0]
            if last_frame_fix:
                decoded = decoded[:, :, :-tsf, :, :]
            if output_type in ("np", "pil"):
                return self.video_processor.postprocess_video(decoded, output_type=output_type)
            return decoded

        # Compute base tile sizes (in latent space)
        base_tile_h = (h_lat + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_w = (w_lat + (horizontal_tiles - 1) * overlap) // horizontal_tiles

        output: Optional[torch.Tensor] = None  # [B, C_img, F, H, W], fused using compute_dtype
        weights: Optional[torch.Tensor] = None  # [B, 1, F, H, W], fused using compute_dtype

        # Iterate tiles in latent space (no temporal tiling)
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                h_start = h * (base_tile_w - overlap)
                v_start = v * (base_tile_h - overlap)

                h_end = min(h_start + base_tile_w, w_lat) if h < horizontal_tiles - 1 else w_lat
                v_end = min(v_start + base_tile_h, h_lat) if v < vertical_tiles - 1 else h_lat

                # Slice latent tile and decode
                tile_latents = latents[:, :, :, v_start:v_end, h_start:h_end]
                decoded_tile = self.vae.decode(tile_latents, vt, return_dict=False)[0]  # [B, C, F, Ht, Wt]
                # Cast to high precision to reduce blending blur
                decoded_tile = decoded_tile.to(dtype=compute_dtype)

                # Initialize output buffers (compute_dtype)
                if output is None:
                    output = torch.zeros(
                        (b, decoded_tile.shape[1], f_out, h_out, w_out),
                        device=decoded_tile.device,
                        dtype=compute_dtype,
                    )
                    weights = torch.zeros(
                        (b, 1, f_out, h_out, w_out),
                        device=decoded_tile.device,
                        dtype=compute_dtype,
                    )

                # Tile placement in output pixel space
                out_h_start = v_start * sf
                out_h_end = v_end * sf
                out_w_start = h_start * sf
                out_w_end = h_end * sf

                tile_out_h = out_h_end - out_h_start
                tile_out_w = out_w_end - out_w_start

                # Linear feathering weights [B, 1, F, Ht, Wt] (compute_dtype)
                tile_weights = torch.ones(
                    (b, 1, decoded_tile.shape[2], tile_out_h, tile_out_w),
                    device=decoded_tile.device,
                    dtype=compute_dtype,
                )

                overlap_out_h = overlap * sf
                overlap_out_w = overlap * sf

                # Horizontal feathering: left/right overlaps
                if overlap_out_w > 0:
                    if h > 0:
                        h_blend = torch.linspace(
                            0, 1, steps=overlap_out_w, device=decoded_tile.device, dtype=compute_dtype
                        )
                        tile_weights[:, :, :, :, :overlap_out_w] *= h_blend.view(1, 1, 1, 1, -1)
                    if h < horizontal_tiles - 1:
                        h_blend = torch.linspace(
                            1, 0, steps=overlap_out_w, device=decoded_tile.device, dtype=compute_dtype
                        )
                        tile_weights[:, :, :, :, -overlap_out_w:] *= h_blend.view(1, 1, 1, 1, -1)

                # Vertical feathering: top/bottom overlaps
                if overlap_out_h > 0:
                    if v > 0:
                        v_blend = torch.linspace(
                            0, 1, steps=overlap_out_h, device=decoded_tile.device, dtype=compute_dtype
                        )
                        tile_weights[:, :, :, :overlap_out_h, :] *= v_blend.view(1, 1, 1, -1, 1)
                    if v < vertical_tiles - 1:
                        v_blend = torch.linspace(
                            1, 0, steps=overlap_out_h, device=decoded_tile.device, dtype=compute_dtype
                        )
                        tile_weights[:, :, :, -overlap_out_h:, :] *= v_blend.view(1, 1, 1, -1, 1)

                # Accumulate blended tile
                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += decoded_tile * tile_weights
                weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += tile_weights

        # Normalize, then clamp to [-1, 1] in compute_dtype to avoid color artifacts
        output = output / (weights + 1e-8)
        output = output.clamp(-1.0, 1.0)
        output = output.to(dtype=self.vae.dtype)

        # Optional: drop the last tsf frames after last_frame_fix
        if last_frame_fix:
            output = output[:, :, :-tsf, :, :]

        if output_type in ("np", "pil"):
            return self.video_processor.postprocess_video(output, output_type=output_type)
        return output

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_segments: Optional[List[Dict[str, Any]]] = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: float = 25,
        guidance_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: Optional[int] = 8,
        sigmas: Optional[Union[List[float], torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        seed: Optional[int] = 0,
        cond_image: Optional[Union["PIL.Image.Image", torch.Tensor]] = None,
        cond_strength: float = 0.5,
        latents: Optional[torch.Tensor] = None,
        temporal_tile_size: int = 80,
        temporal_overlap: int = 24,
        temporal_overlap_cond_strength: float = 0.5,
        adain_factor: float = 0.25,
        guidance_latents: Optional[torch.Tensor] = None,
        guiding_strength: float = 1.0,
        negative_index_latents: Optional[torch.Tensor] = None,
        negative_index_strength: float = 1.0,
        skip_steps_sigma_threshold: Optional[float] = 1,
        decode_timestep: Optional[float] = 0.05,
        decode_noise_scale: Optional[float] = 0.025,
        decode_horizontal_tiles: int = 4,
        decode_vertical_tiles: int = 4,
        decode_overlap: int = 3,
        output_type: Optional[str] = "latent",  # "latent" | "pt" | "np" | "pil"
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
    ):
        r"""
        Generate an image-to-video sequence via temporal sliding windows and multi-prompt scheduling.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                Positive text prompt(s) per window. If a single string contains '|', parts are split by bars.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) to suppress undesired content.
            prompt_segments (`List[dict]`, *optional*):
                Segment mapping with {"start_window", "end_window", "text"} to override prompts per window.
            height (`int`, defaults to `512`):
                Output image height in pixels; must be divisible by 32.
            width (`int`, defaults to `704`):
                Output image width in pixels; must be divisible by 32.
            num_frames (`int`, defaults to `161`):
                Number of output frames (in decoded pixel space).
            frame_rate (`float`, defaults to `25`):
                Frames-per-second; used to normalize temporal coordinates in `video_coords`.
            guidance_scale (`float`, defaults to `1.0`):
                CFG scale; values > 1 enable classifier-free guidance.
            guidance_rescale (`float`, defaults to `0.0`):
                Optional rescale to mitigate overexposure under CFG (see `rescale_noise_cfg`).
            num_inference_steps (`int`, *optional*, defaults to `8`):
                Denoising steps per window. Ignored if `sigmas` is provided.
            sigmas (`List[float]` or `torch.Tensor`, *optional*):
                Explicit sigma schedule per window; if set, overrides `num_inference_steps`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Controls stochasticity; list accepted but first element is used (batch=1).
            seed (`int`, *optional*, defaults to `0`):
                If provided, seeds the shared generator for global latents and derives a window-local generator with
                `seed + w_start` per temporal window.
            cond_image (`PIL.Image.Image` or `torch.Tensor`, *optional*):
                Conditioning image; fixes frame 0 via per-token mask when `cond_strength > 0`.
            cond_strength (`float`, defaults to `0.5`):
                Strength of first-frame hard conditioning (smaller cond_mask ⇒ stronger preservation).
            latents (`torch.Tensor`, *optional*):
                Initial latents [B, C_lat, F_lat, H_lat, W_lat]; if None, sampled with `randn_tensor`.
            temporal_tile_size (`int`, defaults to `80`):
                Temporal window size (in decoded frames); internally scaled by VAE temporal compression.
            temporal_overlap (`int`, defaults to `24`):
                Overlap between consecutive windows (in decoded frames); internally scaled by compression.
            temporal_overlap_cond_strength (`float`, defaults to `0.5`):
                Strength for injecting previous window tail latents at new window head.
            adain_factor (`float`, defaults to `0.25`):
                AdaIN normalization strength for cross-window consistency (0 disables).
            guidance_latents (`torch.Tensor`, *optional*):
                Reference latents injected at window head; length trimmed by overlap for subsequent windows.
            guiding_strength (`float`, defaults to `1.0`):
                Injection strength for `guidance_latents`.
            negative_index_latents (`torch.Tensor`, *optional*):
                A single-frame latent appended at window head for "negative index" semantics.
            negative_index_strength (`float`, defaults to `1.0`):
                Injection strength for `negative_index_latents`.
            skip_steps_sigma_threshold (`float`, *optional*, defaults to `1`):
                Skip steps whose sigma exceeds this threshold.
            decode_timestep (`float`, *optional*, defaults to `0.05`):
                Decode-time timestep (if VAE supports timestep_conditioning).
            decode_noise_scale (`float`, *optional*, defaults to `0.025`):
                Decode-time noise mix scale (if VAE supports timestep_conditioning).
            decode_horizontal_tiles (`int`, defaults to `4`):
                Number of horizontal tiles during VAE decoding.
            decode_vertical_tiles (`int`, defaults to `4`):
                Number of vertical tiles during VAE decoding.
            decode_overlap (`int`, defaults to `3`):
                Overlap (in latent pixels) between tiles during VAE decoding.
            output_type (`str`, *optional*, defaults to `"latent"`):
                The output format of the generated video. Choose between "latent", "pt", "np", or "pil". If "latent",
                returns latents without decoding.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Extra attention parameters forwarded to the transformer.
            callback_on_step_end (`PipelineCallback` or `MultiPipelineCallbacks`, *optional*):
                Per-step callback hook.
            callback_on_step_end_tensor_inputs (`List[str]`, defaults to `["latents"]`):
                Keys from locals() to pass into the callback.
            max_sequence_length (`int`, defaults to `128`):
                Tokenizer max length for prompt encoding.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated frames. The output format depends on
                `output_type`:
                - "latent"/"pt": `torch.Tensor` [B, C, F, H, W]; "latent" is in normalized latent space, "pt" is VAE
                  output space.
                - "np": `np.ndarray` post-processed.
                - "pil": `List[PIL.Image.Image]` list of PIL images.

        Shapes:
            Latent sizes (when auto-generated):
                - F_lat = (num_frames - 1) // vae_temporal_compression_ratio + 1
                - H_lat = height // vae_spatial_compression_ratio
                - W_lat = width // vae_spatial_compression_ratio

        Notes:
            - Seeding: when `seed` is provided, each temporal window uses a local generator seeded with `seed +
              w_start`, while the shared generator is seeded once for global latents if no generator is passed;
              otherwise the passed-in generator is reused.
            - CFG: unified `noise_pred = uncond + w * (text - uncond)` with optional `guidance_rescale`.
            - Memory: denoising performs full-frame predictions (no spatial tiling); decoding can be tiled to avoid
              OOM.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Input validation: height/width must be divisible by 32
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 1. Device & generator
        device = self._execution_device
        # Normalize generator input: accept list but use the first (batch_size=1)
        if isinstance(generator, list):
            generator = generator[0]
        if seed is not None and generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)

        # 2. Optional i2v first frame conditioning: encode cond_image and inject at frame 0 via prepare_latents
        cond_latents = None
        if cond_image is not None and cond_strength > 0:
            img = self.video_processor.preprocess(cond_image, height=height, width=width)
            img = img.to(device=device, dtype=self.vae.dtype)
            enc = self.vae.encode(img.unsqueeze(2))  # [B, C, 1, h, w]
            cond_latents = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc.latents
            cond_latents = cond_latents.to(torch.float32)
            cond_latents = self._normalize_latents(
                cond_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )

        # 3. Global initial latents [B,C,F,H,W], optionally seeded/conditioned
        latents, negative_index_latents, latent_num_frames, latent_height, latent_width = self.prepare_latents(
            batch_size=1,
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            device=device,
            generator=generator,
            dtype=torch.float32,
            latents=latents,
            cond_latents=cond_latents,
            cond_strength=cond_strength,
            negative_index_latents=negative_index_latents,
        )
        if guidance_latents is not None:
            guidance_latents = guidance_latents.to(device=device, dtype=torch.float32)
            if latents.shape[2] != guidance_latents.shape[2]:
                raise ValueError("The number of frames in `latents` and `guidance_latents` must be the same")

        # 4. Sliding windows in latent frames
        tile_size_lat = max(1, temporal_tile_size // self.vae_temporal_compression_ratio)
        overlap_lat = max(0, temporal_overlap // self.vae_temporal_compression_ratio)
        windows = split_into_temporal_windows(
            latent_num_frames, tile_size_lat, overlap_lat, self.vae_temporal_compression_ratio
        )

        # 5. Multi-prompt segments parsing
        segment_texts = parse_prompt_segments(prompt, prompt_segments)

        out_latents = None
        first_window_latents = None

        # 6. Process each temporal window
        for w_idx, (w_start, w_end) in enumerate(windows):
            if self.interrupt:
                break

            # 6.1 Encode prompt embeddings per window segment
            seg_index = min(w_idx, len(segment_texts) - 1) if segment_texts else 0
            pos_text = segment_texts[seg_index] if segment_texts else (prompt if isinstance(prompt, str) else "")
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=[pos_text],
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                prompt_attention_mask=None,
                negative_prompt_attention_mask=None,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=None,
            )
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

            # 6.2 Window-level timesteps reset: fresh sampling for each temporal window
            if sigmas is not None:
                s = torch.tensor(sigmas, dtype=torch.float32) if not isinstance(sigmas, torch.Tensor) else sigmas
                self.scheduler.set_timesteps(sigmas=s, device=device)
                self._num_timesteps = len(sigmas)
            else:
                self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
                self._num_timesteps = num_inference_steps

            # 6.3 Extract window latents [B,C,T,H,W]
            window_latents = latents[:, :, w_start:w_end]
            window_guidance_latents = guidance_latents[:, :, w_start:w_end] if guidance_latents is not None else None
            window_T = window_latents.shape[2]

            # 6.4 Build per-window cond mask and inject previous tails / reference
            window_cond_mask_5d = torch.ones(
                (1, 1, window_T, latent_height, latent_width), device=device, dtype=torch.float32
            )
            self._current_tile_T = window_T
            prev_overlap_len = 0
            # Inter-window tail latent injection (Extend)
            if w_idx > 0 and overlap_lat > 0 and out_latents is not None:
                k = min(overlap_lat, out_latents.shape[2])
                prev_tail = out_latents[:, :, -k:]
                window_latents, window_cond_mask_5d, prev_overlap_len = inject_prev_tail_latents(
                    window_latents,
                    prev_tail,
                    window_cond_mask_5d,
                    overlap_lat,
                    temporal_overlap_cond_strength,
                    prev_overlap_len,
                )
            # Reference/negative-index latent injection (append 1 frame at window head; controlled by negative_index_strength)
            if window_guidance_latents is not None:
                guiding_len = (
                    window_guidance_latents.shape[2] if w_idx == 0 else window_guidance_latents.shape[2] - overlap_lat
                )
                window_latents, window_cond_mask_5d, prev_overlap_len = inject_prev_tail_latents(
                    window_latents,
                    window_guidance_latents[:, :, -guiding_len:],
                    window_cond_mask_5d,
                    guiding_len,
                    guiding_strength,
                    prev_overlap_len,
                )
            else:
                guiding_len = 0
            window_latents, window_cond_mask_5d, prev_overlap_len = inject_prev_tail_latents(
                window_latents,
                negative_index_latents,
                window_cond_mask_5d,
                1,
                negative_index_strength,
                prev_overlap_len,
            )
            if w_idx == 0 and cond_image is not None and cond_strength > 0:
                # First-frame I2V: smaller mask means stronger preservation of the original latent
                window_cond_mask_5d[:, :, 0] = 1.0 - cond_strength

            # Update effective window latent sizes (consider injections on T/H/W)
            w_B, w_C, w_T_eff, w_H_eff, w_W_eff = window_latents.shape
            p = self.transformer_spatial_patch_size
            pt = self.transformer_temporal_patch_size

            # 6.5 Pack full-window latents/masks once
            # Seeding policy: derive a window-local generator to decouple RNG across windows
            if seed is not None:
                tile_seed = int(seed) + int(w_start)
                local_gen = torch.Generator(device=device).manual_seed(tile_seed)
            else:
                local_gen = generator
            # randn*mask + (1-mask)*latents implements hard-condition initialization
            init_rand = randn_tensor(window_latents.shape, generator=local_gen, device=device, dtype=torch.float32)
            mixed_latents = init_rand * window_cond_mask_5d + (1 - window_cond_mask_5d) * window_latents
            window_latents_packed = self._pack_latents(
                window_latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )
            latents_packed = self._pack_latents(
                mixed_latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )
            cond_mask_tokens = self._pack_latents(
                window_cond_mask_5d, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )
            if self.do_classifier_free_guidance:
                cond_mask = torch.cat([cond_mask_tokens, cond_mask_tokens], dim=0)
            else:
                cond_mask = cond_mask_tokens

            # 6.6 Denoising loop per full window (no spatial tiling)
            sigmas_current = self.scheduler.sigmas.to(device=latents_packed.device)
            if sigmas_current.shape[0] >= 2:
                for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[:-1])):
                    if self.interrupt:
                        break
                    # Skip semantics: if sigma exceeds threshold, skip this step (do not call scheduler.step)
                    sigma_val = float(sigmas_current[i].item())
                    if skip_steps_sigma_threshold is not None and float(skip_steps_sigma_threshold) > 0.0:
                        if sigma_val > float(skip_steps_sigma_threshold):
                            continue

                    self._current_timestep = t

                    # Model input (stack 2 copies under CFG)
                    latent_model_input = (
                        torch.cat([latents_packed] * 2) if self.do_classifier_free_guidance else latents_packed
                    )
                    # Broadcast timesteps, combine with per-token cond mask (I2V at window head)
                    timestep = t.expand(latent_model_input.shape[0])
                    if cond_mask is not None:
                        # Broadcast timestep to per-token mask under CFG: [B] -> [B, S, 1]
                        timestep = timestep[:, None, None] * cond_mask

                    # Micro-conditions: only provide video_coords (num_frames/height/width set to 1)
                    rope_interpolation_scale = (
                        self.vae_temporal_compression_ratio,
                        self.vae_spatial_compression_ratio,
                        self.vae_spatial_compression_ratio,
                    )
                    # Inpainting pre-blend (ComfyUI parity: KSamplerX0Inpaint:400)
                    if cond_mask_tokens is not None:
                        latents_packed = latents_packed * cond_mask_tokens + window_latents_packed * (
                            1.0 - cond_mask_tokens
                        )

                    # Negative-index/overlap lengths (for segmenting time coordinates; RoPE-compatible)
                    k_negative_count = (
                        1 if (negative_index_latents is not None and float(negative_index_strength) > 0.0) else 0
                    )
                    k_overlap_count = overlap_lat if (w_idx > 0 and overlap_lat > 0) else 0
                    video_coords = build_video_coords_for_window(
                        latents=window_latents,
                        overlap_len=int(k_overlap_count),
                        guiding_len=int(guiding_len),
                        negative_len=int(k_negative_count),
                        rope_interpolation_scale=rope_interpolation_scale,
                        frame_rate=frame_rate,
                    )
                    with self.transformer.cache_context("cond_uncond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input.to(dtype=self.transformer.dtype),
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            encoder_attention_mask=prompt_attention_mask,
                            num_frames=1,
                            height=1,
                            width=1,
                            rope_interpolation_scale=rope_interpolation_scale,
                            video_coords=video_coords,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    # Unified CFG
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        if self.guidance_rescale > 0:
                            noise_pred = rescale_noise_cfg(
                                noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                            )

                    # Use global timestep for scheduling, but apply suppressive blending with hard-condition tokens (e.g., first frame) after step to avoid brightness/flicker due to time misalignment
                    latents_packed = self.scheduler.step(
                        noise_pred, t, latents_packed, generator=local_gen, return_dict=False
                    )[0]
                    # Inpainting post-blend (ComfyUI parity: restore hard-conditioned regions after update)
                    if cond_mask_tokens is not None:
                        latents_packed = latents_packed * cond_mask_tokens + window_latents_packed * (
                            1.0 - cond_mask_tokens
                        )
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents_packed = callback_outputs.pop("latents", latents_packed)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    if XLA_AVAILABLE:
                        xm.mark_step()
            else:
                # Not enough sigmas to perform a valid step; skip this window safely.
                pass

            # 6.7 Unpack back to [B,C,T,H,W] once
            window_out = self._unpack_latents(
                latents_packed,
                w_T_eff,
                w_H_eff,
                w_W_eff,
                p,
                pt,
            )
            if prev_overlap_len > 0:
                window_out = window_out[:, :, :-prev_overlap_len]

            # 6.8 Overlap handling and fusion
            if out_latents is None:
                # First window: keep all latent frames and cache as AdaIN reference
                out_latents = window_out
                first_window_latents = out_latents
            else:
                window_out = window_out[:, :, 1:]  # Drop the first frame of the new window
                if adain_factor > 0 and first_window_latents is not None:
                    window_out = adain_normalize_latents(window_out, first_window_latents, adain_factor)
                overlap_len = max(overlap_lat - 1, 1)
                prev_tail_chunk = out_latents[:, :, -window_out.shape[2] :]
                fused = linear_overlap_fuse(prev_tail_chunk, window_out, overlap_len)
                out_latents = torch.cat([out_latents[:, :, : -window_out.shape[2]], fused], dim=2)

        # 7. Decode or return latent
        if output_type == "latent":
            video = out_latents
        else:
            # Decode via tiling to avoid OOM from full-frame decoding; latents are already de-normalized, so keep auto_denormalize disabled
            video = self.vae_decode_tiled(
                out_latents,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                horizontal_tiles=int(decode_horizontal_tiles),
                vertical_tiles=int(decode_vertical_tiles),
                overlap=int(decode_overlap),
                generator=generator,
                output_type=output_type,  # Keep type consistent; postprocess is applied afterwards
            )
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
