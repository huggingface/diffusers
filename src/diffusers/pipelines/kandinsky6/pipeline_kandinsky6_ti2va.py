# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""Kandinsky 6 TI2VA Diffusers pipeline."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from ..pipeline_utils import DiffusionPipeline
from diffusers.utils import replace_example_docstring
from torch import Tensor, nn

from .pipeline_output import Kandinsky6TI2VAPipelineOutput

EXAMPLE_DOC_STRING = """
    Examples:

        ```python
        >>> import torch
        >>> from diffusers import Kandinsky6TI2VAPipeline
        >>> from diffusers.utils import encode_video

        >>> model_id = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
        >>> pipe = Kandinsky6TI2VAPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()

        >>> output = pipe(
        ...     prompt="A cat and a dog baking a cake together in a kitchen.",
        ...     height=480,
        ...     width=864,
        ...     num_frames=121,
        ...     num_inference_steps=16,
        ...     guidance_scale=1.0,
        ...     sample_audio=True,
        ... )

        >>> video = output.frames[0].permute(1, 2, 3, 0)
        >>> audio = torch.as_tensor(output.audio[0])[:, None].repeat(1, 2)
        >>> encode_video(
        ...     video,
        ...     fps=24,
        ...     output_path="output.mp4",
        ...     audio=audio,
        ...     audio_sample_rate=pipe.audio_sample_rate,
        ... )
        ```
"""


@dataclass
class LatentBundle:
    """Packed K6 latent state; audio is optional for a video-only call."""

    video: Tensor | None
    audio: Tensor | None
    video_cu_seqlens: Tensor | None
    audio_cu_seqlens: Tensor | None


class TextEmbeds(TypedDict):
    """Text adapter output consumed by the K6 DiT."""

    text_embeds: Tensor
    pooled_embed: Tensor


_PROMPT_TEMPLATE = "\n".join(
    [
        "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
        "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
        "Describe the location of the video, main characters or objects and their action.",
        "Describe the dynamism of the video and presented actions.",
        "Name the visual style of the video: whether it is a professional footage, user generated content, "
        "some kind of animation, video game or scren content.",
        "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
        "Pay attention to the order of key actions shown in the scene.<|im_end|>",
        "<|im_start|>user\n{}<|im_end|>",
    ]
)
_QWEN_CROP_START = 129
_CLIP_MAX_LENGTH = 77
_T2VA_EXPANSION_INSTRUCTION = (
    "You are a prompt beautifier that transforms short user video+audio descriptions into rich, detailed English "
    "prompts specifically optimized for video+audio generation models. Preserve any direct speech between <S> "
    "and <E> exactly as written. If the prompt asks for speech without exact words, add suitable direct speech "
    "between those tags. Put general audio descriptions between <AUDCAP> and <ENDAUDCAP>. Describe the scene, "
    "actions, camera motion, visual style, and audio in detail. Make the prompt dynamic. Answer only with the "
    "expanded prompt.\n\n"
    "Rewrite Prompt: {}"
)
_I2VA_EXPANSION_INSTRUCTION = (
    "You are a prompt beautifier that transforms a short user video+audio description and a provided reference "
    "image into a rich, detailed English prompt optimized for video+audio generation. The reference image is the "
    "ground truth for the initial scene. Keep every visible fact that you mention consistent with it. Do not mainly "
    "describe the image: focus on the requested actions, motion, interactions, camera changes, speech, and audio, "
    "explaining the changes relative to the initial image. Add only details compatible with the reference image; "
    "never invent conflicting objects, identities, colors, locations, or actions. Preserve direct speech between "
    "<S> and <E> exactly. Put general audio descriptions between <AUDCAP> and <ENDAUDCAP>. Make the prompt dynamic "
    "and describe how the scene evolves from the provided image. Answer only with the expanded prompt.\n\n"
    "Rewrite Prompt: {}"
)


def _prepare_expansion_image(image: Any) -> Any:
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as pil_image:
            return pil_image.convert("RGB").copy()
    return image


def _build_video_input(
    video: Tensor,
    has_visual_cond: bool,
    first_frames: Tensor | None,
    video_cu_seqlens: Tensor | None,
    visual_cond_scheme: str,
) -> Tensor:
    """Append visual-conditioning channels for batched ``(B,T,H,W,C)`` video."""
    if not has_visual_cond:
        return video

    cond = torch.zeros_like(video)
    mask = torch.zeros((*video.shape[:-1], 1), dtype=video.dtype, device=video.device)
    if first_frames is None:
        return torch.cat([video, cond, mask], dim=-1)
    if video_cu_seqlens is None:
        raise ValueError(f"{visual_cond_scheme} requires video_cu_seqlens")

    first_frames = first_frames.to(device=video.device, dtype=video.dtype)
    tail_cond = visual_cond_scheme == "tail_cond_first_frame"
    if visual_cond_scheme == "i2v":
        video[:, 0] = first_frames
    elif tail_cond:
        video[:, -1] = first_frames
    elif visual_cond_scheme == "pretrain":
        cond[:, 0] = first_frames
    else:
        raise ValueError(f"unknown visual_cond_scheme={visual_cond_scheme!r}")
    mask[:, -1 if tail_cond else 0] = 1
    return torch.cat([video, cond, mask], dim=-1)


def _resolve_null_embeds(
    null_text_embeds: TextEmbeds | list[TextEmbeds | None],
    null_text_rope: Tensor | list[Tensor | None],
) -> tuple[Tensor, Tensor, Tensor | list[Tensor]]:
    """Returns (text_embed, pooled_embed, text_rope) for the uncond DiT pass.

    When null_text_embeds is a list of two items (T2VA per-modality nulls),
    returns list arguments so the DiT can apply different null text per modality.
    """
    if isinstance(null_text_embeds, list):
        # null_text_embeds[0] = video null, null_text_embeds[1] = audio null (may be None)
        null_v = null_text_embeds[0]
        null_a = null_text_embeds[1]
        null_rope_v = null_text_rope[0] if isinstance(null_text_rope, list) else null_text_rope
        null_rope_a = null_text_rope[1] if isinstance(null_text_rope, list) else None

        if null_a is not None and null_rope_a is not None:
            # Different null text per modality — pass as list to DiT
            text_embed = [null_v["text_embeds"], null_a["text_embeds"]]
            pooled_embed = [null_v["pooled_embed"], null_a["pooled_embed"]]
            rope = [null_rope_v, null_rope_a]
        else:
            text_embed = null_v["text_embeds"]
            pooled_embed = null_v["pooled_embed"]
            rope = null_rope_v
    else:
        text_embed = null_text_embeds["text_embeds"]
        pooled_embed = null_text_embeds["pooled_embed"]
        rope = null_text_rope

    return text_embed, pooled_embed, rope


def _raw_dit(dit: nn.Module) -> nn.Module:
    inner = getattr(dit, "module", None)
    return inner if inner is not None and hasattr(dit, "set_cache") else dit


def _rebuild_text_rope(dit: nn.Module, template: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
    """Rebuild text RoPE using the standalone Diffusers transformer names."""
    raw = getattr(dit, "module", None)
    raw = raw if raw is not None and hasattr(raw, "set_cache") else dit
    if isinstance(template, list):
        if getattr(raw, "is_multimodal", False):
            modules = [raw.video_text_rope_embeddings, raw.audio_text_rope_embeddings]
        else:
            modules = [raw.text_rope_embeddings] * len(template)
        return [compute_rope1d(module, int(item.shape[0])) for module, item in zip(modules, template, strict=True)]
    module = raw.video_text_rope_embeddings if getattr(raw, "is_multimodal", False) else raw.text_rope_embeddings
    return compute_rope1d(module, int(template.shape[0]))


def apply_cfg(cond: Tensor, uncond: Tensor, guidance_weight: float) -> Tensor:
    """Classifier-free guidance: uncond + w * (cond - uncond)."""
    return uncond + guidance_weight * (cond - uncond)


@torch.no_grad()
def denoise_loop(  # noqa: PLR0912, PLR0913, PLR0915
    bundle: LatentBundle,
    dit: nn.Module,
    text_embeds: TextEmbeds,
    null_text_embeds: TextEmbeds | list[TextEmbeds | None],
    visual_rope: Tensor | None,
    audio_rope: Tensor | None,
    text_rope: Tensor | list[Tensor],
    null_text_rope: Tensor | list[Tensor | None],
    num_steps: int,
    guidance_weight: float,
    scheduler: Any,
    first_frames: Tensor | None = None,
    visual_cond_scheme: str = "pretrain",
    sparse_params: dict | None = None,
    sample_video: bool = True,
    sample_audio: bool = True,
    *,
    attention_mask: Tensor | None = None,
    null_attention_mask: Tensor | None = None,
    visual_token_type_ids: Tensor | None = None,
    recompute_ropes_each_step: bool = False,
    scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    step_callback: Callable[[int, Tensor, LatentBundle], LatentBundle | Tensor | None] | None = None,
    progress_bar: Any | None = None,
) -> LatentBundle:
    """Run the native Euler loop and invoke a Diffusers step callback.

    ``step_callback`` is called after each Euler update with the zero-based
    step index, the current scheduler timestep, and the mutable latent
    bundle.  It may return ``None``, a replacement ``LatentBundle``, or a
    replacement video ``Tensor``.  The callback is deliberately optional so
    the generated pipeline has the same numerical behavior when it is not
    supplied.
    """
    video = bundle.video  # (B, T, H, W, C) or (B, T, H, W, C+extra) for instruct
    audio = bundle.audio  # (B, A, audio_dim) or None
    is_multimodal = video is not None and audio is not None
    is_piflow = bool(getattr(scheduler, "is_piflow", False))
    guidance_epsilon = 1e-6
    if is_piflow and abs(guidance_weight - 1.0) > guidance_epsilon:
        raise ValueError("PiflowScheduler requires guidance_weight=1.0")

    audio_scheduler = deepcopy(scheduler) if is_multimodal else scheduler
    device = video.device if video is not None else audio.device
    scheduler.set_timesteps(num_steps, device=device)
    if audio_scheduler is not scheduler:
        audio_scheduler.set_timesteps(num_steps, device=device)
    step_timesteps = scheduler.timesteps
    if step_timesteps.shape[0] != num_steps:
        raise ValueError(
            "Diffusers scheduler returned an unexpected number of timesteps: "
            f"expected {num_steps}, got {step_timesteps.shape[0]}"
        )

    bs = (
        bundle.video_cu_seqlens.shape[0] - 1
        if bundle.video_cu_seqlens is not None
        else bundle.audio_cu_seqlens.shape[0] - 1
    )

    null_te, null_pe, null_rope = _resolve_null_embeds(null_text_embeds, null_text_rope)

    out_c: int | None = None  # determined after first step, used to strip instruct channels
    raw = _raw_dit(dit)
    vis_shape = (
        (int(visual_rope.shape[0]), int(visual_rope.shape[1]), int(visual_rope.shape[2]))
        if visual_rope is not None
        else None
    )
    scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
    tail_cond = visual_cond_scheme == "tail_cond_first_frame"

    def _cache_scope(name: str):
        cache_context = getattr(dit, "cache_context", None)
        return cache_context(name) if callable(cache_context) else nullcontext()

    def _guided(cond: Tensor, uncond: Tensor) -> Tensor:
        return cond if is_piflow else apply_cfg(cond, uncond, guidance_weight)

    for step_index, t in enumerate(step_timesteps):
        # Diffusers schedulers expose the model-scale timestep directly.
        t_step: Tensor | list[Tensor] = t.unsqueeze(0).expand(bs)  # (bs,)

        model_input_v = (
            _build_video_input(
                video,
                dit.visual_cond,
                first_frames,
                bundle.video_cu_seqlens,
                visual_cond_scheme,
            )
            if video is not None
            else None
        )

        # Freeze one modality at t=0 for partial sampling (T2VA only)
        if is_multimodal:
            t_frozen = torch.zeros_like(t_step[0]).expand(bs)
            if not sample_audio:
                t_step = [t_step, t_frozen]
            elif not sample_video:
                t_step = [t_frozen, t_step]

        def _forward(
            te,
            pe,
            rope,
            attn_mask,
            *,
            _model_input_v=model_input_v,
            _audio=audio,
            _t_step=t_step,
        ):
            vr, ar, tr = visual_rope, audio_rope, rope
            if recompute_ropes_each_step:
                if vis_shape is not None:
                    vr = compute_visual_rope(raw.visual_rope_embeddings, vis_shape, scale)
                if audio_rope is not None:
                    ar = compute_rope1d(raw.audio_rope_embeddings, int(audio_rope.shape[0]))
                tr = _rebuild_text_rope(raw, rope)
            return dit(
                x_video=_model_input_v,
                x_audio=_audio,
                text_embed=te,
                pooled_text_embed=pe,
                time=_t_step,
                visual_rope=vr,
                audio_rope=ar,
                text_rope=tr,
                sparse_params=sparse_params,
                attention_mask=attn_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        with _cache_scope("cond"):
            vel_cond = _forward(
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                text_rope,
                attention_mask,
            )
        if abs(guidance_weight - 1.0) > guidance_epsilon:
            with _cache_scope("uncond"):
                vel_uncond = _forward(null_te, null_pe, null_rope, null_attention_mask)
        else:
            vel_uncond = vel_cond

        # Diffusers schedulers own the update for each modality. Multimodal
        # runs use an independent scheduler state for audio because the
        # standard scheduler advances its step index on every call.
        if isinstance(vel_cond, tuple):
            vel_v, vel_a = vel_cond
            uvel_v, uvel_a = vel_uncond if vel_uncond is not vel_cond else (vel_v, vel_a)
            if out_c is None:
                out_c = video.shape[-1] if video is not None else vel_v.shape[-1]

            if video is not None and sample_video:
                guided_v = _guided(vel_v, uvel_v)
                video = scheduler.step(guided_v, t, video, return_dict=False)[0]
                if tail_cond and first_frames is not None:
                    video[:, -1] = first_frames.to(device=device, dtype=video.dtype)

            if audio is not None and sample_audio:
                guided_a = _guided(vel_a, uvel_a)
                audio = audio_scheduler.step(guided_a, t, audio, return_dict=False)[0]
        else:
            if out_c is None:
                out_c = video.shape[-1] if video is not None else vel_cond.shape[-1]
            vel = _guided(vel_cond, vel_uncond)
            if video is not None and sample_video:
                video = scheduler.step(vel, t, video, return_dict=False)[0]
                if tail_cond and first_frames is not None:
                    video[:, -1] = first_frames.to(device=device, dtype=video.dtype)
            elif audio is not None and sample_audio:
                audio = scheduler.step(vel, t, audio, return_dict=False)[0]

        if step_callback is not None:
            callback_result = step_callback(
                step_index,
                t,
                LatentBundle(
                    video=video,
                    audio=audio,
                    video_cu_seqlens=bundle.video_cu_seqlens,
                    audio_cu_seqlens=bundle.audio_cu_seqlens,
                ),
            )
            if isinstance(callback_result, LatentBundle):
                video = callback_result.video
                audio = callback_result.audio
            elif isinstance(callback_result, Tensor):
                video = callback_result

        if progress_bar is not None:
            progress_bar.update()

    # I2V / I2VA: keep injected reference frames unchanged
    if first_frames is not None and video is not None and bundle.video_cu_seqlens is not None:
        ff = first_frames.to(device=device, dtype=video.dtype)
        if visual_cond_scheme == "i2v":
            video[:, 0] = ff
        elif tail_cond:
            video[:, -1] = ff

    # Strip instruct extra-channel padding from the video latent
    if out_c is not None and video is not None:
        video = video[..., :out_c]

    return LatentBundle(
        video=video,
        audio=audio,
        video_cu_seqlens=bundle.video_cu_seqlens,
        audio_cu_seqlens=bundle.audio_cu_seqlens,
    )


def prepare_video_latents(
    bs: int,
    duration: int,
    H_lat: int,
    W_lat: int,
    C: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> LatentBundle:
    """Sample random video noise latent and compute cu_seqlens.

    Returns a LatentBundle with video=(bs*duration, H_lat, W_lat, C) and
    uniform video_cu_seqlens=(bs+1,) int32.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    video = torch.randn(bs * duration, H_lat, W_lat, C, device=device, dtype=dtype, generator=g)
    cu = duration * torch.arange(bs + 1, dtype=torch.int32, device=device)
    return LatentBundle(video=video, audio=None, video_cu_seqlens=cu, audio_cu_seqlens=None)


def audio_latent_duration(
    video_latent_frames: int,
    *,
    fps: float = 24.0,
    audio_fps: int = 44100,
    downsample_factor: int = 1024,
) -> int:
    """Audio latent length matching K5 T2VA: ceil(sample_frames/fps * audio_fps / downsample)."""
    sample_frames = (video_latent_frames - 1) * 4 + 1
    return int(math.ceil(sample_frames / fps * audio_fps / downsample_factor))


def prepare_audio_latents(
    bundle: LatentBundle,
    audio_duration: int,
    audio_dim: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> LatentBundle:
    """Attach random audio noise latent to an existing LatentBundle.

    audio_duration: number of latent audio frames per batch item.
    """
    bs = bundle.video_cu_seqlens.shape[0] - 1 if bundle.video_cu_seqlens is not None else 1
    g = torch.Generator(device=device)
    g.manual_seed(seed + 1)  # offset from video seed for independence
    audio = torch.randn(bs * audio_duration, audio_dim, device=device, dtype=dtype, generator=g)
    cu = audio_duration * torch.arange(bs + 1, dtype=torch.int32, device=device)
    return LatentBundle(
        video=bundle.video,
        audio=audio,
        video_cu_seqlens=bundle.video_cu_seqlens,
        audio_cu_seqlens=cu,
    )


def compute_rope1d(rope: nn.Module, length: int, device: torch.device | None = None) -> Tensor:
    """Build 1-D RoPE and move non-persistent tables for Diffusers offload."""
    if length < 1:
        raise ValueError(f"rope length must be positive, got {length}")
    if device is None:
        device = next(rope.buffers()).device
    else:
        device = torch.device(device)
        # Accelerate's model CPU-offload hook only moves the transformer when
        # its forward starts. RoPE is materialized before that forward, so
        # explicitly move its non-persistent lookup table first.
        rope.to(device)
    pos = torch.arange(length, device=device)
    return rope(pos)


def compute_visual_rope(
    rope: nn.Module,
    shape: tuple[int, int, int],
    scale_factor: tuple[float, float, float],
    device: torch.device | None = None,
) -> Tensor:
    """Build 3-D RoPE and move non-persistent tables for Diffusers offload."""
    T, H, W = (int(shape[0]), int(shape[1]), int(shape[2]))
    if T < 1 or H < 1 or W < 1:
        raise ValueError(f"visual rope shape must be positive, got {shape}")
    if device is None:
        device = next(rope.buffers()).device
    else:
        device = torch.device(device)
        rope.to(device)
    pos = [
        torch.arange(T, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
    ]
    scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
    return rope((T, H, W), pos, scale)


@torch.no_grad()
def postprocess_audio(
    bundle: LatentBundle,
    audio_vae,
    normalization_mode: str = "clip",
) -> list[np.ndarray] | None:
    """Decode audio latents → list of (samples,) int16 numpy arrays.

    Returns None when bundle.audio is None (T2V mode).

    ``clip`` preserves the decoded amplitude and saturates it to [-1, 1].
    ``normalize`` peak-normalizes each waveform before converting it to int16.
    """
    audio = bundle.audio
    if audio is None:
        return None

    cu = bundle.audio_cu_seqlens
    assert cu is not None
    bs = cu.shape[0] - 1

    # Reverse audio VAE normalization
    audio_scaled = audio / getattr(audio_vae, "scaling_factor", 1.0)
    audio_scaled = audio_scaled + getattr(audio_vae, "mean_value", 0.0)

    segments = [
        audio_scaled[i] if audio_scaled.ndim == 3 else audio_scaled[cu[i].item() : cu[i + 1].item()] for i in range(bs)
    ]  # (A, audio_dim) per sample

    # Native generation keeps equal-length samples in the batch. Decode them
    # together so batching is preserved through the audio VAE and vocoder.
    if len({tuple(segment.shape) for segment in segments}) == 1:
        decoded = audio_vae.wrapped_decode(torch.stack(segments).transpose(1, 2))
        if decoded.ndim == 1:
            decoded = decoded.unsqueeze(0)
        elif decoded.ndim == 3 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        waveforms = [waveform.cpu().float().numpy() for waveform in decoded]
    else:
        # Preserve support for legacy packed bundles with variable lengths.
        waveforms = [
            audio_vae.wrapped_decode(segment.transpose(1, 0).unsqueeze(0)).squeeze().cpu().float().numpy()
            for segment in segments
        ]

    result: list[np.ndarray] = []
    for waveform in waveforms:
        if normalization_mode == "normalize":
            peak = np.max(np.abs(waveform))
            if peak > 0:
                waveform = waveform / peak * 32767
        elif normalization_mode == "clip":
            waveform = np.clip(waveform, -1.0, 1.0) * 32767
        else:
            raise ValueError(f"unknown audio normalization_mode={normalization_mode}")

        result.append(waveform.astype(np.int16))

    return result


@torch.no_grad()
def postprocess_video(
    bundle: LatentBundle,
    vae,
    bs: int,
) -> Tensor:
    """Decode video latents → (bs, 3, T, H, W) uint8 in [0, 255].

    Input layout: packed ``(sum_T, H_lat, W_lat, C)`` or batched
    ``(bs, T, H_lat, W_lat, C)``.
    """
    video = bundle.video
    assert video is not None

    if video.ndim == 4:
        frames = video.reshape(bs, -1, video.shape[-3], video.shape[-2], video.shape[-1])
    elif video.ndim == 5:
        if video.shape[0] != bs:
            raise ValueError(f"batched video has batch size {video.shape[0]}, expected {bs}")
        frames = video
    else:
        raise ValueError(f"video must have rank 4 or 5, got {video.ndim}")
    # (bs, T, H, W, C) → (bs, C, T, H, W) for VAE input
    frames = (frames / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
    # Hunyuan VAE loads as fp16; DiT latents are bf16 — match weight dtype.
    vae_dtype = next(vae.parameters()).dtype
    frames = vae.decode(frames.to(dtype=vae_dtype)).sample

    return ((frames.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)


@torch.no_grad()
def resize_image(
    image: Tensor,
    max_area: int,
    divisibility: int = 16,
    world_size: int = 1,
) -> tuple[Tensor, float]:
    """Aspect-preserving resize to fit ``max_area`` with sides divisible by ``divisibility``.

    K5 ``i2v_pipeline.resize_image`` parity. ``image`` is ``(B, C, H, W)``.
    Returns ``(resized, scale_k)``.
    """
    from math import sqrt

    h, w = image.shape[2:]
    area = h * w
    div = divisibility
    if div == 16:
        if world_size in (2, 4):
            div *= 2
        elif world_size == 8:
            div *= 4
    k = sqrt(max_area / area) / div
    new_h = int(round(h * k) * div)
    new_w = int(round(w * k) * div)
    try:
        import torchvision.transforms.functional as TF
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "I2VA image processing requires torchvision. Install it with `pip install torchvision`."
        ) from exc

    return TF.resize(image, (new_h, new_w)), k


def _load_pil_rgb(image: str | object):
    from PIL import Image

    if isinstance(image, str):
        try:
            pil_image = Image.open(image).convert("RGB")
            pil_image.load()
        except Exception as exc:
            raise ValueError(f"Cannot decode i2va input image {image!r}: {exc}") from exc
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"i2va image must be a path or PIL image, got {type(image).__name__}")
    return pil_image


@torch.no_grad()
def encode_i2va_first_frame(
    image: str | object,
    vae,
    device: torch.device | str,
    height: int | None = None,
    width: int | None = None,
    *,
    max_area: int | None = None,
    divisibility: int = 16,
    world_size: int = 1,
) -> tuple[Tensor, int, int]:
    """Encode one image into the K6 packed first-frame latent layout."""
    try:
        import torchvision.transforms.functional as TF
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "I2VA image processing requires torchvision. Install it with `pip install torchvision`."
        ) from exc

    pil_image = _load_pil_rgb(image)
    tensor = TF.pil_to_tensor(pil_image).unsqueeze(0)
    if max_area is not None and (height is None or width is None):
        tensor, _ = resize_image(tensor, max_area, divisibility, world_size)
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    elif height is not None and width is not None:
        src_h, src_w = tensor.shape[-2:]
        scale = min(src_h / height, src_w / width)
        tensor = TF.resize(tensor, (int(src_h / scale), int(src_w / scale)))
        cur_h, cur_w = tensor.shape[-2:]
        tensor = TF.crop(tensor, (cur_h - height) // 2, (cur_w - width) // 2, height, width)
    else:
        raise ValueError("encode_i2va_first_frame needs height/width or max_area")

    tensor = tensor.to(device=device, dtype=next(vae.parameters()).dtype) / 127.5 - 1.0
    tensor = tensor.transpose(0, 1).unsqueeze(0)
    encoded = vae.encode(tensor)
    posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
    latent = posterior.sample()
    latent = latent.squeeze(0).permute(1, 2, 3, 0)
    return latent * vae.config.scaling_factor, int(height), int(width)


def append_i2va_tail_condition(
    latent_visual: Tensor,
    first_frames: Tensor,
    batch_size: int,
    video_duration: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append one clean reference frame to batched ``(B,T,H,W,C)`` latents."""
    if latent_visual.shape[0] != batch_size or latent_visual.shape[1] != video_duration:
        raise ValueError(
            "generated visual latent shape mismatch: expected "
            f"({batch_size}, {video_duration}, H, W, C), got {tuple(latent_visual.shape)}"
        )

    _, _, height, width, dim = latent_visual.shape
    first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
    expected = (batch_size, height, width, dim)
    if tuple(first_frames.shape) != expected:
        raise ValueError(f"first-frame latent shape mismatch: expected {expected}, got {tuple(first_frames.shape)}")

    latent_visual = torch.cat([latent_visual, first_frames[:, None]], dim=1)
    token_types = torch.cat(
        [
            torch.zeros((batch_size, video_duration), dtype=torch.long, device=latent_visual.device),
            torch.ones((batch_size, 1), dtype=torch.long, device=latent_visual.device),
        ],
        dim=1,
    )
    return latent_visual, token_types, token_types == 0


class Kandinsky6TI2VAPipeline(DiffusionPipeline):
    r"""Pipeline for text/image-to-video-and-audio generation with Kandinsky 6.

    This pipeline inherits the generic loading, device placement, and progress
    handling provided by ``DiffusionPipeline``. The transformer must expose
    the K6 multimodal forward contract, while the VAE adapters must implement
    the portable video/audio post-processing contracts.

    Args:
        transformer: Multimodal K6 transformer used to denoise video and audio
            latents.
        vae: Video VAE used to decode generated video latents.
        text_encoder: Qwen2.5-VL text encoder for token-level embeddings.
        audio_vae: Audio VAE used to decode generated audio latents. It may be
            ``None`` only when ``sample_audio=False``.
        scheduler: Diffusion scheduler used by the denoising loop.
        tokenizer: Qwen2.5-VL processor.
        text_encoder_2: CLIP text encoder for pooled embeddings.
        tokenizer_2: CLIP tokenizer.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae->audio_vae"
    _callback_tensor_inputs = ["latents"]
    _DEFAULT_NEGATIVE_PROMPT = (
        "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
        "worst quality, low quality, ugly, deformed, walking backwards"
    )

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: Any,
        audio_vae: nn.Module | None,
        scheduler: Any,
        tokenizer: Any,
        text_encoder_2: Any,
        tokenizer_2: Any,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            audio_vae=audio_vae,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )
        transformer_config = self.transformer.config
        scale_factor = transformer_config.get("scale_factor", (1.0, 2.0, 2.0))
        text_token_padding = transformer_config.get("text_token_padding", True)
        self.scale_factor = tuple(float(value) for value in scale_factor)
        self.text_token_padding = bool(text_token_padding)

        audio_config = self.audio_vae.config if self.audio_vae is not None else {}
        self.audio_sample_rate = int(audio_config.get("sample_rate", 44_100))
        self.audio_downsample_factor = int(audio_config.get("downsample_factor", 1_024))

    @staticmethod
    def _value_batch_size(value: Any) -> int:
        if value is None:
            return 1
        if isinstance(value, list):
            return len(value)
        if isinstance(value, Tensor):
            return int(value.shape[0]) if value.ndim in (3, 5) else 1
        if isinstance(value, dict) and isinstance(value.get("text_embeds"), Tensor):
            embeds = value["text_embeds"]
            return int(embeds.shape[0]) if embeds.ndim == 3 else 1
        return 1

    @classmethod
    def _batch_size(
        cls,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        image: str | object | list[str | object] | None,
        latents: Tensor | None,
        audio_latents: Tensor | None,
        prompt_embeds: TextEmbeds | None,
        negative_prompt_embeds: TextEmbeds | None,
    ) -> int:
        sizes = [
            cls._value_batch_size(value)
            for value in (
                prompt,
                negative_prompt,
                image,
                latents,
                audio_latents,
                prompt_embeds,
                negative_prompt_embeds,
            )
        ]
        batch_size = max(sizes)
        if batch_size < 1 or any(size not in (1, batch_size) for size in sizes):
            raise ValueError(f"all batched inputs must have the same batch size, got {sizes}")
        return batch_size

    @staticmethod
    def _as_batch(value: Any, batch_size: int, name: str) -> list[Any] | None:
        if value is None:
            return None
        values = value if isinstance(value, list) else [value]
        if len(values) == 1:
            return values * batch_size
        if len(values) != batch_size:
            raise ValueError(f"{name} must contain one item or {batch_size} items, got {len(values)}")
        return values

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        max_sequence_length: int,
        output_type: str,
        prompt_embeds: TextEmbeds | None = None,
        negative_prompt_embeds: TextEmbeds | None = None,
        prompt_cu_seqlens: Tensor | None = None,
        negative_prompt_cu_seqlens: Tensor | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        sample_audio: bool = True,
        image: str | object | None = None,
        visual_cond_scheme: str = "pretrain",
    ) -> None:
        """Validate arguments shared by text-to-video and image-to-video calls.

        Args:
            prompt: Text prompt or a batch of prompts.
            image: Optional reference image or batch of reference images.
            negative_prompt: Negative text prompt or a batch of prompts.
            height: Requested output height in pixels.
            width: Requested output width in pixels.
            num_frames: Requested output frame count.
            num_inference_steps: Number of denoising steps.
            max_sequence_length: Maximum Qwen prompt length after the template.
            output_type: One of ``pt``, ``torch``, ``np``, ``numpy``, or
                ``latent``.
            prompt_embeds: Optional precomputed positive prompt embeddings.
            negative_prompt_embeds: Optional precomputed negative embeddings.
            prompt_cu_seqlens: Cumulative positive Qwen sequence lengths.
            negative_prompt_cu_seqlens: Cumulative negative Qwen sequence
                lengths.
            callback_on_step_end_tensor_inputs: Tensor names exposed to the
                step callback.
            sample_audio: Whether to generate and decode audio.
            visual_cond_scheme: Image-conditioning scheme when ``image`` is set.

        Raises:
            ValueError: If an input combination is unsupported.
        """
        for name, value in (("prompt", prompt), ("negative_prompt", negative_prompt)):
            if value is not None and not isinstance(value, (str, list)):
                raise ValueError(f"`{name}` has to be a `str` or a `list`, but is {type(value)}")
            if isinstance(value, list) and (not value or not all(isinstance(item, str) for item in value)):
                raise ValueError(f"`{name}` must be a non-empty list of strings")

        if prompt is None and prompt_embeds is None:
            raise ValueError("provide either `prompt` or `prompt_embeds`")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("provide either `prompt` or `prompt_embeds`, not both")
        if (prompt_embeds is None) != (prompt_cu_seqlens is None):
            raise ValueError("`prompt_embeds` and `prompt_cu_seqlens` must be provided together")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("provide either `negative_prompt` or `negative_prompt_embeds`, not both")
        if (negative_prompt_embeds is None) != (negative_prompt_cu_seqlens is None):
            raise ValueError("`negative_prompt_embeds` and `negative_prompt_cu_seqlens` must be provided together")
        if prompt_embeds is not None:
            self._as_text_embeds(prompt_embeds)
        if negative_prompt_embeds is not None:
            self._as_text_embeds(negative_prompt_embeds)

        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if height < 1 or width < 1:
            raise ValueError("height and width must be positive")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("height and width must be divisible by 8")
        if output_type not in ("pt", "torch", "np", "numpy", "latent"):
            raise ValueError("output_type must be 'pt', 'torch', 'np', 'numpy', or 'latent'")
        if max_sequence_length < 1 or max_sequence_length > 1024:
            raise ValueError("max_sequence_length must be between 1 and 1024")
        if sample_audio and self.audio_vae is None:
            raise ValueError("sample_audio=True requires an audio_vae")

        if getattr(self.transformer, "is_multimodal", False) is not True:
            raise ValueError("Kandinsky6TI2VAPipeline requires a multimodal transformer")
        patch_size = tuple(int(value) for value in getattr(self.transformer, "patch_size", (1, 2, 2)))
        if len(patch_size) < 3 or any(value <= 0 for value in patch_size):
            raise ValueError(f"transformer.patch_size must contain three positive values, got {patch_size}")
        if (height // 8) % patch_size[1] != 0 or (width // 8) % patch_size[2] != 0:
            raise ValueError(
                "height and width must produce whole DiT spatial patches; "
                f"got latent size {(height // 8, width // 8)} and patch size {patch_size}"
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            name in self._callback_tensor_inputs for name in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {callback_on_step_end_tensor_inputs}"
            )
        if visual_cond_scheme not in ("pretrain", "i2v", "tail_cond_first_frame"):
            raise ValueError(f"unknown visual_cond_scheme={visual_cond_scheme!r}")
        if image is not None:
            if not getattr(self.transformer, "visual_cond", False):
                raise ValueError("image conditioning requires transformer.visual_cond=True")
            if (
                visual_cond_scheme == "tail_cond_first_frame"
                and getattr(self.transformer, "visual_token_type_num_embeddings", 0) < 2
            ):
                raise ValueError("tail_cond_first_frame requires transformer.visual_token_type_num_embeddings >= 2")

    @staticmethod
    def _as_text_embeds(value: Any) -> TextEmbeds:
        if not isinstance(value, dict):
            raise TypeError("prompt embeddings must be a mapping with text_embeds and pooled_embed")
        missing = {"text_embeds", "pooled_embed"} - set(value)
        if missing:
            raise ValueError(f"prompt embeddings are missing: {sorted(missing)}")
        return value

    def encode_prompt(
        self, text: str | list[str], max_sequence_length: int
    ) -> tuple[TextEmbeds, Tensor, Tensor | None]:
        """Encode text with Qwen2.5-VL and CLIP and build packed token metadata.

        Args:
            text (`str` or `list[str]`): Prompt or prompts to encode.
            max_sequence_length (`int`): Maximum number of Qwen text tokens.

        Returns:
            `tuple`: Text embeddings, cumulative sequence lengths, and an
            optional Qwen attention mask.
        """
        texts = [text] if isinstance(text, str) else text
        full_texts = [_PROMPT_TEMPLATE.format(item) for item in texts]
        qwen_device = next(self.text_encoder.parameters()).device
        inputs = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_sequence_length + _QWEN_CROP_START,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(qwen_device)
        qwen_output = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )
        embeds = qwen_output["hidden_states"][-1][:, _QWEN_CROP_START:]
        attention = inputs["attention_mask"][:, _QWEN_CROP_START:].to(dtype=torch.bool)
        batch_size = len(texts)
        if self.text_token_padding:
            qwen_embeds = embeds if batch_size > 1 else embeds[0]
            qwen_attention = attention if batch_size > 1 else attention[0]
            lengths = torch.full((batch_size,), embeds.shape[1], dtype=torch.int32, device=embeds.device)
            cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=embeds.device), lengths.cumsum(0)])
        elif batch_size > 1:
            max_tokens = int(attention.sum(dim=1).max().item())
            qwen_embeds = embeds[:, :max_tokens]
            qwen_attention = attention[:, :max_tokens]
            lengths = torch.full((batch_size,), max_tokens, dtype=torch.int32, device=embeds.device)
            cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=embeds.device), lengths.cumsum(0)])
        else:
            qwen_embeds = embeds[attention]
            qwen_attention = None
            lengths = attention.sum(dim=1).to(torch.int32)
            cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=embeds.device), lengths.cumsum(0)])

        clip_device = next(self.text_encoder_2.parameters()).device
        clip_inputs = self.tokenizer_2(
            texts,
            max_length=_CLIP_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(clip_device)
        pooled_embed = self.text_encoder_2(**clip_inputs)["pooler_output"]
        return (
            {
                "text_embeds": qwen_embeds,
                "pooled_embed": pooled_embed,
            },
            cu_seqlens,
            qwen_attention,
        )

    @torch.no_grad()
    def expand_prompts(
        self,
        prompt: str | list[str],
        image: Any | list[Any] | None = None,
        max_sequence_length: int = 1024,
        mode: str | None = None,
    ) -> str | list[str]:
        """Expand T2VA prompts, or image-grounded I2VA prompts when an image is supplied."""
        if isinstance(prompt, list):
            images = image if isinstance(image, (list, tuple)) else [image] * len(prompt)
            return [
                self.expand_prompts(
                    item,
                    image=item_image,
                    max_sequence_length=max_sequence_length,
                    mode=mode,
                )
                for item, item_image in zip(prompt, images, strict=True)
            ]
        mode = mode or ("i2va" if image is not None else "t2va")
        if mode not in ("t2va", "i2va"):
            raise ValueError(f"unknown prompt expansion mode={mode!r}")
        image = _prepare_expansion_image(image) if image is not None else None
        instruction = (_I2VA_EXPANSION_INSTRUCTION if mode == "i2va" else _T2VA_EXPANSION_INSTRUCTION).format(prompt)
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": instruction})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = next(self.text_encoder.parameters()).device
        inputs = self.tokenizer(
            text=[text],
            images=[image] if image is not None else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated = self.text_encoder.generate(**inputs, max_new_tokens=max_sequence_length)
        qwen_crop_start = inputs["input_ids"].shape[1]
        trimmed = [output[qwen_crop_start:] for output in generated]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str],
        *,
        prompt_embeds: TextEmbeds | None,
        negative_prompt_embeds: TextEmbeds | None,
        prompt_cu_seqlens: Tensor | None,
        negative_prompt_cu_seqlens: Tensor | None,
        max_sequence_length: int,
    ) -> tuple[TextEmbeds, Tensor, Tensor | None, TextEmbeds, Tensor, Tensor | None]:

        if prompt_embeds is None:
            prompt_embeds, prompt_cu_seqlens, prompt_attention_mask = self.encode_prompt(prompt, max_sequence_length)
        elif prompt_cu_seqlens is None:
            raise ValueError("prompt_cu_seqlens is required with prompt_embeds")
        else:
            prompt_attention_mask = None
        prompt_embeds = self._as_text_embeds(prompt_embeds)

        if negative_prompt_embeds is None:
            negative_prompt_embeds, negative_prompt_cu_seqlens, negative_attention_mask = self.encode_prompt(
                negative_prompt, max_sequence_length
            )
        elif negative_prompt_cu_seqlens is None:
            raise ValueError("negative_prompt_cu_seqlens is required with negative_prompt_embeds")
        else:
            negative_attention_mask = None
        negative_prompt_embeds = self._as_text_embeds(negative_prompt_embeds)
        return (
            prompt_embeds,
            prompt_cu_seqlens,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_cu_seqlens,
            negative_attention_mask,
        )

    @staticmethod
    def _seed_from_generator(
        generator: torch.Generator | None,
        device: torch.device,
    ) -> int:
        if generator is None:
            return int(torch.randint(0, 2**31, (1,), device=device).item())
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        return int(
            torch.randint(
                0,
                2**31,
                (1,),
                generator=generator,
                device=generator_device,
            ).item()
        )

    @staticmethod
    def _coerce_video_latents(
        latents: Tensor,
        *,
        latent_frames: int,
        height: int,
        width: int,
        channels: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentBundle:
        value = latents.to(device=device, dtype=dtype)
        if value.ndim == 4:
            value = value.unsqueeze(0)
        if value.ndim != 5:
            raise ValueError("video latents must have shape (T,H,W,C), (B,T,H,W,C), or (B,C,T,H,W)")
        if value.shape[0] != batch_size:
            raise ValueError(f"latents batch size must be {batch_size}, got {value.shape[0]}")
        expected = (batch_size, latent_frames, height // 8, width // 8, channels)
        bchtw = (batch_size, channels, latent_frames, height // 8, width // 8)
        if tuple(value.shape) == bchtw:
            value = value.permute(0, 2, 3, 4, 1)
        elif tuple(value.shape) != expected:
            raise ValueError("5-D latents must use BCHTW or BTHWC layout with the configured channel count")
        cu_seqlens = latent_frames * torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        return LatentBundle(
            video=value,
            audio=None,
            video_cu_seqlens=cu_seqlens,
            audio_cu_seqlens=None,
        )

    @staticmethod
    def _coerce_audio_latents(
        bundle: LatentBundle,
        latents: Tensor,
        *,
        audio_duration: int,
        audio_dim: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentBundle:
        value = latents.to(device=device, dtype=dtype)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError("audio latents must have shape (A,D) or (B,A,D)")
        if value.shape[0] != batch_size:
            raise ValueError(f"audio_latents batch size must be {batch_size}, got {value.shape[0]}")
        expected = (batch_size, audio_duration, audio_dim)
        if tuple(value.shape) != expected:
            raise ValueError(f"audio latent shape mismatch: expected {expected}, got {tuple(value.shape)}")
        cu_seqlens = audio_duration * torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        return LatentBundle(
            video=bundle.video,
            audio=value,
            video_cu_seqlens=bundle.video_cu_seqlens,
            audio_cu_seqlens=cu_seqlens,
        )

    def _prepare_latents(
        self,
        *,
        latents: Tensor | None,
        audio_latents: Tensor | None,
        latent_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        sample_fps: float,
        sample_audio: bool,
        batch_size: int,
    ) -> LatentBundle:
        channels = int(getattr(self.transformer, "in_visual_dim", 16))
        seed = self._seed_from_generator(generator, device)
        if latents is None:
            bundle = prepare_video_latents(
                bs=batch_size,
                duration=latent_frames,
                H_lat=height // 8,
                W_lat=width // 8,
                C=channels,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            bundle.video = bundle.video.reshape(batch_size, latent_frames, height // 8, width // 8, channels)
        else:
            bundle = self._coerce_video_latents(
                latents,
                latent_frames=latent_frames,
                height=height,
                width=width,
                channels=channels,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        if not sample_audio:
            if audio_latents is not None:
                raise ValueError("audio_latents cannot be supplied when sample_audio=False")
            return bundle
        if self.audio_vae is None:
            raise ValueError("sample_audio=True requires an audio_vae")

        audio_dim = int(getattr(self.transformer, "in_audio_dim", 20))
        downsample_factor = int(self.audio_vae.downsample_factor)
        audio_duration = audio_latent_duration(
            latent_frames,
            fps=sample_fps,
            audio_fps=self.audio_sample_rate,
            downsample_factor=downsample_factor,
        )
        if audio_latents is None:
            bundle = prepare_audio_latents(
                bundle,
                audio_duration=audio_duration,
                audio_dim=audio_dim,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            bundle.audio = bundle.audio.reshape(batch_size, audio_duration, audio_dim)
            return bundle
        return self._coerce_audio_latents(
            bundle,
            audio_latents,
            audio_duration=audio_duration,
            audio_dim=audio_dim,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _move_text_embeds(
        embeds: TextEmbeds,
        cu_seqlens: Tensor,
        attention_mask: Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[TextEmbeds, Tensor, Tensor | None]:
        moved = {key: value.to(device=device, dtype=dtype) for key, value in embeds.items()}
        mask = attention_mask.to(device=device) if attention_mask is not None else None
        return moved, cu_seqlens.to(device=device), mask

    @staticmethod
    def _packed_bundle(bundle: LatentBundle) -> LatentBundle:
        return LatentBundle(
            video=bundle.video.flatten(0, 1) if bundle.video is not None else None,
            audio=bundle.audio.flatten(0, 1) if bundle.audio is not None else None,
            video_cu_seqlens=bundle.video_cu_seqlens,
            audio_cu_seqlens=bundle.audio_cu_seqlens,
        )

    @staticmethod
    def _text_length(embeds: TextEmbeds, cu_seqlens: Tensor) -> int:
        value = embeds["text_embeds"]
        return int(value.shape[1] if value.ndim == 3 else cu_seqlens[-1].item())

    def _text_ropes(
        self,
        text_length: int,
        negative_text_length: int,
        audio_length: int | None,
        device: torch.device,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor], Tensor | None]:
        if getattr(self.transformer, "is_multimodal", False):
            text_rope = [
                compute_rope1d(self.transformer.video_text_rope_embeddings, text_length, device=device),
                compute_rope1d(self.transformer.audio_text_rope_embeddings, text_length, device=device),
            ]
            negative_text_rope = [
                compute_rope1d(
                    self.transformer.video_text_rope_embeddings,
                    negative_text_length,
                    device=device,
                ),
                compute_rope1d(
                    self.transformer.audio_text_rope_embeddings,
                    negative_text_length,
                    device=device,
                ),
            ]
        else:
            text_rope = compute_rope1d(self.transformer.text_rope_embeddings, text_length, device=device)
            negative_text_rope = compute_rope1d(
                self.transformer.text_rope_embeddings,
                negative_text_length,
                device=device,
            )
        audio_rope = (
            compute_rope1d(self.transformer.audio_rope_embeddings, audio_length, device=device)
            if audio_length is not None
            else None
        )
        return text_rope, negative_text_rope, audio_rope

    @staticmethod
    def _format_video(frames: Tensor, output_type: str) -> Tensor | np.ndarray:
        if output_type in ("pt", "torch"):
            return frames
        if output_type in ("np", "numpy"):
            return frames.permute(0, 2, 3, 4, 1).cpu().numpy()
        raise ValueError("output_type must be 'pt', 'torch', 'np', 'numpy', or 'latent'")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: str | object | list[str | object] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        sample_fps: float = 24.0,
        num_inference_steps: int = 50,
        max_sequence_length: int = 1024,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
        latents: Tensor | None = None,
        audio_latents: Tensor | None = None,
        prompt_embeds: TextEmbeds | None = None,
        negative_prompt_embeds: TextEmbeds | None = None,
        cu_seqlens: Tensor | tuple[Tensor, Tensor] | None = None,
        prompt_cu_seqlens: Tensor | None = None,
        negative_prompt_cu_seqlens: Tensor | None = None,
        sample_audio: bool = True,
        expand_prompts: bool = False,
        output_type: str = "pt",
        return_dict: bool = True,
        callback_on_step_end: Callable[..., Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        visual_cond_scheme: str | None = None,
    ) -> Kandinsky6TI2VAPipelineOutput | tuple[Tensor | np.ndarray, list[np.ndarray] | None]:
        r"""Generate synchronized video and, optionally, audio from text or an image.

        Args:
            prompt: Text prompt or a batch of prompts.
            image: Optional reference image or batch of reference images.
            negative_prompt: Optional negative prompt.
            height: Output video height in pixels; must be divisible by 8.
            width: Output video width in pixels; must be divisible by 8.
            num_frames: Number of decoded video frames.
            sample_fps: Output video frame rate. Kandinsky 6 is trained for
                24.0 fps; other values may affect audio/video alignment.
            num_inference_steps: Number of denoising steps.
            max_sequence_length: Maximum Qwen prompt length after the template.
            guidance_scale: Classifier-free guidance weight.
            generator: Random generator used for latent initialization.
            latents: Optional precomputed video latents.
            audio_latents: Optional precomputed audio latents.
            prompt_embeds: Optional precomputed positive embeddings.
            negative_prompt_embeds: Optional precomputed negative embeddings.
            cu_seqlens: Optional positive/negative cumulative sequence lengths.
            prompt_cu_seqlens: Optional cumulative sequence lengths for
                precomputed positive prompt embeddings.
            negative_prompt_cu_seqlens: Optional cumulative sequence lengths
                for precomputed negative prompt embeddings.
            sample_audio: Whether to generate synchronized audio.
            expand_prompts: Whether to use the built-in Qwen video+audio prompt expander before encoding.
            output_type: ``pt``/``torch`` for tensors, ``np``/``numpy`` for
                NumPy arrays, or ``latent`` for undecoded video latents.
            return_dict: Whether to return ``Kandinsky6TI2VAPipelineOutput``.
            callback_on_step_end: Optional callback invoked after each step.
            callback_on_step_end_tensor_inputs: Names passed to the callback.
            visual_cond_scheme: Optional image-conditioning scheme. Defaults to
                ``pretrain`` without ``image`` and ``tail_cond_first_frame``
                with ``image``.

        Examples:

        Returns:
            ``Kandinsky6TI2VAPipelineOutput`` containing video frames and int16 NumPy
            audio waveforms, or a tuple when ``return_dict=False``.
        """
        if cu_seqlens is not None:
            if isinstance(cu_seqlens, tuple):
                if len(cu_seqlens) != 2:
                    raise ValueError("cu_seqlens tuple must contain positive and negative lengths")
                if prompt_cu_seqlens is None:
                    prompt_cu_seqlens = cu_seqlens[0]
                if negative_prompt_cu_seqlens is None:
                    negative_prompt_cu_seqlens = cu_seqlens[1]
            else:
                if prompt_cu_seqlens is None:
                    prompt_cu_seqlens = cu_seqlens
                if negative_prompt_cu_seqlens is None:
                    negative_prompt_cu_seqlens = cu_seqlens

        if sample_fps != 24.0:
            warnings.warn(
                f"Kandinsky 6 was trained for 24.0 fps; received sample_fps={sample_fps}.",
                UserWarning,
                stacklevel=2,
            )

        batch_size = self._batch_size(
            prompt,
            negative_prompt,
            image,
            latents,
            audio_latents,
            prompt_embeds,
            negative_prompt_embeds,
        )
        prompt_batch = self._as_batch(prompt, batch_size, "prompt")
        negative_prompt_batch = self._as_batch(negative_prompt, batch_size, "negative_prompt")
        image_batch = self._as_batch(image, batch_size, "image")
        if visual_cond_scheme is None:
            visual_cond_scheme = "tail_cond_first_frame" if image is not None else "pretrain"
        self.check_inputs(
            image=image_batch,
            prompt=prompt_batch,
            negative_prompt=negative_prompt_batch,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            output_type=output_type,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_cu_seqlens=prompt_cu_seqlens,
            negative_prompt_cu_seqlens=negative_prompt_cu_seqlens,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            sample_audio=sample_audio,
            visual_cond_scheme=visual_cond_scheme,
        )

        callback_inputs = (
            self._callback_tensor_inputs
            if callback_on_step_end_tensor_inputs is None
            else callback_on_step_end_tensor_inputs
        )
        if expand_prompts and prompt_batch is not None and prompt_embeds is None:
            prompt_batch = self.expand_prompts(
                prompt_batch,
                image=image_batch,
                max_sequence_length=max_sequence_length,
                mode="i2va" if image_batch is not None else "t2va",
            )
        patch_size = tuple(int(value) for value in getattr(self.transformer, "patch_size", (1, 2, 2)))

        device = getattr(self, "_execution_device", self.transformer.device)
        dtype = getattr(self.transformer, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        first_frames = None
        if image_batch is not None:
            encoded_frames = [
                encode_i2va_first_frame(item, self.vae, device, height=height, width=width) for item in image_batch
            ]
            first_frames = torch.cat([item[0] for item in encoded_frames], dim=0)
            encoded_height, encoded_width = encoded_frames[0][1:]
            if any(item[1:] != (encoded_height, encoded_width) for item in encoded_frames):
                raise ValueError("all reference images must produce the same resolution")
            height, width = encoded_height, encoded_width
        negative_prompt_batch = negative_prompt_batch or [self._DEFAULT_NEGATIVE_PROMPT] * batch_size
        (
            positive,
            positive_cu,
            positive_mask,
            negative,
            negative_cu,
            negative_mask,
        ) = self._encode_prompt(
            prompt_batch or [""] * batch_size,
            negative_prompt_batch,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_cu_seqlens=prompt_cu_seqlens,
            negative_prompt_cu_seqlens=negative_prompt_cu_seqlens,
            max_sequence_length=max_sequence_length,
        )
        if positive_cu is None or negative_cu is None:
            raise ValueError("positive and negative prompt cumulative lengths are required")
        positive, positive_cu, positive_mask = self._move_text_embeds(
            positive,
            positive_cu,
            positive_mask,
            device=device,
            dtype=dtype,
        )
        negative, negative_cu, negative_mask = self._move_text_embeds(
            negative,
            negative_cu,
            negative_mask,
            device=device,
            dtype=dtype,
        )

        latent_frames = (num_frames - 1) // 4 + 1
        bundle = self._prepare_latents(
            latents=latents,
            audio_latents=audio_latents,
            latent_frames=latent_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            sample_fps=sample_fps,
            sample_audio=sample_audio,
            batch_size=batch_size,
        )
        visual_token_type_ids = None
        generated_visual_mask = None
        if image is not None and visual_cond_scheme == "tail_cond_first_frame":
            if bundle.video is None or first_frames is None:
                raise ValueError("image conditioning requires video latents")
            video, visual_token_type_ids, generated_visual_mask = append_i2va_tail_condition(
                bundle.video,
                first_frames,
                batch_size=batch_size,
                video_duration=latent_frames,
            )
            bundle = LatentBundle(
                video=video,
                audio=bundle.audio,
                video_cu_seqlens=(latent_frames + 1) * torch.arange(batch_size + 1, dtype=torch.int32, device=device),
                audio_cu_seqlens=bundle.audio_cu_seqlens,
            )
        audio_length = int(bundle.audio.shape[1]) if bundle.audio is not None else None
        visual_shape = (
            latent_frames // patch_size[0],
            (height // 8) // patch_size[1],
            (width // 8) // patch_size[2],
        )
        if visual_shape[0] < 1:
            raise ValueError(f"invalid visual latent shape {visual_shape}")
        visual_rope = compute_visual_rope(
            self.transformer.visual_rope_embeddings,
            visual_shape,
            self.scale_factor,
            device=device,
        )
        if generated_visual_mask is not None:
            visual_rope = torch.cat([visual_rope, visual_rope[:1]], dim=0)
        text_rope, negative_text_rope, audio_rope = self._text_ropes(
            text_length=self._text_length(positive, positive_cu),
            negative_text_length=self._text_length(negative, negative_cu),
            audio_length=audio_length,
            device=device,
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:

            def _step_callback(step_index: int, timestep: Tensor, current: LatentBundle) -> LatentBundle:
                if callback_on_step_end is not None:
                    callback_result = callback_on_step_end(
                        self,
                        step_index,
                        timestep,
                        {name: current.video for name in callback_inputs},
                    )
                    if callback_result is None:
                        callback_result = {}
                    if not isinstance(callback_result, dict):
                        raise TypeError("callback_on_step_end must return a dict or None")
                    if callback_result.get("latents") is not None:
                        current.video = callback_result["latents"]
                return current

            result = denoise_loop(
                bundle=bundle,
                dit=self.transformer,
                text_embeds=positive,
                null_text_embeds=negative,
                visual_rope=visual_rope,
                audio_rope=audio_rope,
                text_rope=text_rope,
                null_text_rope=negative_text_rope,
                num_steps=num_inference_steps,
                guidance_weight=guidance_scale,
                first_frames=first_frames,
                visual_cond_scheme=visual_cond_scheme,
                sample_video=True,
                sample_audio=sample_audio,
                attention_mask=positive_mask,
                null_attention_mask=negative_mask,
                visual_token_type_ids=visual_token_type_ids,
                scale_factor=self.scale_factor,
                scheduler=self.scheduler,
                step_callback=_step_callback if callback_on_step_end is not None else None,
                progress_bar=progress_bar,
            )

        if generated_visual_mask is not None and result.video is not None:
            generated_video = result.video[:, generated_visual_mask[0]]
            result = LatentBundle(
                video=generated_video,
                audio=result.audio,
                video_cu_seqlens=latent_frames * torch.arange(batch_size + 1, dtype=torch.int32, device=device),
                audio_cu_seqlens=result.audio_cu_seqlens,
            )

        if output_type == "latent":
            if result.video is None:
                raise ValueError("denoising returned no video latent")
            frames: Tensor | np.ndarray = result.video.permute(0, 4, 1, 2, 3)
        else:
            packed_result = self._packed_bundle(result)
            decoded_frames = postprocess_video(packed_result, self.vae, bs=batch_size)
            frames = self._format_video(decoded_frames, output_type)
        audio = postprocess_audio(self._packed_bundle(result), self.audio_vae) if sample_audio else None
        if not return_dict:
            return frames, audio
        return Kandinsky6TI2VAPipelineOutput(frames=frames, audio=audio)
