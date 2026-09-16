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

"""Kandinsky 6 super-resolution Diffusers pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple

import av
import numpy as np
import torch
from ..pipeline_utils import DiffusionPipeline
from diffusers.utils import replace_example_docstring
from pydantic import BaseModel, ConfigDict, Field
from ...schedulers.scheduling_piflow import DXPolicy, policy_rollout_fm, shift_timesteps
from torch import Tensor, nn
from torch.distributed import all_gather
from torch.nn import functional

from .pipeline_output import Kandinsky6SRPipelineOutput

EXAMPLE_DOC_STRING = """
    Examples:

        ```python
        >>> import torch
        >>> from diffusers import Kandinsky6SRPipeline

        >>> model_id = "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers"
        >>> pipe = Kandinsky6SRPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        >>> pipe = pipe.to("cuda")

        >>> output = pipe.from_video(
        ...     "input.mp4",
        ...     resolution_scale=2.25,
        ...     save_path="output.mp4",
        ... )
        >>> output.path
        'output.mp4'
        ```
"""


# These values are the SR data contract and must stay local to this pipeline.
VAE_SPATIAL_FACTOR = 16
VAE_TEMPORAL_FACTOR = 4
RESOLUTIONS: dict[int, list[tuple[int, int]]] = {512: [(512, 512), (512, 768), (768, 512)]}
TARGET_FPS = 24
MAX_NUM_FRAMES = 121
RESAMPLE_FPS_TOLERANCE = 1.5
LATENT_NDIM = 4
X0_T_CLAMP = 1e-5
SUPPORTED_TARGET_SCALES = ("2x", "4x")
ResolutionScale = Literal[2, 4]
TotalResolutionScale = Literal[2, 4, 2.25]
constants = SimpleNamespace(VAE_SPATIAL_FACTOR=VAE_SPATIAL_FACTOR)
TileGridMode = Literal["legacy", "even"]
_sr_constants = None


def _call_module_method(module: nn.Module, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call a non-forward method through a Diffusers offload hook."""
    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return getattr(module, method)(*args, **kwargs)
    args, kwargs = hook.pre_forward(module, *args, **kwargs)
    output = getattr(module, method)(*args, **kwargs)
    return hook.post_forward(module, output)


@contextmanager
def _execution_device_context(device: torch.device | str):
    """Scope SR model execution to Diffusers' resolved CUDA device."""
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        with torch.cuda.device(resolved_device):
            yield
    else:
        yield


# Shared tiling utilities for spatial video tiling with Hanning-window blending.


class TileGrid(NamedTuple):
    """Spatial tile grid parameters.

    ``tops`` and ``lefts`` are the authoritative per-axis tile start positions.
    They cover ``[0, length - tile]`` with the nominal stride; the **last**
    position is clamped to ``length - tile`` if a uniform stride would
    overshoot. This means consecutive gaps in ``tops`` / ``lefts`` may be
    shorter than the nominal stride near the right/bottom edge.

    ``n_h``, ``n_w``, ``total_tiles``, ``stride_h``, ``stride_w`` are derived
    properties exposed for backward compatibility with callers that only
    read them. To compute a tile position by index, prefer
    ``grid.tops[row]`` / ``grid.lefts[col]`` over ``row * grid.stride_h`` —
    the latter is wrong for the last clamped tile.
    """

    tile_h: int
    tile_w: int
    tops: tuple[int, ...]
    lefts: tuple[int, ...]

    @property
    def n_h(self) -> int:
        """Number of tile rows."""
        return len(self.tops)

    @property
    def n_w(self) -> int:
        """Number of tile columns."""
        return len(self.lefts)

    @property
    def total_tiles(self) -> int:
        """Total number of tiles (``n_h * n_w``)."""
        return self.n_h * self.n_w

    @property
    def stride_h(self) -> int:
        """Nominal vertical stride (gap between first two rows; ``tile_h`` if only one row)."""
        return self.tops[1] - self.tops[0] if len(self.tops) > 1 else self.tile_h

    @property
    def stride_w(self) -> int:
        """Nominal horizontal stride (gap between first two cols; ``tile_w`` if only one col)."""
        return self.lefts[1] - self.lefts[0] if len(self.lefts) > 1 else self.tile_w


def axis_positions(length: int, tile: int, stride: int) -> tuple[int, ...]:
    """Return tile start positions covering ``[0, length - tile]``.

    Walks at the given ``stride`` from 0 and appends positions while the tile
    still fits inside ``length``. If the last walked position is not exactly
    ``length - tile``, appends one final position clamped to the right edge.
    Single-position result if ``tile >= length``.

    Args:
        length: Axis length in pixels.
        tile: Tile size on this axis.
        stride: Nominal stride between consecutive tiles.

    Returns:
        Strictly increasing tuple of start positions.
    """
    if tile <= 0 or stride <= 0:
        msg = f"tile and stride must be positive (got tile={tile}, stride={stride})"
        raise ValueError(msg)
    if tile >= length:
        return (0,)

    positions: list[int] = []
    p = 0
    while p + tile <= length:
        positions.append(p)
        p += stride
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return tuple(positions)


def compute_tile_grid(
    h: int,
    w: int,
    resolution_scale: int,
    overlap: float = 0.5,
    tile_hw: tuple[int, int] | None = None,
) -> TileGrid:
    """Compute tile geometry with configurable overlap.

    By default, tile size is ``(h // resolution_scale, w // resolution_scale)``.
    Pass ``tile_hw`` to override with explicit tile dimensions (e.g. derived
    from a fixed base resolution rather than the source frame size).

    The grid covers the full frame: if the nominal stride does not divide
    evenly into ``(h - tile_h)`` / ``(w - tile_w)``, the last tile in each
    axis is clamped to the right/bottom edge. ``stitch_tiles_hanning``
    handles the resulting non-uniform overlap correctly via Hanning-window
    normalisation.

    Args:
        h: Video height in pixels.
        w: Video width in pixels.
        resolution_scale: Divisor for tile size when ``tile_hw`` is ``None``.
        overlap: Fraction of tile overlap in ``[0, 1)``. Default ``0.5`` (50%).
        tile_hw: Explicit ``(tile_h, tile_w)`` override. When set,
            ``resolution_scale`` is ignored for tile sizing.

    Returns:
        ``TileGrid`` with tile sizes and per-axis tile start positions.
    """
    if tile_hw is None:
        tile_h = h // resolution_scale
        tile_w = w // resolution_scale
    else:
        tile_h, tile_w = tile_hw
    stride_h = max(1, int(tile_h * (1.0 - overlap)))
    stride_w = max(1, int(tile_w * (1.0 - overlap)))

    tops = axis_positions(h, tile_h, stride_h)
    lefts = axis_positions(w, tile_w, stride_w)
    return TileGrid(tile_h=tile_h, tile_w=tile_w, tops=tops, lefts=lefts)


def extract_all_tiles(
    video: torch.Tensor,
    grid: TileGrid,
) -> list[torch.Tensor]:
    """Extract all spatial tiles from a video tensor.

    Args:
        video: ``[T, C, H, W]`` tensor.
        grid: Tile grid parameters from ``compute_tile_grid``.

    Returns:
        List of ``[T, C, tile_h, tile_w]`` tensors in row-major order
        (``tops`` x ``lefts``).
    """
    tiles: list[torch.Tensor] = []
    for top in grid.tops:
        for left in grid.lefts:
            tile = video[:, :, top : top + grid.tile_h, left : left + grid.tile_w]
            tiles.append(tile)
    return tiles


def hanning_window_2d(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Hanning window with non-zero endpoints.

    Uses ``hann_window(n + 2)[1:-1]`` to avoid exact zeros at boundaries,
    ensuring non-zero weight where only one tile contributes.

    Args:
        h: Window height.
        w: Window width.
        device: Target device.

    Returns:
        ``[h, w]`` float tensor with values in ``(0, 1]``.
    """
    wy = torch.hann_window(h + 2, device=device)[1:-1]
    wx = torch.hann_window(w + 2, device=device)[1:-1]
    return wy[:, None] * wx[None, :]


def stitch_tiles_hanning(
    tiles: list[torch.Tensor],
    grid: TileGrid,
    original_h: int,
    original_w: int,
    scale: int = 1,
) -> torch.Tensor:
    """Stitch tiles into a full frame using Hanning-window weighted blending.

    Each tile is multiplied by a 2D Hanning window and accumulated into
    the output canvas. The final result is normalised by the accumulated
    weights so that overlapping regions blend smoothly. The same scheme
    works for non-uniform overlap at the right/bottom edge: in the clamped
    region both ``pred_acc`` and ``weight_acc`` receive more contributions,
    and the per-pixel division cancels it out.

    When ``scale > 1``, tiles are assumed to be at HR resolution
    (i.e. each tile covers ``tile_h * scale x tile_w * scale`` pixels)
    and the output canvas is ``original_h * scale x original_w * scale``.
    Grid positions are scaled accordingly.

    Args:
        tiles: List of ``[C, T, th, tw]`` float tensors in row-major order
            (same order as ``extract_all_tiles``). When ``scale == 1``,
            ``th == grid.tile_h``; when ``scale > 1``, ``th == grid.tile_h * scale``.
        grid: Tile grid parameters (at LQ / original resolution).
        original_h: LQ frame height.
        original_w: LQ frame width.
        scale: Upscale factor. Output resolution is
            ``(original_h * scale, original_w * scale)``.

    Returns:
        ``[C, T, original_h * scale, original_w * scale]`` float tensor.
    """
    first = tiles[0]
    c, t = first.shape[0], first.shape[1]
    hr_tile_h, hr_tile_w = first.shape[2], first.shape[3]
    device = first.device

    window = hanning_window_2d(hr_tile_h, hr_tile_w, device)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, hr_tile_h, hr_tile_w]

    out_h = original_h * scale
    out_w = original_w * scale
    pred_acc = torch.zeros(c, t, out_h, out_w, device=device)
    weight_acc = torch.zeros(1, 1, out_h, out_w, device=device)

    tile_idx = 0
    for top in grid.tops:
        for left in grid.lefts:
            y = top * scale
            x = left * scale
            tile = tiles[tile_idx]
            pred_acc[:, :, y : y + hr_tile_h, x : x + hr_tile_w] += tile * window
            weight_acc[:, :, y : y + hr_tile_h, x : x + hr_tile_w] += window
            tile_idx += 1

    return pred_acc / weight_acc.clamp(min=1e-6)


def axis_positions_even(length: int, tile: int, min_overlap: float, snap: int) -> tuple[int, ...]:
    """Return evenly distributed, ``snap``-aligned tile start positions.

    Uses the minimal position count whose uniform stride keeps the tile
    overlap at or above ``min_overlap``, then spreads the positions evenly
    over ``[0, length - tile]`` in integer ``snap`` units. When ``length`` or
    ``tile`` is not ``snap``-aligned the same layout is computed at pixel
    precision (``snap=1``) — the latent-grid guard downstream still enforces
    alignment where it actually matters (the LU path).

    Args:
        length: Axis length in pixels.
        tile: Tile size on this axis.
        min_overlap: Overlap floor as a fraction of ``tile`` in ``[0, 1)``.
        snap: Position alignment unit (the VAE spatial factor).

    Returns:
        Strictly increasing positions covering ``[0, length - tile]``.

    Raises:
        ValueError: On non-positive ``tile``/``snap`` or ``min_overlap``
            outside ``[0, 1)``.
    """
    if tile <= 0 or snap <= 0 or not 0 <= min_overlap < 1:
        msg = f"invalid axis spec: tile={tile}, snap={snap}, min_overlap={min_overlap}"
        raise ValueError(msg)
    if tile >= length:
        return (0,)
    unit = snap if length % snap == 0 and tile % snap == 0 else 1
    span_units = (length - tile) // unit
    max_stride_units = max(1, math.floor(tile * (1.0 - min_overlap) / unit))
    count = math.ceil(span_units / max_stride_units) + 1
    return tuple(round(i * span_units / (count - 1)) * unit for i in range(count))


def compute_tile_grid_even(
    h: int,
    w: int,
    tile_hw: tuple[int, int],
    min_overlap: float,
    snap: int,
) -> TileGrid:
    """Build a :class:`TileGrid` with the even per-axis layout.

    Drop-in for ``compute_tile_grid(..., tile_hw=...)``: same ``TileGrid``
    contract (``tops``/``lefts`` are authoritative), only the position layout
    differs — see :func:`axis_positions_even`.

    Args:
        h: Video height in pixels.
        w: Video width in pixels.
        tile_hw: Explicit ``(tile_h, tile_w)``.
        min_overlap: Overlap floor as a fraction of the tile size.
        snap: Position alignment unit (the VAE spatial factor).

    Returns:
        ``TileGrid`` with evenly distributed, aligned tile positions.
    """
    tile_h, tile_w = tile_hw
    return TileGrid(
        tile_h=tile_h,
        tile_w=tile_w,
        tops=axis_positions_even(h, tile_h, min_overlap, snap),
        lefts=axis_positions_even(w, tile_w, min_overlap, snap),
    )


def resolve_scale_request(scale: float) -> tuple[int, float]:
    """Map a requested ``--resolution-scale`` to ``(tiling_scale, pre_upscale)``.

    Args:
        scale: The requested total upscale — ``2``, ``4`` or ``2.25``.

    Returns:
        ``(tiling_scale, pre_upscale)``; ``pre_upscale`` is ``1.0`` for the
        integer scales.
    """
    requested = float(scale)
    if requested == 2.25:  # noqa: PLR2004 — the one supported fractional total
        return 2, 1.125
    if requested in (2.0, 4.0):
        return int(requested), 1.0
    raise ValueError("SR supports total scales 2, 4, and 2.25")


def pre_upscale_video(video: torch.Tensor, factor: float, spatial_multiple: int) -> torch.Tensor:
    """Bilinear-upscale a ``[T, C, H, W]`` uint8 video by ``factor`` in pixel space.

    Target dims are rounded to the nearest multiple of ``spatial_multiple``
    (the VAE spatial factor) so the whole-video encode and the latent tile
    grid stay integer-aligned (``latent_tile_grid_from_pixel_grid`` rejects
    unaligned grids). For sources whose scaled dims already land on the
    factor (512x768 x1.125 -> 576x864) the rounding is a no-op and the total
    scale is exact.

    Args:
        video: ``[T, C, H, W]`` uint8 source video.
        factor: Pixel upscale factor (> 1).
        spatial_multiple: VAE spatial factor to align the target dims to.

    Returns:
        ``[T, C, H', W']`` uint8 video with ``H' ~= H * factor`` aligned.
    """
    if video.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"video must have rank 4 [T,C,H,W], got {tuple(video.shape)}")
    if factor <= 0 or spatial_multiple <= 0:
        raise ValueError("factor and spatial_multiple must be positive")
    height, width = video.shape[-2:]
    target_h = max(spatial_multiple, round(height * factor / spatial_multiple) * spatial_multiple)
    target_w = max(spatial_multiple, round(width * factor / spatial_multiple) * spatial_multiple)
    resized = functional.interpolate(
        video.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.round_().clamp_(0, 255).to(torch.uint8)


def get_visual_size(x: torch.Tensor) -> int:
    """Return the resolution key matching a visual latent tensor's spatial dims.

    Scales the tensor's spatial dims by the VAE spatial factor and looks them
    up in the ``RESOLUTIONS`` registry. Reads ``constants.VAE_SPATIAL_FACTOR``
    late-bound so ``set_vae_factors`` (16 for the KVAE) applies — a
    ``from``-import would freeze the import-time default.

    Args:
        x: Visual latent tensor of shape ``(T, H, W, C)``.

    Returns:
        The matching resolution key.

    Raises:
        ValueError: If tensor dimensions do not match any known resolution.
    """
    actual_size = (x.shape[1] * constants.VAE_SPATIAL_FACTOR, x.shape[2] * constants.VAE_SPATIAL_FACTOR)
    for key, value in RESOLUTIONS.items():
        if actual_size in value:
            return key
    valid_sizes = {size for sizes in RESOLUTIONS.values() for size in sizes}
    msg = (
        f"Visual tensor spatial dimensions {actual_size} do not match any known resolution. "
        f"Tensor shape: {x.shape}. Valid resolutions: {valid_sizes}"
    )
    raise ValueError(msg)


def degrade_lq_latent(
    lq_latent: torch.Tensor,
    noise_scale: float = 0.7,
    noise_type: Literal["linear", "ddpm"] = "linear",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mix random Gaussian noise into an LQ latent for data augmentation.

    Args:
        lq_latent: LQ latent tensor of arbitrary shape.
        noise_scale: Noise fraction ``s`` in ``(0, 1)``. When ``0``, the tensor
            is returned unchanged.
        noise_type: ``"linear"`` for ``(1-s)*lq + s*eps`` or ``"ddpm"`` for
            ``sqrt(1-s²)*lq + s*eps`` (variance-preserving).
        generator: Optional RNG generator for deterministic noise sampling.

    Returns:
        Degraded LQ latent with the same shape and dtype as the input.
    """
    if noise_scale <= 0:
        return lq_latent
    eps = torch.randn(lq_latent.shape, device=lq_latent.device, dtype=lq_latent.dtype, generator=generator)
    if noise_type == "ddpm":
        return (1 - noise_scale**2) ** 0.5 * lq_latent + noise_scale * eps
    return (1 - noise_scale) * lq_latent + noise_scale * eps


def get_sparse_params(
    model: Any,
    visual: torch.Tensor,
    visual_cu_seqlens: torch.Tensor,
) -> dict[str, Any] | None:
    """Build sparse-attention parameters for the given visual tensor and model.

    Supports ``"nabla"`` and ``"nabla_framewise_causal"`` attention types;
    returns ``None`` for dense attention. The target ``P`` is used directly (the
    training-time linear warmup schedule does not apply at inference).

    Args:
        model: Model with ``patch_size`` and ``attention_params`` attributes.
        visual: Visual latent tensor of shape ``(T, H, W, C)``.
        visual_cu_seqlens: Cumulative frame counts for each sample in the batch.

    Returns:
        A dict of sparse attention parameters, or ``None`` for dense attention.

    Raises:
        ValueError: If ``model.patch_size[0]`` is not 1.
    """
    if model.patch_size[0] != 1:
        msg = f"Expected model.patch_size[0] == 1, got {model.patch_size[0]}"
        raise ValueError(msg)
    t, h, w, _ = visual.shape
    t, h, w = (
        t // model.patch_size[0],
        h // model.patch_size[1],
        w // model.patch_size[2],
    )
    visual_size = get_visual_size(visual)
    attention_configs = model.attention_params
    try:
        attention_params = attention_configs[visual_size]
    except KeyError:
        # JSON object keys are always strings, while the native YAML config
        # uses integer resolution keys.
        attention_params = attention_configs[str(visual_size)]

    def config_value(name: str, default: Any = None) -> Any:
        if isinstance(attention_params, Mapping):
            return attention_params.get(name, default)
        return getattr(attention_params, name, default)

    if config_value("type") == "nabla":
        return {
            "attention_type": config_value("type"),
            "to_fractal": True,
            "P": config_value("P"),
            "wT": config_value("wT"),
            "wW": config_value("wW"),
            "wH": config_value("wH"),
            "add_sta": config_value("add_sta"),
            "visual_shape": (t, h, w),
            "visual_seqlens": visual_cu_seqlens,
            "method": config_value("method", "topcdf"),
        }
    if config_value("type") == "nabla_framewise_causal":
        return {
            "attention_type": config_value("type"),
            "to_fractal": True,
            "P": config_value("P"),
            "wT": config_value("wT"),
            "wW": config_value("wW"),
            "wH": config_value("wH"),
            "add_sta": config_value("add_sta"),
            "mf": config_value("mf"),
            "visual_shape": (t, h, w),
            "visual_seqlens": visual_cu_seqlens,
        }
    return None


def replicate_cached_embeds(
    cached: dict[str, torch.Tensor],
    bs: int,
    device: str | int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Replicate bs=1 cached text embeddings for an arbitrary batch size.

    Args:
        cached: Dict with ``text_embeds``, ``pooled_embed``, ``cu_seqlens``
            produced by ``save_empty_text_embeddings.py`` for a single sample.
        bs: Target batch size.
        device: Target device.

    Returns:
        Tuple of (text_embeds dict, cu_seqlens tensor) for the full batch.
    """
    single_text = cached["text_embeds"]
    single_pooled = cached["pooled_embed"]
    seq_len = int(cached["cu_seqlens"][-1].item())

    text_embeds = single_text.repeat(bs, 1).to(device)
    pooled_embed = single_pooled.repeat(bs, 1).to(device)
    cu_seqlens = seq_len * torch.arange(bs + 1, dtype=torch.int32, device=device)

    return {"text_embeds": text_embeds, "pooled_embed": pooled_embed}, cu_seqlens


def kvae_weights_path(checkpoint_prefix: str) -> str:
    """Return the kvae weights file for a sidecar prefix, preferring safetensors.

    The release format is ``{prefix}.safetensors`` (flat state dict), the
    training format ``{prefix}.ckpt`` (pickled ``{"state_dict": ...}``); both
    load through ``CachedCausalVAE.init_from_ckpt``, which dispatches on the
    suffix.

    Args:
        checkpoint_prefix: Local kvae sidecar prefix (``{prefix}.yaml`` sits
            next to the weights).

    Returns:
        Path to the weights file to load.
    """
    safetensors_path = f"{checkpoint_prefix}.safetensors"
    return safetensors_path if Path(safetensors_path).exists() else f"{checkpoint_prefix}.ckpt"


def module_dtype(module: torch.nn.Module) -> torch.dtype | None:
    """Return the dtype used by the first parameterized layer, if any."""
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return None


def cast_to_module_dtype(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Cast floating-point inputs to the module's parameter dtype."""
    dtype = module_dtype(module)
    if dtype is not None and value.is_floating_point() and value.dtype != dtype:
        return value.to(dtype=dtype)
    return value


def encode_pixels_to_latent(vae: torch.nn.Module, pixels: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
    """Encode ``(B, C, T, H, W)`` pixels in ``[0, 255]`` with the KVAE.

    Normalizes with the KVAE's own convention, then returns the latent
    (``(latent, split_list)[0]`` — the regularizer mode).

    Args:
        vae: Causal video KVAE.
        pixels: ``(B, C, T, H, W)`` tensor in ``[0, 255]``.
        sample: Unused — the KVAE always returns the regularizer mode.

    Returns:
        Latent ``(B, C, T', H', W')`` — not yet scaled by ``scaling_factor``.
    """
    del sample  # the KVAE always returns the regularizer mode
    x = vae.normalize_data(pixels)
    x = cast_to_module_dtype(vae, x)
    result = vae.encode(x)
    return result[0]


def decode_latent_to_uint8(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Decode a ``(B, C, T, H, W)`` latent to ``uint8`` pixels in ``[0, 255]``.

    The latent must already be unscaled (divided by ``scaling_factor``);
    denormalization uses the KVAE's ``denormalize_data``.

    Args:
        vae: Causal video KVAE.
        latent: ``(B, C, T, H, W)`` unscaled latent.

    Returns:
        ``(B, C, T_pixel, H, W)`` ``uint8`` pixels in ``[0, 255]``.
    """
    latent = cast_to_module_dtype(vae, latent)
    return denormalize_to_uint8(vae, vae.decode(latent).sample)


def denormalize_to_float(vae: torch.nn.Module, decoded: torch.Tensor) -> torch.Tensor:
    """Convert a decoded tensor to float pixels in ``[0, 1]`` without quantization.

    Args:
        vae: Causal video KVAE.
        decoded: Decode output in the VAE's normalized pixel space.

    Returns:
        Float pixels in ``[0, 1]`` with the same shape.
    """
    return (vae.denormalize_data(decoded) / 255.0).clamp(0.0, 1.0)


def denormalize_to_uint8(vae: torch.nn.Module, decoded: torch.Tensor) -> torch.Tensor:
    """Convert an already-decoded tensor in ``[-1, 1]`` to ``uint8`` ``[0, 255]``.

    Uses the KVAE ``denormalize_data`` convention ((x + 1) * 128). For decode
    outputs that did not pass through :func:`decode_latent_to_uint8` (e.g.
    debug-clip dumps of already-decoded pixels).

    Args:
        vae: Causal video KVAE.
        decoded: Decode output in ``[-1, 1]`` (any shape).

    Returns:
        ``uint8`` pixels in ``[0, 255]`` with the same shape.
    """
    return (denormalize_to_float(vae, decoded) * 255.0).to(torch.uint8)


def latent_upscaler_scale(upscaler: nn.Module) -> int:
    """Return the loaded LU's fixed spatial upscale factor."""
    raw = str(getattr(upscaler, "target_scale", "4x"))
    if raw not in SUPPORTED_TARGET_SCALES:
        msg = f"Unexpected latent upscaler target_scale={raw!r}; expected one of {SUPPORTED_TARGET_SCALES}."
        raise ValueError(msg)
    return int(raw.removesuffix("x"))


def run_latent_upscaler(upscaler: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Forward a scaled LQ latent through its configured LU entry."""
    target_scale = getattr(upscaler, "target_scale", None)
    if target_scale is not None:
        entry = f"x{latent_upscaler_scale(upscaler)}"
        try:
            return upscaler(z, entry=entry, return_intermediates=False)
        except TypeError:
            # Flat upscalers do not accept the multi-scale keyword arguments.
            return upscaler(z)
    return upscaler(z)


@torch.no_grad()
def get_model_prediction(
    dit: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    text_embeds: dict[str, torch.Tensor],
    null_text_embeds: dict[str, torch.Tensor],
    visual_cu_seqlens: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    null_text_cu_seqlens: torch.Tensor,
    visual_rope_pos: list[torch.Tensor],
    text_rope_pos: torch.Tensor,
    null_text_rope_pos: torch.Tensor,
    scale_factor: tuple[float, ...],
    guidance_weight: float,
    sparse_params: Any | None = None,
) -> torch.Tensor:
    """Compute classifier-free guidance model prediction.

    The returned tensor has the same semantics as the raw model output
    (velocity when ``prediction_target="velocity"``, clean ``x_0`` when
    ``prediction_target="x0"``).  The caller is responsible for converting
    the prediction to a velocity if needed for the Euler step.

    Args:
        dit: DiT model used for inference.
        x: Visual latent tensor to denoise.
        t: Timestep tensor broadcast over the batch.
        text_embeds: Dict with ``text_embeds`` and ``pooled_embed`` keys.
        null_text_embeds: Same structure as ``text_embeds`` but for null captions.
        visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
        text_cu_seqlens: Cumulative sequence lengths for text tokens.
        null_text_cu_seqlens: Cumulative sequence lengths for null text tokens.
        visual_rope_pos: RoPE position indices for visual tokens.
        text_rope_pos: RoPE position indices for text tokens.
        null_text_rope_pos: RoPE position indices for null text tokens.
        scale_factor: Per-axis RoPE frequency scaling.
        guidance_weight: CFG strength (1 = no guidance).
        sparse_params: Optional sparse attention parameters.

    Returns:
        CFG-combined model prediction (velocity or ``x_0``).
    """
    model_time = (t * 1000).to(dtype=x.dtype)
    motion_score = torch.full((1,), 900.0, device=x.device, dtype=x.dtype)
    cond_pred = dit(
        x,
        text_embeds["text_embeds"],
        text_embeds["pooled_embed"],
        model_time,
        visual_cu_seqlens,
        text_cu_seqlens,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=scale_factor,
        sparse_params=sparse_params,
        motion_score=motion_score,
    )
    if guidance_weight == 1.0:
        return cond_pred
    uncond_pred = dit(
        x,
        null_text_embeds["text_embeds"],
        null_text_embeds["pooled_embed"],
        model_time,
        visual_cu_seqlens,
        null_text_cu_seqlens,
        visual_rope_pos,
        null_text_rope_pos,
        scale_factor=scale_factor,
        sparse_params=sparse_params,
        motion_score=motion_score,
    )
    return uncond_pred + guidance_weight * (cond_pred - uncond_pred)


@torch.no_grad()
def generate(
    img: torch.Tensor,
    model: torch.nn.Module,
    device: str | int,
    num_steps: int,
    text_embeds: dict[str, torch.Tensor],
    null_text_embeds: dict[str, torch.Tensor],
    visual_cu_seqlens: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    null_text_cu_seqlens: torch.Tensor,
    visual_rope_pos: list[torch.Tensor],
    text_rope_pos: torch.Tensor,
    null_text_rope_pos: torch.Tensor,
    scale_factor: tuple[float, ...],
    guidance_weight: float,
    scheduler_scale: float,
    first_frames: torch.Tensor | None = None,
    tp_mesh: Any | None = None,
    visual_cond_scheme: str = "pretrain",
    start_timestep: float = 1.0,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    rfg_scale: float = 1.0,
    progress_callback: Callable[[int], Any] | None = None,
    progress_bar: Any = None,
) -> torch.Tensor:
    """Run the full denoising loop from start_timestep to 0.

    When ``prediction_target="x0"`` the model predicts clean ``x_0`` instead
    of velocity.  The velocity for the Euler step is recovered as
    ``v = (x_t - x_0_pred) / clamp(t, min=X0_T_CLAMP)`` to avoid numerical
    instability as ``t → 0``.

    Args:
        img: Initial latent tensor ``[T, H, W, C]`` (or wider for instruct).
        model: DiT model.
        device: Target CUDA device.
        num_steps: Number of denoising steps.
        text_embeds: Conditional text embeddings.
        null_text_embeds: Unconditional text embeddings.
        visual_cu_seqlens: Cumulative visual sequence lengths.
        text_cu_seqlens: Cumulative text sequence lengths.
        null_text_cu_seqlens: Cumulative null-text sequence lengths.
        visual_rope_pos: RoPE position indices for visual tokens.
        text_rope_pos: RoPE position indices for text tokens.
        null_text_rope_pos: RoPE position indices for null text tokens.
        scale_factor: Per-axis RoPE frequency scaling.
        guidance_weight: CFG strength.
        scheduler_scale: Timestep scheduler warp factor.
        first_frames: Optional I2V conditioning first frames.
        tp_mesh: Tensor-parallel device mesh (optional).
        visual_cond_scheme: ``"pretrain"`` or ``"i2v"`` injection strategy.
        start_timestep: Starting timestep (``1.0`` = full noise).
        prediction_target: ``"velocity"`` or ``"x0"``.
        channelcat_drop_threshold: When ``> 0``, zero out channel-cat LQ
            channels (and mask) once the denoising timestep drops below this
            value.  ``0`` = disabled (default).
        rfg_scale: Reference-Free Guidance strength on the anchor.  Only
            applied when ``model.instruct_type == "hybrid_anchor"`` and
            ``rfg_scale != 1.0`` — in that case a second forward is run with
            the anchor and anchor_mask channels zeroed, and predictions blend
            as ``v_uncond + rfg_scale * (v_cond - v_uncond)``.  ``1.0``
            (default) skips the second forward entirely.
    Returns:
        Denoised latent tensor (same leading dims as ``img``).
    """
    img = img.to(device)
    visual_cu_seqlens = visual_cu_seqlens.to(device)
    text_cu_seqlens = text_cu_seqlens.to(device)
    null_text_cu_seqlens = null_text_cu_seqlens.to(device)
    visual_rope_pos = [position.to(device) for position in visual_rope_pos]
    text_rope_pos = text_rope_pos.to(device)
    null_text_rope_pos = null_text_rope_pos.to(device)
    text_embeds = {key: value.to(device) for key, value in text_embeds.items()}
    null_text_embeds = {key: value.to(device) for key, value in null_text_embeds.items()}
    sparse_params = get_sparse_params(model, img, visual_cu_seqlens)
    timesteps = torch.linspace(start_timestep, 0, num_steps, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)
    if tp_mesh:
        tp_rank = tp_mesh["tp"].get_local_rank()
        tp_world_size = tp_mesh["tp"].size()
        img = torch.chunk(img, tp_world_size, dim=1)[tp_rank]
        if first_frames is not None:
            first_frames = torch.chunk(first_frames, tp_world_size, dim=1)[tp_rank]
    if model.visual_cond and first_frames is not None:
        first_frames = first_frames.to(device=img.device, dtype=img.dtype)
    out_channels: int = 0
    lq_cond_channels: torch.Tensor | None = None
    if channelcat_drop_threshold > 0 and img.shape[-1] > model.in_visual_dim:
        lq_cond_channels = img[..., model.in_visual_dim :].clone()
    for timestep, timestep_diff in list(zip(timesteps[:-1], torch.diff(timesteps), strict=False)):
        time = timestep.unsqueeze(0).repeat(visual_cu_seqlens.shape[0] - 1)
        if lq_cond_channels is not None:
            if timestep.item() >= channelcat_drop_threshold:
                img[..., model.in_visual_dim :] = lq_cond_channels
            else:
                img[..., model.in_visual_dim :] = 0
        if model.visual_cond and img.shape[-1] == model.in_visual_dim:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros([*img.shape[:-1], 1], dtype=img.dtype, device=img.device)
            if first_frames is not None:
                if visual_cond_scheme == "pretrain":
                    visual_cond[visual_cu_seqlens[:-1]] = first_frames
                elif visual_cond_scheme == "i2v":
                    img[visual_cu_seqlens[:-1]] = first_frames
                else:
                    msg = f"unknown visual_cond_scheme={visual_cond_scheme}"
                    raise ValueError(msg)
                visual_cond_mask[visual_cu_seqlens[:-1]] = 1
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        v_cond = get_model_prediction(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_cu_seqlens,
            text_cu_seqlens,
            null_text_cu_seqlens,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            scale_factor,
            guidance_weight,
            sparse_params=sparse_params,
        )
        if model.instruct_type == "hybrid_anchor" and rfg_scale != 1.0:
            model_input_uncond = model_input.clone()
            model_input_uncond[..., model.in_visual_dim :] = 0
            v_uncond = get_model_prediction(
                model,
                model_input_uncond,
                time,
                text_embeds,
                null_text_embeds,
                visual_cu_seqlens,
                text_cu_seqlens,
                null_text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                scale_factor,
                guidance_weight,
                sparse_params=sparse_params,
            )
            model_pred = v_uncond + rfg_scale * (v_cond - v_uncond)
        else:
            model_pred = v_cond
        out_channels = model_pred.shape[-1]
        if prediction_target == "x0":
            t_clamped = timestep.clamp(min=X0_T_CLAMP)
            velocity = (img[..., :out_channels] - model_pred) / t_clamped
        else:
            velocity = model_pred
        img[..., :out_channels] += timestep_diff * velocity
        if progress_callback is not None:
            progress_callback()
        if progress_bar is not None:
            progress_bar.update()
            progress_bar.refresh()
    if model.visual_cond and first_frames is not None and (visual_cond_scheme == "i2v"):
        img[visual_cu_seqlens[:-1]] = first_frames
    return img[..., :out_channels]


def _encode_lq_videos(lq_videos: list[torch.Tensor], vae: torch.nn.Module, device: str | int) -> torch.Tensor:
    """VAE-encode a list of LQ videos into a stacked latent tensor.

    Args:
        lq_videos: List of ``[T, H, W, 3]`` float32 tensors in ``[0, 255]``.
        vae: Pre-trained VAE model (eval mode).
        device: Target CUDA device.

    Returns:
        Stacked latent tensor ``[sum(T'), H', W', C]`` scaled by
        ``vae.config.scaling_factor``.
    """
    latents: list[torch.Tensor] = []
    for lq in lq_videos:
        lq_input = lq.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        lq_latent = encode_pixels_to_latent(vae, lq_input)
        lq_latent = lq_latent.squeeze(0).permute(1, 2, 3, 0).float()
        lq_latent *= vae.config.scaling_factor
        latents.append(lq_latent)
    return torch.cat(latents, dim=0)


def _build_initial_latent(
    dit: torch.nn.Module,
    lq_latent: torch.Tensor,
    bs: int,
    duration: int,
    height: int,
    width: int,
    device: str | int,
    seed: int,
    lq_noise_scale: float,
    lq_noise_type: Literal["linear", "ddpm"],
    lq_channel_noise_scale: float,
    anchor_latent: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    *,
    anchor_free: bool = False,
) -> torch.Tensor:
    """Build the initial latent tensor for the SR denoising loop.

    Constructs ``[starting | cond | mask]`` depending on ``instruct_type``:

    - ``"noise"``: degraded LQ as starting point, zero LQ cond + zero mask.
    - ``"channel"``: random Gaussian noise as starting point, LQ cond + ones mask.
    - ``"hybrid"``: degraded LQ as starting point, LQ cond + ones mask.
    - ``"hybrid_anchor"``: degraded LQ as starting point, sparse HR anchor +
      sparse anchor_mask.  Requires ``anchor_latent`` and ``anchor_mask``.

    Args:
        dit: DiT model (reads ``instruct_type`` and ``in_visual_dim``).
        lq_latent: Scaled LQ latent ``[bs*T, H, W, C]``.
        bs: Batch size.
        duration: Number of temporal frames per sample.
        height: Latent height.
        width: Latent width.
        device: Target CUDA device.
        seed: RNG seed for noise generation.
        lq_noise_scale: Noise fraction mixed into the LQ starting point.
        lq_noise_type: ``"linear"`` or ``"ddpm"`` noising formula.
        lq_channel_noise_scale: Noise fraction mixed into the channel-cat
            conditioning (LQ for ``channel``/``hybrid``, anchor for
            ``hybrid_anchor``).
        anchor_latent: Pre-scaled sparse HR anchor ``[bs*T, H, W, C]`` (zeros
            in slots without anchor).  Required when
            ``instruct_type == "hybrid_anchor"``.
        anchor_mask: Sparse binary mask ``[bs*T, H, W, 1]`` flagging anchor
            slots.  Required when ``instruct_type == "hybrid_anchor"``.

    Returns:
        Initial latent tensor ready for the denoising loop (bs*T, H, W, 33).

    Raises:
        ValueError: If ``instruct_type`` is not supported, or if
            ``hybrid_anchor`` is missing ``anchor_latent`` / ``anchor_mask``.
    """
    lq_latent = lq_latent.to(device)
    lq_latent = cast_to_module_dtype(dit, lq_latent)
    if anchor_latent is not None:
        anchor_latent = anchor_latent.to(device)
        anchor_latent = cast_to_module_dtype(dit, anchor_latent)
    if anchor_mask is not None:
        anchor_mask = anchor_mask.to(device)
        anchor_mask = cast_to_module_dtype(dit, anchor_mask)
    if dit.instruct_type == "noise":
        g_noise = torch.Generator(device=device)
        g_noise.manual_seed(seed)
        degraded_lq = degrade_lq_latent(lq_latent, lq_noise_scale, lq_noise_type, generator=g_noise)
        if not getattr(dit, "visual_cond", False):
            return degraded_lq
        zero_cond = torch.zeros_like(degraded_lq)
        zero_mask = torch.zeros([*degraded_lq.shape[:-1], 1], dtype=degraded_lq.dtype, device=device)
        return torch.cat([degraded_lq, zero_cond, zero_mask], dim=-1)
    if dit.instruct_type in ("channel", "hybrid"):
        if dit.instruct_type == "hybrid":
            g_noise = torch.Generator(device=device)
            g_noise.manual_seed(seed)
            starting = degrade_lq_latent(lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=g_noise)
        else:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            starting = torch.randn(bs * duration, height, width, dit.in_visual_dim, device=device, generator=g)
        channel_lq = degrade_lq_latent(lq_latent, lq_channel_noise_scale, noise_type="linear")
        mask = torch.ones_like(lq_latent[..., :1])
        return torch.cat([starting, channel_lq, mask], dim=-1)
    if dit.instruct_type == "hybrid_anchor":
        if anchor_latent is None or anchor_mask is None:
            if not anchor_free:
                msg = "hybrid_anchor requires anchor_latent and anchor_mask"
                raise ValueError(msg)
            anchor_latent = torch.zeros_like(lq_latent)
            anchor_mask = torch.zeros_like(lq_latent[..., :1])
        g_noise = torch.Generator(device=device)
        g_noise.manual_seed(seed)
        starting = degrade_lq_latent(lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=g_noise)
        anchor = degrade_lq_latent(anchor_latent, lq_channel_noise_scale, noise_type="linear")
        return torch.cat([starting, anchor, anchor_mask], dim=-1)
    msg = f"generate_sample_sr does not support instruct_type={dit.instruct_type!r}"
    raise ValueError(msg)


def _empty_text_embeds(bs: int, device: str | int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build placeholder text embeddings for a text-free DiT.

    A ``use_text=False`` DiT ignores every text input (encoder, cross-attention
    and pooled-text conditioning are all absent), so no real embeddings are
    needed. These zero-size tensors only satisfy the indexing and ``cu_seqlens``
    bookkeeping in ``generate_sample_sr`` / ``get_model_prediction`` without ever
    loading the cached empty-caption file.

    Args:
        bs: Batch size.
        device: Target CUDA device.

    Returns:
        Tuple of ``(text_embeds_dict, text_cu_seqlens)`` with empty sequences.
    """
    embeds = {"text_embeds": torch.zeros(0, 1, device=device), "pooled_embed": torch.zeros(bs, 1, device=device)}
    cu_seqlens = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    return (embeds, cu_seqlens)


def _encode_text(
    bs: int,
    device: str | int,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
    *,
    use_text: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Encode text or use cached embeddings for SR generation.

    For SR we always use empty captions. Returns both text and null-text
    embeddings (identical when using cached embeds).

    Args:
        bs: Batch size.
        device: Target CUDA device.
        text_embedder: Text encoder (Qwen2.5-VL + CLIP). Can be ``None``
            when ``cached_text_embeds`` is provided.
        cached_text_embeds: Pre-computed empty-caption embeddings (bs=1)
            loaded from disk.
        use_text: Whether the DiT consumes text. When ``False`` placeholder
            zero embeddings are returned and neither a text encoder nor cached
            embeddings are required.

    Returns:
        Tuple of ``(text_embed, text_cu_seqlens, null_text_embed,
        null_text_cu_seqlens)``.

    Raises:
        ValueError: If a text DiT is given neither ``text_embedder`` nor
            ``cached_text_embeds``.
    """
    if not use_text:
        embeds, cu_seqlens = _empty_text_embeds(bs, device)
        return (embeds, cu_seqlens, embeds, cu_seqlens)
    if cached_text_embeds is not None:
        bs_text_embed, text_cu_seqlens = replicate_cached_embeds(cached_text_embeds, bs, device)
        return (bs_text_embed, text_cu_seqlens, bs_text_embed, text_cu_seqlens)
    if text_embedder is not None:
        empty_captions = [""] * bs
        bs_text_embed, text_cu_seqlens = text_embedder.encode(empty_captions, images=None, type_of_content="video")
        bs_null_text_embed, null_text_cu_seqlens = text_embedder.encode(
            empty_captions, images=None, type_of_content="video"
        )
        bs_text_embed = {k: v.to(device) for k, v in bs_text_embed.items()}
        text_cu_seqlens = text_cu_seqlens.to(device)
        bs_null_text_embed = {k: v.to(device) for k, v in bs_null_text_embed.items()}
        null_text_cu_seqlens = null_text_cu_seqlens.to(device)
        return (bs_text_embed, text_cu_seqlens, bs_null_text_embed, null_text_cu_seqlens)
    msg = "generate_sample_sr requires either text_embedder or cached_text_embeds"
    raise ValueError(msg)


@torch.no_grad()
def generate_sample_sr(
    *,
    dit: torch.nn.Module,
    vae: torch.nn.Module,
    scale_factor: tuple[float, ...] = (1.0, 1.0, 1.0),
    num_steps: int = 50,
    guidance_weight: float = 5.0,
    scheduler_scale: float = 5.0,
    seed: int = 42,
    device: str | int = "cuda",
    tp_mesh: dict[str, Any] | None = None,
    lq_noise_scale: float = 0.0,
    lq_noise_type: Literal["linear", "ddpm"] = "linear",
    lq_channel_noise_scale: float = 0.0,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
    lq_videos: list[torch.Tensor] | None = None,
    lq_latents: torch.Tensor | None = None,
    n_samples: int | None = None,
    cap_noise_timestep: bool = False,
    vae_decode_batch: bool = False,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    anchor_latents: torch.Tensor | None = None,
    anchor_masks: torch.Tensor | None = None,
    anchor_free: bool = False,
    rfg_scale: float = 1.0,
) -> torch.Tensor:
    """Generate super-resolved video from LQ inputs (pixels or latents).

    Accepts either raw pixel videos (``lq_videos``) or pre-encoded latents
    (``lq_latents``).  When ``lq_latents`` is provided, VAE encoding is
    skipped and the latents are used directly.

    Builds the initial latent tensor, runs the denoising loop via
    ``generate()``, and decodes back to pixel space.

    Supports three SR instruct modes:

    - ``instruct_type="channel"``: random noise in HQ channels with LQ latent
        and mask concatenated as extra channels → ``[noise(C) | lq(C) | mask(1)]``.
    - ``instruct_type="noise"``: LQ latent is used directly as the starting
        point (replaces random noise) → ``[lq(C)]``, zero extra channels.
    - ``instruct_type="hybrid"``: LQ latent as starting point (like ``"noise"``)
        AND LQ latent concatenated as extra channels (like ``"channel"``) →
        ``[degraded_lq(C) | lq(C) | mask(1)]``.

    When ``lq_noise_scale > 0``, noise is mixed into the LQ latent used as
    the starting point (applies to ``"noise"`` and ``"hybrid"``).

    When ``lq_channel_noise_scale > 0``, noise is mixed into the LQ latent
    concatenated as extra channels (applies to ``"channel"`` and ``"hybrid"``).

    Args:
        dit: DiT model with ``instruct_type`` in
            ``{"channel", "noise", "hybrid"}``.
        vae: Pre-trained VAE model.
        scale_factor: Per-axis RoPE frequency scaling.
        num_steps: Number of diffusion denoising steps.
        guidance_weight: Classifier-free guidance strength.
        scheduler_scale: Timestep scheduler scaling factor.
        seed: RNG seed for noise initialization.
        device: Target CUDA device.
        tp_mesh: Tensor parallelism mesh (optional).
        lq_noise_scale: Noise fraction ``s`` mixed into LQ latent used as
            the noise/starting point (``0`` = disabled).  Applies to
            ``"noise"`` and ``"hybrid"`` modes.
        lq_noise_type: Noising formula — ``"linear"`` or ``"ddpm"``.
        lq_channel_noise_scale: Noise fraction mixed into LQ latent
            concatenated as extra channels (``0`` = disabled).  Applies to
            ``"channel"`` and ``"hybrid"`` modes.
        text_embedder: Text encoder (Qwen2.5-VL + CLIP). Can be ``None``
            when ``cached_text_embeds`` is provided.
        cached_text_embeds: Pre-computed empty-caption embeddings (bs=1)
            loaded from disk. When provided, ``text_embedder`` is not called.
        lq_videos: List of ``[T, H, W, 3]`` float32 tensors in ``[0, 255]``.
        lq_latents: Pre-encoded LQ latents (skips VAE encoding).
        n_samples: Required when ``lq_latents`` is provided.
        cap_noise_timestep: When ``True`` and ``instruct_type`` in
            ``("noise", "hybrid")``, starts the denoising loop from raw
            ``t = lq_noise_scale`` instead of ``t = 1.0``.  Matches the
            training-time behaviour when ``cap_noise_timestep`` is also set
            during training.  Default ``False`` preserves current behaviour.
        vae_decode_batch: When ``True``, decode all samples in a single VAE
            call instead of one-by-one. Faster but uses more GPU memory.
        prediction_target: ``"velocity"`` for velocity prediction (default),
            ``"x0"`` for clean-image prediction.
        channelcat_drop_threshold: When ``> 0``, zero out channel-cat LQ
            channels during denoising once ``t < threshold``.  ``0`` =
            disabled (default).
        anchor_latents: Pre-encoded sparse HR-anchor latents
            ``[bs*T', H', W', C]``, **already scaled** by VAE
            ``scaling_factor``.  Required for ``instruct_type="hybrid_anchor"``.
        anchor_masks: Sparse binary anchor mask ``[bs*T', H', W', 1]``.
            Required for ``instruct_type="hybrid_anchor"``.
        anchor_free: When ``True`` and ``instruct_type="hybrid_anchor"`` with
            no anchors provided, run anchor-free (zeroed anchor + mask) — the
            tiled path for GT-less real-world LQ videos.
        rfg_scale: Reference-Free Guidance strength on the anchor.  ``1.0``
            (default) runs a single cond forward per denoising step; values
            ``!= 1.0`` add a second uncond forward (anchor + anchor_mask
            zeroed) and blend ``v_uncond + rfg_scale * (v_cond - v_uncond)``.
            Only applies when ``instruct_type="hybrid_anchor"``.  Cost: +1
            DiT forward per denoising step when active.
    Returns:
        Generated SR videos as ``[bs, 3, T, H, W]`` uint8 tensor.

    Raises:
        ValueError: If neither ``text_embedder`` nor ``cached_text_embeds``
            is provided or if ``instruct_type`` is unsupported.
    """
    piflow_params = getattr(dit, "piflow_params", None)
    if piflow_params is not None:
        if rfg_scale != 1.0:
            msg = "π-Flow checkpoints do not support Reference-Free Guidance (drop rfg_scale)"
            raise ValueError(msg)
    return run_stages(
        dit=dit,
        vae=vae,
        scale_factor=scale_factor,
        num_steps=num_steps,
        guidance_weight=guidance_weight,
        scheduler_scale=scheduler_scale,
        seed=seed,
        device=device,
        tp_mesh=tp_mesh,
        lq_noise_scale=lq_noise_scale,
        lq_noise_type=lq_noise_type,
        lq_channel_noise_scale=lq_channel_noise_scale,
        text_embedder=text_embedder,
        cached_text_embeds=cached_text_embeds,
        lq_videos=lq_videos,
        lq_latents=lq_latents,
        n_samples=n_samples,
        cap_noise_timestep=cap_noise_timestep,
        vae_decode_batch=vae_decode_batch,
        prediction_target=prediction_target,
        channelcat_drop_threshold=channelcat_drop_threshold,
        anchor_latents=anchor_latents,
        anchor_masks=anchor_masks,
        anchor_free=anchor_free,
        rfg_scale=rfg_scale,
        piflow_params=piflow_params,
    )


@torch.no_grad()
def piflow_generate(
    img: torch.Tensor,
    model: torch.nn.Module,
    text_embeds: dict[str, torch.Tensor],
    visual_cu_seqlens: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    visual_rope_pos: list[torch.Tensor],
    text_rope_pos: torch.Tensor,
    scale_factor: tuple[float, ...],
    *,
    nfe: int,
    num_policy_substeps: int,
    final_step_size_scale: float,
    shift: float,
    n_grid: int,
    out_dim: int,
    eps: float = 1e-06,
    tp_mesh: dict[str, Any] | None = None,
    start_timestep: float = 1.0,
    device: str | int | None = None,
    progress_callback=None,
    scheduler: Any | None = None,
    progress_bar: Any = None,
) -> torch.Tensor:
    """Few-step π-Flow denoise: ``nfe`` network calls, network-free integration between.

    Walks the deterministic ``nfe``-segment schedule from ``raw_t = start_timestep``
    down to 0 (final segment scaled by ``final_step_size_scale``, matching ``sample_t``).
    Each segment: one ``model`` forward -> ``DXPolicy`` -> integrate over the segment.
    """
    device = device if device is not None else img.device
    img = img.to(device)
    visual_cu_seqlens = visual_cu_seqlens.to(device)
    text_cu_seqlens = text_cu_seqlens.to(device)
    visual_rope_pos = [position.to(device) for position in visual_rope_pos]
    text_rope_pos = text_rope_pos.to(device)
    text_embeds = {key: value.to(device) for key, value in text_embeds.items()}
    sparse_params = get_sparse_params(model, img, visual_cu_seqlens)
    if tp_mesh:
        tp_world_size = tp_mesh["tp"].size()
        tp_rank = tp_mesh["tp"].get_local_rank()
        img = torch.chunk(img, tp_world_size, dim=1)[tp_rank]
    ndim = img.dim()
    if nfe < 1:
        msg = f"piflow_generate requires nfe >= 1, got {nfe}"
        raise ValueError(msg)
    if scheduler is not None:
        scheduler.set_timesteps(nfe, device=device)
        if scheduler.timesteps.shape[0] != nfe:
            raise ValueError(f"Piflow scheduler returned {scheduler.timesteps.shape[0]} steps, expected {nfe}")
    else:
        final_step_size_scale = max(float(final_step_size_scale), eps)
        one_minus_final = 1.0 - final_step_size_scale
        base_seg = 1.0 / (float(nfe) - one_minus_final)
        final_seg = final_step_size_scale * base_seg
    x = img
    for step_index in range(nfe):
        if scheduler is not None:
            timestep = scheduler.timesteps[step_index]
            sigma_src = scheduler.sigmas[step_index].item()
        else:
            idx = nfe - step_index
            raw_src = min(max((idx - one_minus_final) * base_seg, eps), start_timestep)
            seg = final_seg if idx == 1 else base_seg
            raw_dst = max(raw_src - seg, 0.0)
            sigma_src = shift_timesteps(torch.tensor(raw_src, device=device), shift).item()
        n_objects = visual_cu_seqlens.shape[0] - 1
        t_src = torch.full(
            (n_objects,),
            float(timestep.item()) if scheduler is not None else sigma_src * 1000.0,
            device=device,
            dtype=x.dtype,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v0 = model(
                x,
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                t_src,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=scale_factor,
                sparse_params=sparse_params,
            )
        if scheduler is not None:
            x_pred = scheduler.step(v0, timestep, x[..., :out_dim], return_dict=False)[0]
        else:
            v0_grid = v0.unsqueeze(1) if n_grid == 1 else v0
            total_tokens = x.shape[0]
            sigma_tok = torch.full((total_tokens, *(ndim - 1) * [1]), sigma_src, device=device)
            seg_tok = torch.full((total_tokens,), seg, device=device)
            policy = DXPolicy(v0_grid, x[..., :out_dim], sigma_tok, seg_tok, shift=shift, mode="grid", eps=eps)
            raw_src_tok = torch.full((total_tokens,), raw_src, device=device)
            raw_dst_tok = torch.full((total_tokens,), raw_dst, device=device)
            x_pred, _, _ = policy_rollout_fm(
                x[..., :out_dim], sigma_tok, raw_src_tok, raw_dst_tok, num_policy_substeps, policy
            )
        x = torch.cat([x_pred, x[..., out_dim:]], dim=-1)
        if progress_callback is not None:
            progress_callback()
        if progress_bar is not None:
            progress_bar.update()
            progress_bar.refresh()
    return x[..., :out_dim]


class RunConfig(BaseModel):
    """Parameters for one tiled SR run."""

    model_config = ConfigDict(extra="forbid")

    device: str
    num_steps: int = Field(default=5, ge=2)
    seed: int = 42
    overlap: float = Field(default=0.25, ge=0.0, lt=1.0)
    tiles_batch_size: int = Field(default=1, gt=0)
    resolution_scale: ResolutionScale = 4
    tile_grid_mode: TileGridMode = "even"
    tile_grid_min_overlap: float = Field(default=0.20, ge=0.0, lt=1.0)


@dataclass
class SRParams:
    """SR sampling parameters extracted from the SR training config."""

    scale_factor: dict[int, list[float]]
    visual_size: list[int]
    scheduler_scale: float = 5.0
    lq_noise_scale: float = 0.7
    lq_noise_type: str = "ddpm"
    lq_channel_noise_scale: float = 0.0
    cap_noise_timestep: bool = False
    fps: int = 24


@dataclass
class SRComponents:
    """The weighted components and config consumed by the SR sampler."""

    dit: torch.nn.Module
    vae: torch.nn.Module
    latent_upscaler: torch.nn.Module | Any | None
    sr_params: SRParams
    cached_text_embeds: dict[str, torch.Tensor] | None = None
    sampler: Callable[..., torch.Tensor] | None = None
    spatial_factor: int | None = None


def _closest_base_resolution(h: int, w: int, visual_size: int) -> tuple[int, int]:
    if visual_size not in RESOLUTIONS:
        raise ValueError(f"Unsupported SR visual_size={visual_size}; known sizes: {sorted(RESOLUTIONS)}")
    ratio = w / h if h else 1.0
    return min(RESOLUTIONS[visual_size], key=lambda hw: abs(hw[1] / hw[0] - ratio))


def _tile_geometry(  # noqa: PLR0913
    h: int,
    w: int,
    visual_size: int,
    scale: int,
    overlap: float,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
    grid_mode: TileGridMode = "even",
    grid_min_overlap: float = 0.20,
) -> tuple[tuple[int, int], tuple[int, int], TileGrid]:
    base_h, base_w = _closest_base_resolution(h, w, visual_size)
    if base_h % scale or base_w % scale:
        raise ValueError(f"resolution_scale={scale} does not divide the base resolution {base_h}x{base_w} exactly")
    tile_hw = (base_h // scale, base_w // scale)
    if grid_mode == "even":
        grid = compute_tile_grid_even(h, w, tile_hw, grid_min_overlap, spatial_factor)
    else:
        grid = compute_tile_grid(h, w, scale, overlap, tile_hw=tile_hw)
    return (base_h, base_w), tile_hw, grid


def _spatial_factor(components: Any) -> int:
    explicit = getattr(components, "spatial_factor", None)
    if explicit is not None:
        return int(explicit)
    vae = getattr(components, "vae", None)
    config = getattr(vae, "config", None)
    return int(
        getattr(vae, "spatial_factor", None)
        or getattr(config, "spatial_factor", None)
        or getattr(_sr_constants, "VAE_SPATIAL_FACTOR", None)
        or VAE_SPATIAL_FACTOR
    )


def _component_params(components: Any) -> Any:
    params = getattr(components, "sr_params", None)
    if params is None:
        raise ValueError("SR components must provide sr_params")
    return params


def _visual_size(params: Any) -> int:
    visual_size = getattr(params, "visual_size", None)
    if isinstance(visual_size, int):
        return visual_size
    if not visual_size:
        raise ValueError("SR params must provide a non-empty visual_size")
    return int(visual_size[0])


def _scale_factor(params: Any, visual_size: int) -> tuple[float, ...]:
    values = getattr(params, "scale_factor", None)
    if isinstance(values, Mapping):
        try:
            values = values[visual_size]
        except KeyError:
            # YAML keeps numeric resolution keys as integers; JSON converts
            # object keys to strings when the config is stored in a bundle.
            values = values[str(visual_size)]
    return tuple(float(value) for value in values)


def _extract_latent(result: Any) -> torch.Tensor:
    if isinstance(result, tuple):
        result = result[0]
    distribution = getattr(result, "latent_dist", None)
    if distribution is not None:
        result = distribution.sample()
    if not isinstance(result, torch.Tensor):
        raise TypeError("VAE encode must return a Tensor, (Tensor, ...), or latent_dist output")
    return result


def _validate_equal_shapes(values: list[torch.Tensor], name: str) -> None:
    if not values:
        raise ValueError(f"{name} batch must not be empty")
    shape = tuple(values[0].shape)
    if any(tuple(value.shape) != shape for value in values[1:]):
        raise ValueError(f"all {name} batch items must have equal shape")


@torch.no_grad()
def encode_lq_videos_to_lr_latents(
    lq_videos: list[torch.Tensor],
    vae: torch.nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """Encode an equal-shaped batch of ``[T,C,H,W]`` videos in one VAE call."""
    _validate_equal_shapes(lq_videos, "video")
    pixel = torch.stack(lq_videos).permute(0, 2, 1, 3, 4).to(device=device)
    pixel = vae.normalize_data(pixel.float()) if hasattr(vae, "normalize_data") else pixel.float() / 127.5 - 1.0
    pixel = cast_to_module_dtype(vae, pixel)
    result = _extract_latent(vae.encode(pixel))
    if result.ndim != VIDEO_BATCH_RANK:
        raise ValueError(f"batched VAE encode must return rank 5, got shape {tuple(result.shape)}")
    return result.permute(0, 2, 1, 3, 4).float()


def latent_tile_grid_from_pixel_grid(pixel_grid: TileGrid, spatial_factor: int) -> TileGrid:
    """Convert an aligned pixel grid to the corresponding latent grid."""
    values = (pixel_grid.tile_h, pixel_grid.tile_w, *pixel_grid.tops, *pixel_grid.lefts)
    if any(value % spatial_factor for value in values):
        raise ValueError(
            f"Pixel tile grid (tile {pixel_grid.tile_h}x{pixel_grid.tile_w}, "
            f"tops={pixel_grid.tops}, lefts={pixel_grid.lefts}) not aligned to VAE spatial factor "
            f"{spatial_factor}."
        )
    return TileGrid(
        pixel_grid.tile_h // spatial_factor,
        pixel_grid.tile_w // spatial_factor,
        tuple(top // spatial_factor for top in pixel_grid.tops),
        tuple(left // spatial_factor for left in pixel_grid.lefts),
    )


@torch.no_grad()
def upscale_lr_latent_tile(
    lr_latent_tile: torch.Tensor,
    latent_upscaler: torch.nn.Module,
    vae: torch.nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """Upscale one raw latent tile into the sampler's channels-last layout."""
    scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
    tile = lr_latent_tile.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.float32)
    tile = cast_to_module_dtype(latent_upscaler, tile)
    if str(device).startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            upscaled = run_latent_upscaler(latent_upscaler, tile * scaling_factor)
    else:
        upscaled = run_latent_upscaler(latent_upscaler, tile * scaling_factor)
    return upscaled.squeeze(0).permute(1, 2, 3, 0).float()


def _upsample_tiles_to_base(raw_tiles: list[torch.Tensor], base_h: int, base_w: int) -> list[torch.Tensor]:
    return [
        functional.interpolate(tile.float(), size=(base_h, base_w), mode="bilinear", align_corners=False).permute(
            0, 2, 3, 1
        )
        for tile in raw_tiles
    ]


# Reusable Kandinsky SR inference stages and tiled orchestration.
#
# The functions in this module deliberately do not depend on a benchmark
# runner. A framework can call ``text_encode`` / ``prepare_latents`` /
# ``denoise`` / ``vae_decode`` directly, or use the tiled functions for the
# scale-aware path. Native benchmark integrations may add timing contexts to
# the tiled calls; the public :class:`Kandinsky6SRPipeline` does not expose
# those callbacks.


@dataclass(frozen=True)
class SRLatentState:
    """Inputs and initial noisy latent for one SR DiT invocation."""

    lq_latent: torch.Tensor
    image: torch.Tensor
    batch_size: int
    duration: int
    height: int
    width: int


@dataclass(frozen=True)
class SRTextState:
    """Conditional and null text state consumed by SR denoising."""

    text_embeds: dict[str, torch.Tensor]
    text_cu_seqlens: torch.Tensor
    null_text_embeds: dict[str, torch.Tensor]
    null_text_cu_seqlens: torch.Tensor


@torch.no_grad()
def prepare_latents(
    *,
    dit: torch.nn.Module,
    vae: torch.nn.Module,
    device: str | int,
    lq_videos: list[torch.Tensor] | None = None,
    lq_latents: torch.Tensor | None = None,
    n_samples: int | None = None,
    seed: int = 42,
    lq_noise_scale: float = 0.0,
    lq_noise_type: str = "linear",
    lq_channel_noise_scale: float = 0.0,
    anchor_latents: torch.Tensor | None = None,
    anchor_masks: torch.Tensor | None = None,
    anchor_free: bool = False,
) -> SRLatentState:
    """Encode LQ input and build the initial latent for one SR tile batch."""
    if lq_latents is not None:
        lq_latent = lq_latents.to(device)
        if n_samples is None:
            raise ValueError("n_samples is required when lq_latents is provided")
        batch_size = n_samples
    elif lq_videos is not None:
        batch_size = len(lq_videos)
        lq_latent = _encode_lq_videos(lq_videos, vae, device)
    else:
        raise ValueError("Either lq_videos or lq_latents must be provided")
    duration = lq_latent.shape[0] // batch_size
    height, width = (lq_latent.shape[1], lq_latent.shape[2])
    image = _build_initial_latent(
        dit=dit,
        lq_latent=lq_latent,
        bs=batch_size,
        duration=duration,
        height=height,
        width=width,
        device=device,
        seed=seed,
        lq_noise_scale=lq_noise_scale,
        lq_noise_type=lq_noise_type,
        lq_channel_noise_scale=lq_channel_noise_scale,
        anchor_latent=anchor_latents.to(device) if anchor_latents is not None else None,
        anchor_mask=anchor_masks.to(device) if anchor_masks is not None else None,
        anchor_free=anchor_free,
    )
    return SRLatentState(lq_latent, image, batch_size, duration, height, width)


@torch.no_grad()
def text_encode(
    *,
    dit: torch.nn.Module,
    batch_size: int,
    device: str | int,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
) -> SRTextState:
    """Build SR conditional and null text embeddings."""
    text_embeds, text_cu_seqlens, null_text_embeds, null_text_cu_seqlens = _encode_text(
        bs=batch_size,
        device=device,
        text_embedder=text_embedder,
        cached_text_embeds=cached_text_embeds,
        use_text=getattr(dit, "use_text", True),
    )
    return SRTextState(text_embeds, text_cu_seqlens, null_text_embeds, null_text_cu_seqlens)


def _positions(
    *, dit: torch.nn.Module, latent_state: SRLatentState, text_state: SRTextState
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    runtime_device = latent_state.image.device
    visual_cu_seqlens = latent_state.duration * torch.arange(
        latent_state.batch_size + 1, dtype=torch.int32, device=runtime_device
    )
    visual_rope_pos = [
        torch.cat([torch.arange(int(end), device=runtime_device) for end in torch.diff(visual_cu_seqlens).cpu()]),
        torch.arange(latent_state.height // dit.patch_size[1], device=runtime_device),
        torch.arange(latent_state.width // dit.patch_size[2], device=runtime_device),
    ]
    text_cu_seqlens = text_state.text_cu_seqlens.to(runtime_device)
    null_text_cu_seqlens = text_state.null_text_cu_seqlens.to(runtime_device)
    text_rope_pos = torch.cat(
        [torch.arange(int(end), device=runtime_device) for end in torch.diff(text_cu_seqlens).cpu()]
    )
    null_text_rope_pos = torch.cat(
        [torch.arange(int(end), device=runtime_device) for end in torch.diff(null_text_cu_seqlens).cpu()]
    )
    return (visual_cu_seqlens, visual_rope_pos, text_rope_pos, null_text_rope_pos)


@torch.no_grad()
def denoise(
    *,
    dit: torch.nn.Module,
    latent_state: SRLatentState,
    text_state: SRTextState,
    scale_factor: tuple[float, ...],
    num_steps: int = 50,
    guidance_weight: float = 5.0,
    scheduler_scale: float = 5.0,
    tp_mesh: dict[str, Any] | None = None,
    lq_noise_scale: float = 0.0,
    cap_noise_timestep: bool = False,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    rfg_scale: float = 1.0,
    piflow_params: dict[str, Any] | None = None,
    device: str | int | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
    progress_bar: Any = None,
) -> torch.Tensor:
    """Run either the Euler or π-Flow SR denoising stage."""
    piflow_params = piflow_params or _scheduler_piflow_params(scheduler)
    piflow_params = piflow_params or getattr(dit, "piflow_params", None)
    if piflow_params is not None:
        if rfg_scale != 1.0:
            raise ValueError("π-Flow checkpoints do not support Reference-Free Guidance")
        if cap_noise_timestep and dit.instruct_type in ("noise", "hybrid"):
            raise NotImplementedError("piflow sampler does not support cap_noise_timestep for noise/hybrid instruct")
    device = device if device is not None else latent_state.image.device
    visual_cu_seqlens, visual_rope_pos, text_rope_pos, null_text_rope_pos = _positions(
        dit=dit, latent_state=latent_state, text_state=text_state
    )
    if piflow_params is not None:
        out_dim = int(getattr(dit, "base_out_visual_dim", dit.in_visual_dim))
        start_t = 1.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = piflow_generate(
                latent_state.image,
                dit,
                text_state.text_embeds,
                visual_cu_seqlens,
                text_state.text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                scale_factor,
                **piflow_params,
                out_dim=out_dim,
                start_timestep=start_t,
                device=device,
                progress_callback=progress_callback,
                scheduler=scheduler,
                progress_bar=progress_bar,
            )
    else:
        start_t = (
            lq_noise_scale if cap_noise_timestep and dit.instruct_type in ("noise", "hybrid", "hybrid_anchor") else 1.0
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                latent_state.image,
                dit,
                device,
                num_steps,
                text_state.text_embeds,
                text_state.null_text_embeds,
                visual_cu_seqlens,
                text_state.text_cu_seqlens,
                text_state.null_text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                scale_factor,
                guidance_weight,
                scheduler_scale,
                tp_mesh=tp_mesh,
                first_frames=None,
                start_timestep=start_t,
                prediction_target=prediction_target,
                channelcat_drop_threshold=channelcat_drop_threshold,
                rfg_scale=rfg_scale,
                progress_callback=progress_callback,
                progress_bar=progress_bar,
            )
    if tp_mesh:
        tensor_list = [
            torch.zeros_like(latent_visual, device=latent_visual.device) for _ in range(tp_mesh["tp"].size())
        ]
        all_gather(tensor_list, latent_visual.contiguous(), group=tp_mesh.get_group(mesh_dim="tp"))
        latent_visual = torch.cat(tensor_list, dim=1)
    return latent_visual


def _scheduler_piflow_params(scheduler: Any | None) -> dict[str, Any] | None:
    if scheduler is None or not bool(getattr(scheduler, "is_piflow", False)):
        return None
    config = scheduler.config
    nfe = getattr(config, "nfe", None)
    if nfe is None:
        raise ValueError("PiflowScheduler used for SR must define nfe in scheduler_config.json")
    return {
        "nfe": int(nfe),
        "num_policy_substeps": int(config.num_policy_substeps),
        "final_step_size_scale": float(config.final_step_size_scale),
        "shift": float(config.shift),
        "n_grid": int(config.n_grid),
        "eps": float(config.eps),
    }


@torch.no_grad()
def vae_decode(
    *,
    vae: torch.nn.Module,
    latent_visual: torch.Tensor,
    batch_size: int,
    duration: int,
    height: int,
    width: int,
    vae_decode_batch: bool = False,
) -> torch.Tensor:
    """Decode denoised SR latents into ``[batch, 3, T, H, W]`` uint8 frames"""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        all_latents = latent_visual.reshape(batch_size, duration, height, width, -1)
        all_latents = (all_latents / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
        if vae_decode_batch:
            return decode_latent_to_uint8(vae, all_latents)
        decoded = [decode_latent_to_uint8(vae, all_latents[i : i + 1]) for i in range(batch_size)]
        return torch.cat(decoded, dim=0)


@torch.no_grad()
def run_stages(
    *,
    dit: torch.nn.Module,
    vae: torch.nn.Module,
    scale_factor: tuple[float, ...],
    num_steps: int = 50,
    guidance_weight: float = 5.0,
    scheduler_scale: float = 5.0,
    seed: int = 42,
    device: str | int = "cuda",
    tp_mesh: dict[str, Any] | None = None,
    lq_noise_scale: float = 0.0,
    lq_noise_type: str = "linear",
    lq_channel_noise_scale: float = 0.0,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
    lq_videos: list[torch.Tensor] | None = None,
    lq_latents: torch.Tensor | None = None,
    n_samples: int | None = None,
    cap_noise_timestep: bool = False,
    vae_decode_batch: bool = False,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    anchor_latents: torch.Tensor | None = None,
    anchor_masks: torch.Tensor | None = None,
    anchor_free: bool = False,
    rfg_scale: float = 1.0,
    piflow_params: dict[str, Any] | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
    progress_bar: Any = None,
) -> torch.Tensor:
    """Run the four reusable SR stages for one tile batch."""
    piflow_params = piflow_params or _scheduler_piflow_params(scheduler)
    piflow_params = piflow_params or getattr(dit, "piflow_params", None)
    latent_state = prepare_latents(
        dit=dit,
        vae=vae,
        device=device,
        lq_videos=lq_videos,
        lq_latents=lq_latents,
        n_samples=n_samples,
        seed=seed,
        lq_noise_scale=lq_noise_scale,
        lq_noise_type=lq_noise_type,
        lq_channel_noise_scale=lq_channel_noise_scale,
        anchor_latents=anchor_latents,
        anchor_masks=anchor_masks,
        anchor_free=anchor_free,
    )
    text_state = text_encode(
        dit=dit,
        batch_size=latent_state.batch_size,
        device=device,
        text_embedder=text_embedder,
        cached_text_embeds=cached_text_embeds,
    )
    latent_visual = denoise(
        dit=dit,
        latent_state=latent_state,
        text_state=text_state,
        scale_factor=scale_factor,
        num_steps=num_steps,
        guidance_weight=guidance_weight,
        scheduler_scale=scheduler_scale,
        device=device,
        tp_mesh=tp_mesh,
        lq_noise_scale=lq_noise_scale,
        cap_noise_timestep=cap_noise_timestep,
        prediction_target=prediction_target,
        channelcat_drop_threshold=channelcat_drop_threshold,
        rfg_scale=rfg_scale,
        piflow_params=piflow_params,
        progress_callback=progress_callback,
        scheduler=scheduler,
        progress_bar=progress_bar,
    )
    return vae_decode(
        vae=vae,
        latent_visual=latent_visual,
        batch_size=latent_state.batch_size,
        duration=latent_state.duration,
        height=latent_state.height,
        width=latent_state.width,
        vae_decode_batch=vae_decode_batch,
    )


def _generate_kwargs(
    components: Any, run_config: Any, params: Any, scale_factor: tuple[float, ...], **inputs: Any
) -> dict[str, Any]:
    return {
        **inputs,
        "dit": components.dit,
        "vae": components.vae,
        "scale_factor": scale_factor,
        "num_steps": run_config.num_steps,
        "anchor_free": True,
        "guidance_weight": 1.0,
        "scheduler_scale": float(getattr(params, "scheduler_scale", 5.0)),
        "seed": run_config.seed,
        "device": run_config.device,
        "tp_mesh": None,
        "lq_noise_scale": float(getattr(params, "lq_noise_scale", 0.7)),
        "lq_noise_type": getattr(params, "lq_noise_type", "ddpm"),
        "lq_channel_noise_scale": float(getattr(params, "lq_channel_noise_scale", 0.0)),
        "cap_noise_timestep": bool(getattr(params, "cap_noise_timestep", False)),
        "cached_text_embeds": getattr(components, "cached_text_embeds", None),
    }


def _as_tiles(sr_batch: torch.Tensor) -> list[torch.Tensor]:
    if sr_batch.ndim != 5:
        raise ValueError(f"SR sampler must return [batch,C,T,H,W], got {tuple(sr_batch.shape)}")
    return [sample.float().cpu() for sample in sr_batch]


def _scale_progress_callback(progress_callback: Callable[[int], Any], chunk_size: int) -> Callable[[int], Any]:

    def update(n: int = 1) -> None:
        progress_callback(n * chunk_size)

    return update


def _loaded_lu_scales(components: Any) -> tuple[int, ...]:
    lu = getattr(components, "latent_upscaler", None)
    if lu is None:
        return ()
    scales = getattr(lu, "scales", None)
    return tuple((int(scale) for scale in scales)) if scales is not None else (latent_upscaler_scale(lu),)


def _run_tile_batches(
    tile_inputs: list[torch.Tensor],
    sr_components: Any,
    run_config: Any,
    scale_factor: tuple[float, ...],
    *,
    prepare_chunk: Callable[[list[torch.Tensor]], list[torch.Tensor]] | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
    sample_batch_size: int = 1,
    progress_bar: Any = None,
) -> list[torch.Tensor]:
    """Run the shared batched sampler loop for latent or pixel tile inputs.

    DiT and SR VAE stay resident for the complete tile loop. Only the optional
    latent-upscaler is streamed per chunk.
    """
    outputs: list[torch.Tensor] = []
    sampler = getattr(sr_components, "sampler", None)
    params = _component_params(sr_components)
    piflow_params = _scheduler_piflow_params(scheduler) or getattr(sr_components.dit, "piflow_params", None)
    progress_steps = int(piflow_params["nfe"] if piflow_params is not None else run_config.num_steps - 1)
    for start in range(0, len(tile_inputs), run_config.tiles_batch_size):
        raw_chunk = tile_inputs[start : start + run_config.tiles_batch_size]
        chunk = raw_chunk
        if prepare_chunk is not None:
            chunk = prepare_chunk(raw_chunk)
        input_kwargs: dict[str, Any]
        if prepare_chunk is None:
            input_kwargs = {"lq_videos": chunk}
        else:
            input_kwargs = {"lq_latents": torch.cat(chunk, dim=0), "n_samples": len(raw_chunk)}
        kwargs = _generate_kwargs(sr_components, run_config, params, scale_factor, **input_kwargs)
        kwargs["seed"] = run_config.seed + start // sample_batch_size
        chunk_progress = (
            _scale_progress_callback(progress_callback, len(raw_chunk)) if progress_callback is not None else None
        )
        if sampler is None:
            outputs.extend(
                _as_tiles(
                    run_stages(
                        **kwargs,
                        piflow_params=piflow_params,
                        progress_callback=chunk_progress,
                        scheduler=scheduler,
                        progress_bar=progress_bar,
                    )
                )
            )
        else:
            outputs.extend(_as_tiles(sampler(**kwargs)))
            if chunk_progress is not None:
                chunk_progress(progress_steps)
    return outputs


def align_to_vae_stride(t: int) -> int:
    """Round ``t`` to the nearest valid pixel-frame count ``1 + 8·k`` (ties up).

    SR data uses ``1 + 8·k`` frames so the temporal VAE stride round-trips
    cleanly. Rounding to nearest (not truncation) keeps a requested duration
    close to the target.

    Args:
        t: Raw frame count.

    Returns:
        Nearest valid frame count ``>= 1``.
    """
    if t <= 1:
        return 1
    k_floor = (t - 1) // 8
    t_floor = 1 + 8 * k_floor
    t_ceil = 1 + 8 * (k_floor + 1)
    return t_ceil if (t - t_floor) >= (t_ceil - t) else t_floor


def select_frame_indices(total_frames: int, src_fps: float, target_fps: float) -> list[int]:
    """Fixed-stride frame indices that downsample ``src_fps`` to ``target_fps``.

    Mirrors ``data_encoding.frame_utils.select_frame_indices``: with
    ``step = src_fps / target_fps >= 1``, ``round(i * step)`` is strictly
    non-decreasing, so the selection never duplicates a source frame.

    Args:
        total_frames: Number of frames in the source.
        src_fps: Source frame rate (must be ``>= target_fps``).
        target_fps: Desired frame rate.

    Returns:
        Sorted list of integer source-frame indices.
    """
    step = src_fps / target_fps
    indices = [round(i * step) for i in range(int(total_frames / step))]
    return [i for i in indices if i < total_frames]


def resample_to_target_fps(
    video: torch.Tensor,
    src_fps: float,
    target_fps: int = TARGET_FPS,
) -> tuple[torch.Tensor, int]:
    """Resample a decoded video toward ``target_fps`` (downsample / no-op / keep).

    Three tiers, mirroring the dataset encoding pipeline:

    - ``|src_fps - target_fps| < RESAMPLE_FPS_TOLERANCE``: pass through unchanged.
    - ``src_fps > target_fps``: fixed-stride downsample via
      :func:`select_frame_indices`; the result is at ``target_fps``.
    - ``src_fps < target_fps``: kept at the native rate with a warning (no ffmpeg
      ``minterpolate`` upsample); the clip stays mildly out of distribution.

    Args:
        video: ``[T, C, H, W]`` decoded source video.
        src_fps: Source frame rate.
        target_fps: Training frame rate to resample toward.

    Returns:
        ``(resampled_video, effective_fps)`` where ``effective_fps`` is the rate
        the returned frames play at (``target_fps`` when downsampled, otherwise
        ``round(src_fps)``) and is the correct rate to save the SR result at.
    """
    if abs(src_fps - target_fps) < RESAMPLE_FPS_TOLERANCE:
        return video, round(src_fps)
    if src_fps > target_fps:
        indices = select_frame_indices(video.shape[0], src_fps, target_fps)
        logger.warning(
            "Source fps {:.2f} > target {}fps: downsampling {} frames -> {} (fixed-stride); "
            "source temporal detail beyond {}fps is discarded.",
            src_fps,
            target_fps,
            video.shape[0],
            len(indices),
            target_fps,
        )
        return video[indices], target_fps
    logger.warning(
        "Source fps {:.2f} < target {}fps: keeping native frames (no minterpolate upsample); "
        "output is mildly out of distribution.",
        src_fps,
        target_fps,
    )
    return video, round(src_fps)


def clip_to_aligned_frames(video: torch.Tensor, max_num_frames: int = MAX_NUM_FRAMES) -> torch.Tensor:
    """Take the first ``max_num_frames`` and floor-align to ``1 + 8k`` frames.

    Args:
        video: ``[T, C, H, W]`` video.
        max_num_frames: Hard cap applied before alignment (``<= 0`` disables it).

    Returns:
        ``[T', C, H, W]`` with ``T' == 1 + 8k`` and ``T' <= max_num_frames``.

    Raises:
        ValueError: If the video has no frames.
    """
    if max_num_frames > 0:
        video = video[:max_num_frames]
    aligned = 1 + 8 * ((video.shape[0] - 1) // 8) if video.shape[0] > 0 else 0
    if aligned == 0:
        msg = "Video has no readable frames."
        raise ValueError(msg)
    return video[:aligned]


def read_video_tchw_uint8(path: Path) -> tuple[torch.Tensor, float]:
    """Decode an mp4/mkv to ``([T, C, H, W] uint8, src_fps)`` (no resample/cap).

    Resampling to the training fps and frame-count alignment are applied by the
    caller via :func:`resample_to_target_fps` and :func:`clip_to_aligned_frames`.

    Args:
        path: Source video file.

    Returns:
        ``([T, C, H, W] uint8 video, native_fps)``.

    Raises:
        ValueError: If the file has no readable frames or reports no fps.
    """
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate or stream.base_rate or 0.0)
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1) for frame in container.decode(stream)
        ]

    if not frames:
        msg = f"Video {path.name} has no readable frames."
        raise ValueError(msg)
    if src_fps <= 0:
        msg = f"Video {path.name} reports no usable fps (got {src_fps})."
        raise ValueError(msg)
    return torch.stack(frames).contiguous(), src_fps


def load_lr_latent(path: Path) -> torch.Tensor:
    """Load a raw, unscaled LR latent ``.pt`` as ``[T, C, H, W]`` float32.

    The tensor must match what ``encode_lq_video_to_lr_latent`` produces for
    the model's VAE (no scaling factor applied) — the LU multiplies internally.

    Args:
        path: Path to a ``.pt`` file holding a 4D latent tensor.

    Returns:
        ``[T, C, H, W]`` float32 latent on CPU.

    Raises:
        ValueError: If the loaded object is not a rank-4 float tensor.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, torch.Tensor) or obj.ndim != LATENT_NDIM:
        shape = getattr(obj, "shape", type(obj).__name__)
        msg = f"Expected a rank-4 [T, C, H, W] latent tensor in {path}, got {shape}."
        raise ValueError(msg)
    logger.info("Loaded LR latent {} from {}", tuple(obj.shape), path)
    return obj.float()


def _video_np(frames: Tensor) -> np.ndarray:
    """(3, T, H, W) uint8 → (T, H, W, 3) uint8."""
    return frames.detach().permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8, copy=False)


def _audio_np(audio: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """int16 / float waveform → float32 mono in [-1, 1]."""
    audio_np = np.asarray(audio[0] if isinstance(audio, list) else audio)
    if np.issubdtype(audio_np.dtype, np.integer):
        audio_np = audio_np.astype(np.float32) / np.iinfo(audio_np.dtype).max
    return np.clip(audio_np, -1.0, 1.0).astype(np.float32, copy=False)


def extract_audio_from_video(
    source_video: str | Path,
    audio_sample_rate: int = 44100,
) -> np.ndarray | None:
    """Decode a source video's audio as mono float32 samples.

    SR changes only the video stream.  Decoding the source audio here lets the
    output mux keep that audio without requiring the SR caller to load the
    entire source container itself.  Resampling also gives the AAC writer a
    stable sample rate for videos whose source audio uses another rate.
    """
    if audio_sample_rate <= 0:
        raise ValueError("audio_sample_rate must be positive")

    chunks: list[np.ndarray] = []
    with av.open(str(source_video), mode="r") as container:
        audio_streams = list(container.streams.audio)
        if not audio_streams:
            return None

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=audio_sample_rate)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32, copy=False)


def mux_video_audio(  # noqa: PLR0913
    frames: Tensor,
    audio: list[np.ndarray] | np.ndarray | None,
    output_path: str | Path,
    fps: int = 24,
    audio_sample_rate: int = 44100,
    video_crf: int = 18,
    source_video: str | Path | None = None,
    lossless: bool = False,
) -> Path:
    """Mux video (+ optional audio) into ``output_path`` via PyAV — no temp files.

    frames: (3, T, H, W) uint8 — single video, channels-first.
    audio:  list of (samples,) int16, a single ndarray, or None for video-only.
    source_video: optional source container from which audio is copied when
        ``audio`` is not supplied.  This is useful when SR receives frames
        extracted from a generated T2VA video.
    lossless: encode video as FFV1 instead of libx264. Audio remains AAC.
    """
    if audio is not None and source_video is not None:
        raise ValueError("pass either audio or source_video, not both")
    if source_video is not None:
        audio = extract_audio_from_video(source_video, audio_sample_rate)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_np = _video_np(frames)

    with av.open(str(output_path), mode="w") as container:
        video_stream = container.add_stream("ffv1" if lossless else "libx264", rate=fps)
        video_stream.width = video_np.shape[2]
        video_stream.height = video_np.shape[1]
        if lossless:
            video_stream.pix_fmt = "bgra"
            video_stream.options = {"level": "3"}
        else:
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": str(video_crf)}

        audio_stream = None
        if audio is not None:
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "mono"
            audio_stream.bit_rate = 192_000

        for i, frame_np in enumerate(video_np):
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in video_stream.encode(frame):
                container.mux(packet)

        if audio_stream is not None:
            audio_np = _audio_np(audio)
            audio_frame = av.AudioFrame.from_ndarray(audio_np[np.newaxis, :], format="fltp", layout="mono")
            audio_frame.sample_rate = audio_sample_rate
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, audio_sample_rate)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            for packet in audio_stream.encode():
                container.mux(packet)

    return output_path


def latent_upscaler_for_scale(upscaler: Any, scale: int) -> nn.Module | None:
    """Keep bank dispatch on the registered parent for Diffusers offload."""
    if upscaler is None:
        return None
    if hasattr(upscaler, "_models"):
        key = f"{int(scale)}x"
        if key not in upscaler._models:
            return None
        # The native helper returns a child module.  A Diffusers CPU-offload
        # hook is attached to the registered bank, so call its forward method
        # with the selected entry instead.
        upscaler.target_scale = key
        return upscaler
    for_scale = getattr(upscaler, "for_scale", None)
    if callable(for_scale):
        return for_scale(scale)
    return upscaler if str(getattr(upscaler, "target_scale", "4x")) == f"{int(scale)}x" else None


SRInput = Tensor | np.ndarray | Sequence[Any] | str | Path
SRBatchInput = SRInput | Sequence[SRInput]
SRScale = Literal[2, 4, 2.25]
SROutputType = Literal["pt", "torch", "np", "numpy"]
VIDEO_RANK = 4
VIDEO_BATCH_RANK = 5


def _is_video_batch(value: Any) -> bool:
    if isinstance(value, (Tensor, np.ndarray)):
        return value.ndim == VIDEO_BATCH_RANK
    if isinstance(value, (str, Path)) or not isinstance(value, Sequence) or not value:
        return False
    first = value[0]
    if isinstance(first, (str, Path)):
        return True
    if isinstance(first, (Tensor, np.ndarray)):
        return first.ndim >= VIDEO_RANK
    return False


def _video_batch_tensor(value: Tensor | np.ndarray) -> list[Tensor]:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim != VIDEO_BATCH_RANK:
        raise ValueError(f"batched video must have rank 5, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3, 4):
        return [sample.permute(0, 3, 1, 2) for sample in tensor]
    if tensor.shape[1] in (1, 3, 4):
        return [sample.permute(1, 0, 2, 3) for sample in tensor]
    if tensor.shape[2] in (1, 3, 4):
        return [sample for sample in tensor]
    raise ValueError("5D video must be [B,C,T,H,W], [B,T,C,H,W], or [B,T,H,W,C]")


def _latent_tensor(value: Tensor | np.ndarray, channels: int) -> Tensor:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim != VIDEO_RANK:
        raise ValueError(f"latent sample must have rank 4, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] == channels:
        return tensor
    if tensor.shape[-1] == channels:
        return tensor.permute(0, 3, 1, 2)
    if tensor.shape[0] == channels:
        return tensor.permute(1, 0, 2, 3)
    raise ValueError("4D latent must be [T,C,H,W], [T,H,W,C], or [C,T,H,W]")


def _latent_batch(value: Tensor | np.ndarray | str | Path | Sequence[Any], channels: int) -> list[Tensor]:
    if isinstance(value, (str, Path)):
        return [_latent_tensor(load_lr_latent(Path(value)), channels)]
    if isinstance(value, (Tensor, np.ndarray)):
        tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.ndim == VIDEO_RANK:
            return [_latent_tensor(tensor, channels)]
        if tensor.ndim == VIDEO_BATCH_RANK:
            if tensor.shape[-1] == channels:
                return [_latent_tensor(sample, channels) for sample in tensor]
            if tensor.shape[1] == channels:
                return [_latent_tensor(sample, channels) for sample in tensor]
            if tensor.shape[2] == channels:
                return [_latent_tensor(sample, channels) for sample in tensor]
            raise ValueError("5D latent must be [B,C,T,H,W], [B,T,C,H,W], or [B,T,H,W,C]")
        raise ValueError(f"latents must have rank 4 or 5, got shape {tuple(tensor.shape)}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("latents batch must not be empty")
        result: list[Tensor] = []
        for item in value:
            result.extend(_latent_batch(item, channels))
        return result
    raise TypeError("latents must be tensors, arrays, paths, or a batch of them")


def _validate_batch_shapes(values: list[Tensor], name: str) -> None:
    if not values:
        raise ValueError(f"{name} batch must not be empty")
    shape = tuple(values[0].shape)
    if any(tuple(value.shape) != shape for value in values[1:]):
        raise ValueError(
            f"all {name} batch items must have equal shape, got {[tuple(value.shape) for value in values]}"
        )


def _batch_output_values(value: Any, batch_size: int, name: str) -> list[Any]:
    if isinstance(value, list):
        if len(value) != batch_size:
            raise ValueError(f"{name} must contain one value per input ({batch_size}), got {len(value)}")
        return value
    if name == "save_path" and value is not None and batch_size > 1:
        raise ValueError("save_path must be a list when processing multiple SR inputs")
    return [value] * batch_size


def _stitch_batch(
    outputs: list[Tensor],
    grid: Any,
    batch_size: int,
    height: int,
    width: int,
    scale: float,
) -> Tensor:
    """Stitch tile-major output back into ``[B,C,T,H,W]`` samples."""
    tile_count = grid.total_tiles
    expected = batch_size * tile_count
    if len(outputs) != expected:
        raise RuntimeError(f"expected {expected} SR tile outputs, got {len(outputs)}")
    return torch.stack(
        [
            stitch_tiles_hanning(
                [outputs[tile_index * batch_size + sample_index] for tile_index in range(tile_count)],
                grid,
                height,
                width,
                scale=scale,
            )
            .clamp(0, 255)
            .to(torch.uint8)
            for sample_index in range(batch_size)
        ]
    )


def _run_batched_tiles(  # noqa: PLR0913
    samples: list[Tensor],
    components: Any,
    run_config: Any,
    *,
    latent_input: bool,
    progress_bar: Any,
    scheduler: Any,
) -> Tensor:
    """Run equal-shape samples tile-major and restore the sample batch."""
    params = _component_params(components)
    visual_size = int(params.visual_size[0]) if isinstance(params.visual_size, list) else int(params.visual_size)
    batch_size = len(samples)
    if latent_input:
        spatial_factor = _spatial_factor(components)
        height = samples[0].shape[-2] * spatial_factor
        width = samples[0].shape[-1] * spatial_factor
        _, _, pixel_grid = _tile_geometry(
            height,
            width,
            visual_size,
            run_config.resolution_scale,
            run_config.overlap,
            spatial_factor,
            run_config.tile_grid_mode,
            run_config.tile_grid_min_overlap,
        )
        tile_grid = latent_tile_grid_from_pixel_grid(pixel_grid, spatial_factor)
        input_grid = tile_grid
        stitch_grid = pixel_grid
        tile_sets = [extract_all_tiles(sample, tile_grid) for sample in samples]
        upscaler = latent_upscaler_for_scale(components.latent_upscaler, run_config.resolution_scale)
        if upscaler is None:
            raise ValueError(
                f"The latent-upscaler path has no {run_config.resolution_scale}x model "
                f"(loaded LU scales: {_loaded_lu_scales(components)})"
            )
        prepare_chunk = lambda chunk: [
            upscale_lr_latent_tile(tile, upscaler, components.vae, run_config.device) for tile in chunk
        ]
    else:
        height, width = samples[0].shape[-2:]
        base, _tile_hw, grid = _tile_geometry(
            height,
            width,
            visual_size,
            run_config.resolution_scale,
            run_config.overlap,
            _spatial_factor(components),
            run_config.tile_grid_mode,
            run_config.tile_grid_min_overlap,
        )
        tile_sets = [_upsample_tiles_to_base(extract_all_tiles(sample, grid), base[0], base[1]) for sample in samples]
        input_grid = grid
        stitch_grid = grid
        prepare_chunk = None

    tile_inputs = [
        tile_sets[sample_index][tile_index]
        for tile_index in range(input_grid.total_tiles)
        for sample_index in range(batch_size)
    ]
    outputs = _run_tile_batches(
        tile_inputs,
        components,
        run_config,
        _scale_factor(params, visual_size),
        scheduler=scheduler,
        prepare_chunk=prepare_chunk,
        progress_bar=progress_bar,
    )
    return _stitch_batch(outputs, stitch_grid, batch_size, height, width, run_config.resolution_scale)


def _sr_progress_plan(
    value: Tensor,
    components: Any,
    run_config: Any,
    *,
    latent_input: bool,
    batch_size: int = 1,
    scheduler: Any | None = None,
) -> tuple[int, int, int]:
    """Return total updates, denoising steps, and tile batches."""
    params = _component_params(components)
    visual_size = int(params.visual_size[0]) if isinstance(params.visual_size, list) else int(params.visual_size)
    spatial_factor = _spatial_factor(components)
    height, width = value.shape[-2:]
    if latent_input:
        height *= spatial_factor
        width *= spatial_factor
    _, _, grid = _tile_geometry(
        height,
        width,
        visual_size,
        run_config.resolution_scale,
        run_config.overlap,
        spatial_factor,
    )
    tile_count = grid.total_tiles * batch_size
    chunk_count = max(1, math.ceil(tile_count / run_config.tiles_batch_size))
    piflow_params = _scheduler_piflow_params(scheduler) or getattr(components.dit, "piflow_params", None)
    if isinstance(piflow_params, Mapping):
        denoise_steps = int(piflow_params.get("nfe", run_config.num_steps - 1))
    elif piflow_params is not None:
        denoise_steps = int(getattr(piflow_params, "nfe", run_config.num_steps - 1))
    else:
        denoise_steps = run_config.num_steps - 1
    if getattr(components, "sampler", None) is not None:
        denoise_steps = 1
    denoise_steps = max(1, denoise_steps)
    return chunk_count * denoise_steps, denoise_steps, chunk_count


class Kandinsky6SRPipeline(DiffusionPipeline):
    r"""Standalone Diffusers pipeline for Kandinsky 6 video super-resolution.

    Components are supplied to the constructor or loaded by Diffusers through
    ``from_pretrained``.  This pipeline deliberately has no config-path
    factory: model construction belongs to the Diffusers component package,
    while this class owns only SR orchestration and I/O.

    ``resolution_scale=2.25`` is the production route.  It performs a 1.125x
    pixel pre-upscale followed by the x2 latent-upscaler path.  The input
    components must already be on a device; ``.to(device)`` can be used after
    construction.

    Args:
        transformer: SR transformer used to denoise latent tiles.
        vae: Video VAE used for pixel-input encoding.
        scheduler: Scheduler used for SR denoising.
        latent_upscaler: Optional x2/x4 latent upscaler bank.
        source_vae: Optional source VAE used only by the KVAE latent bridge.
    """

    model_cpu_offload_seq = "source_vae->latent_upscaler->transformer->vae"
    _optional_components = ["latent_upscaler", "source_vae"]

    def __init__(  # noqa: PLR0913
        self,
        transformer: nn.Module,
        vae: nn.Module,
        scheduler: Any,
        latent_upscaler: nn.Module | None = None,
        source_vae: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            latent_upscaler=latent_upscaler,
            source_vae=source_vae,
        )
        transformer_config = getattr(transformer, "config", None)
        self._sr_params = _coerce_sr_params(getattr(transformer_config, "sr_params", None))
        if self._sr_params is None:
            raise ValueError("SR transformer config must contain 'sr_params'")

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

    def check_inputs(
        self,
        video: SRBatchInput | None = None,
        latents: Tensor | np.ndarray | str | Path | Sequence[Any] | None = None,
        *,
        resolution_scale: float,
        num_inference_steps: int | None = None,
        overlap: float | None = None,
        tiles_batch_size: int | None = None,
        kvae_bridge: bool = False,
        save_path: str | Path | list[str | Path] | None = None,
        fps: int | None = None,
        audio: list[np.ndarray] | np.ndarray | None = None,
        source_video: str | Path | list[str | Path] | None = None,
        audio_sample_rate: int = 44100,
        output_type: SROutputType = "pt",
    ) -> None:
        """Validate SR input, scale, tiling, audio, and output arguments.

        Args:
            video: Pixel video input in a supported frame layout.
            latents: Raw latent tensor or local latent file.
            resolution_scale: Total output scale, including the 2.25x
                fractional route.
            num_inference_steps: Number of denoising steps.
            overlap: Tile overlap fraction in ``[0, 1)``.
            tiles_batch_size: Number of tiles processed together.
            save_path: Optional output video path.
            fps: Optional output video frame rate.
            audio: Optional audio waveform to mux.
            source_video: Optional source video whose audio is copied.
            audio_sample_rate: Sample rate used when muxing audio.
            output_type: ``pt``/``torch`` or ``np``/``numpy``.

        Raises:
            ValueError: If inputs conflict or a value is outside the SR
                pipeline's supported range.
        """
        if (video is None) == (latents is None):
            raise ValueError("pass exactly one of video or latents")
        if output_type not in ("pt", "torch", "np", "numpy"):
            raise ValueError(f"unsupported output_type={output_type!r}")
        if audio is not None and source_video is not None:
            raise ValueError("pass either audio or source_video, not both")
        if audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be positive")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")

        requested_scale = float(resolution_scale)
        resolve_scale_request(requested_scale)
        effective_steps = 5 if num_inference_steps is None else int(num_inference_steps)
        effective_overlap = 0.25 if overlap is None else float(overlap)
        effective_tiles_batch_size = 1 if tiles_batch_size is None else int(tiles_batch_size)
        if effective_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not 0 <= effective_overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if effective_tiles_batch_size < 1:
            raise ValueError("tiles_batch_size must be positive")
        if latents is not None:
            if requested_scale == 2.25 and not kvae_bridge:  # noqa: PLR2004
                raise ValueError(
                    "2.25x latent SR requires kvae_bridge=True and source_vae; "
                    "ordinary latent input supports only x2/x4"
                )
            if kvae_bridge and self.source_vae is None:
                raise ValueError("KVAE bridge latent SR requires source_vae")

    @staticmethod
    def _load_video(value: SRInput) -> tuple[Tensor, float | None, Path | None]:
        if isinstance(value, (str, Path)):
            path = Path(value)
            video, source_fps = read_video_tchw_uint8(path)
            video, output_fps = resample_to_target_fps(video, source_fps)
            return clip_to_aligned_frames(video), float(output_fps), path
        return _to_tchw(value), None, None

    @classmethod
    def _load_videos(cls, value: SRBatchInput) -> tuple[list[Tensor], list[float | None], list[Path | None]]:
        if isinstance(value, (Tensor, np.ndarray)) and value.ndim == VIDEO_BATCH_RANK:
            values: Sequence[Any] = _video_batch_tensor(value)
        elif _is_video_batch(value):
            values = value
        else:
            values = [value]
        loaded = [cls._load_video(item) for item in values]
        videos = [item[0] for item in loaded]
        _validate_batch_shapes(videos, "video")
        return videos, [item[1] for item in loaded], [item[2] for item in loaded]

    @staticmethod
    def _format_frames(frames: Tensor, output_type: SROutputType) -> Tensor | np.ndarray:
        if output_type in ("pt", "torch"):
            return frames
        if output_type in ("np", "numpy"):
            return frames.detach().cpu().numpy()
        raise ValueError(f"Unsupported output_type={output_type!r}; use 'pt'/'torch' or 'np'/'numpy'")

    @torch.no_grad()
    def _run(  # noqa: PLR0913
        self,
        *,
        video: list[Tensor] | None = None,
        latents: list[Tensor] | None = None,
        resolution_scale: float,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        overlap: float | None = None,
        tiles_batch_size: int | None = None,
        kvae_bridge: bool = False,
        cached_text_embeds: dict[str, Tensor] | None = None,
        save_path: str | Path | list[str | Path] | None = None,
        fps: int | None = None,
        audio: list[np.ndarray] | np.ndarray | None = None,
        source_video: str | Path | list[str | Path] | None = None,
        audio_sample_rate: int = 44100,
        output_type: SROutputType = "pt",
        return_dict: bool = True,
    ) -> Kandinsky6SRPipelineOutput | tuple[Any, ...]:
        requested_scale = float(resolution_scale)
        effective_steps = 5 if num_inference_steps is None else int(num_inference_steps)
        effective_overlap = 0.25 if overlap is None else float(overlap)
        effective_tiles_batch_size = 1 if tiles_batch_size is None else int(tiles_batch_size)
        self.check_inputs(
            video=video,
            latents=latents,
            resolution_scale=requested_scale,
            num_inference_steps=effective_steps,
            overlap=effective_overlap,
            tiles_batch_size=effective_tiles_batch_size,
            kvae_bridge=kvae_bridge,
            save_path=save_path,
            fps=fps,
            audio=audio,
            source_video=source_video,
            audio_sample_rate=audio_sample_rate,
            output_type=output_type,
        )
        tiling_scale, pre_upscale = resolve_scale_request(requested_scale)
        components = SRComponents(
            dit=self.transformer,
            vae=self.vae,
            latent_upscaler=self.latent_upscaler,
            sr_params=self._sr_params,
            cached_text_embeds=cached_text_embeds,
            spatial_factor=getattr(self.vae, "spatial_factor", None),
        )

        batch_size = len(latents) if latents is not None else len(video) if video is not None else 0
        save_paths = _batch_output_values(save_path, batch_size, "save_path")
        source_videos = _batch_output_values(source_video, batch_size, "source_video")
        sr_video = video
        try:
            if latents is not None and kvae_bridge:
                sr_video = [
                    _decode_source_latent_video(latent, self.source_vae, self._execution_device) for latent in latents
                ]
                latents = None
            if sr_video is not None and pre_upscale != 1.0:
                sr_video = [
                    pre_upscale_video(video_item, pre_upscale, _spatial_factor(components)) for video_item in sr_video
                ]

            run_config = RunConfig(
                device=str(self._execution_device),
                num_steps=effective_steps,
                seed=self._seed_from_generator(generator, self._execution_device),
                overlap=effective_overlap,
                tiles_batch_size=effective_tiles_batch_size * batch_size,
                resolution_scale=tiling_scale,
            )
            progress_input = latents[0] if latents is not None else sr_video[0] if sr_video is not None else None
            if progress_input is None:
                raise RuntimeError("SR input disappeared during preparation")
            progress_total, progress_steps, progress_tiles = _sr_progress_plan(
                progress_input,
                components,
                run_config,
                latent_input=latents is not None,
                batch_size=batch_size,
                scheduler=self.scheduler,
            )
            with self.progress_bar(total=progress_total) as progress_bar:
                progress_bar.set_description(f"SR [{progress_steps} steps x {progress_tiles} tiles]")
                has_latent_upscaler = (
                    latent_upscaler_for_scale(self.latent_upscaler, run_config.resolution_scale) is not None
                )
                if latents is not None:
                    frames = _run_batched_tiles(
                        latents,
                        components,
                        run_config,
                        latent_input=True,
                        progress_bar=progress_bar,
                        scheduler=self.scheduler,
                    )
                elif has_latent_upscaler:
                    lr_latents = list(
                        encode_lq_videos_to_lr_latents(
                            sr_video,
                            self.vae,
                            self._execution_device,
                        )
                    )
                    frames = _run_batched_tiles(
                        lr_latents,
                        components,
                        run_config,
                        latent_input=True,
                        progress_bar=progress_bar,
                        scheduler=self.scheduler,
                    )
                else:
                    frames = _run_batched_tiles(
                        sr_video,
                        components,
                        run_config,
                        latent_input=False,
                        progress_bar=progress_bar,
                        scheduler=self.scheduler,
                    )
        finally:
            # Diffusers expects custom pipelines to restore the offloaded
            # modules after a call.  This also keeps the next invocation from
            # observing a partially resident model chain.
            self.maybe_free_model_hooks()

        path: str | list[str] | None = None
        if save_path is not None:
            paths = [
                str(
                    mux_video_audio(
                        frames[index],
                        audio,
                        output_path,
                        fps=fps or int(getattr(self._sr_params, "fps", TARGET_FPS)),
                        audio_sample_rate=audio_sample_rate,
                        source_video=source_videos[index],
                    )
                )
                for index, output_path in enumerate(save_paths)
            ]
            path = paths if isinstance(save_path, list) else paths[0]
        audio_source = None
        if source_video is not None:
            audio_source = (
                [str(value) for value in source_videos] if isinstance(source_video, list) else str(source_video)
            )
        output = Kandinsky6SRPipelineOutput(
            frames=self._format_frames(frames, output_type),
            audio=None if audio is None else (audio if isinstance(audio, list) else [audio]),
            path=path,
            metadata={
                "resolution_scale": requested_scale,
                "num_inference_steps": run_config.num_steps,
                "overlap": run_config.overlap,
                "tiles_batch_size": effective_tiles_batch_size,
                **({"audio_source": audio_source} if audio_source is not None else {}),
            },
        )
        if return_dict:
            return output
        return output.frames, output.audio, output.path, output.metadata

    @torch.no_grad()
    def from_video(  # noqa: PLR0913
        self,
        video: SRBatchInput,
        resolution_scale: float = 2.25,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        overlap: float | None = None,
        tiles_batch_size: int | None = None,
        cached_text_embeds: dict[str, Tensor] | None = None,
        save_path: str | Path | list[str | Path] | None = None,
        fps: int | None = None,
        audio: list[np.ndarray] | np.ndarray | None = None,
        source_video: str | Path | list[str | Path] | None = None,
        audio_sample_rate: int = 44100,
        output_type: SROutputType = "pt",
        return_dict: bool = True,
    ) -> Kandinsky6SRPipelineOutput | tuple[Any, ...]:
        """Super-resolve pixels and optionally copy audio from a source path."""
        frames, input_fps, input_paths = self._load_videos(video)
        input_path = input_paths[0] if len(input_paths) == 1 else None
        if source_video is None and save_path is not None:
            if input_path is not None:
                source_video = input_path
            elif all(path is not None for path in input_paths):
                source_video = [path for path in input_paths if path is not None]
        with _execution_device_context(self._execution_device):
            return self._run(
                video=frames,
                resolution_scale=resolution_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                overlap=overlap,
                tiles_batch_size=tiles_batch_size,
                cached_text_embeds=cached_text_embeds,
                save_path=save_path,
                fps=(
                    fps
                    if fps is not None
                    else round(input_fps[0])
                    if len(input_fps) == 1 and input_fps[0] is not None
                    else None
                ),
                audio=audio,
                source_video=source_video,
                audio_sample_rate=audio_sample_rate,
                output_type=output_type,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def from_latents(  # noqa: PLR0913
        self,
        latents: Tensor | np.ndarray | str | Path | Sequence[Any],
        *,
        resolution_scale: float = 2.25,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        overlap: float | None = None,
        tiles_batch_size: int | None = None,
        kvae_bridge: bool = False,
        cached_text_embeds: dict[str, Tensor] | None = None,
        save_path: str | Path | list[str | Path] | None = None,
        fps: int | None = None,
        audio: list[np.ndarray] | np.ndarray | None = None,
        source_video: str | Path | list[str | Path] | None = None,
        audio_sample_rate: int = 44100,
        output_type: SROutputType = "pt",
        return_dict: bool = True,
    ) -> Kandinsky6SRPipelineOutput | tuple[Any, ...]:
        """Super-resolve raw latent samples or a batch of local ``.pt`` files."""
        channels = int(getattr(self.transformer, "in_visual_dim", 16))
        latent_value = _latent_batch(latents, channels)
        with _execution_device_context(self._execution_device):
            return self._run(
                latents=latent_value,
                resolution_scale=resolution_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                overlap=overlap,
                tiles_batch_size=tiles_batch_size,
                kvae_bridge=kvae_bridge,
                cached_text_embeds=cached_text_embeds,
                save_path=save_path,
                fps=fps,
                audio=audio,
                source_video=source_video,
                audio_sample_rate=audio_sample_rate,
                output_type=output_type,
                return_dict=return_dict,
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: SRBatchInput | None = None,
        latents: Tensor | np.ndarray | str | Path | Sequence[Any] | None = None,
        resolution_scale: float = 2.25,
        num_inference_steps: int = 5,
        generator: torch.Generator | None = None,
        overlap: float = 0.25,
        tiles_batch_size: int = 1,
        kvae_bridge: bool = False,
        cached_text_embeds: dict[str, Tensor] | None = None,
        save_path: str | Path | list[str | Path] | None = None,
        fps: int | None = None,
        audio: list[np.ndarray] | np.ndarray | None = None,
        source_video: str | Path | list[str | Path] | None = None,
        audio_sample_rate: int = 44100,
        output_type: SROutputType = "pt",
        return_dict: bool = True,
    ) -> Kandinsky6SRPipelineOutput | tuple[Any, ...]:
        """Super-resolve one or more equal-size pixel videos or latent inputs.

        Args:
            video: Pixel video in ``[T,C,H,W]``/``[T,H,W,C]`` form, a batch
                in a supported rank-5 layout, or local video paths.
            latents: Raw latent tensor, a supported rank-5 latent batch, or
                local ``.pt`` files.
            resolution_scale: Total spatial upscale factor. Supported values
                are ``2``, ``4``, and ``2.25``.
            num_inference_steps: Number of denoising steps for the SR model.
            generator: Optional random generator used to derive the sampling seed.
            overlap: Fraction of overlap between adjacent tiles.
            tiles_batch_size: Number of tiles processed in one sampler batch.
            kvae_bridge: Whether latent inputs come from a base VAE and need to
                be decoded before SR.
            cached_text_embeds: Optional cached empty-caption embeddings.
            save_path: Optional output video path or one path per input.
            fps: Optional output video frame rate.
            audio: Optional audio waveform to mux.
            source_video: Optional source video whose audio is copied.
            audio_sample_rate: Sample rate used when muxing audio.
            output_type: ``pt``/``torch`` or ``np``/``numpy``.
            return_dict: Whether to return ``Kandinsky6SRPipelineOutput``.

        Examples:

        Returns:
            ``Kandinsky6SRPipelineOutput`` or its tuple representation when
            ``return_dict=False``.
        """
        if (video is None) == (latents is None):
            raise ValueError("pass exactly one of video or latents")

        if video is not None:
            return self.from_video(
                video,
                resolution_scale=resolution_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                overlap=overlap,
                tiles_batch_size=tiles_batch_size,
                cached_text_embeds=cached_text_embeds,
                save_path=save_path,
                fps=fps,
                audio=audio,
                source_video=source_video,
                audio_sample_rate=audio_sample_rate,
                output_type=output_type,
                return_dict=return_dict,
            )

        return self.from_latents(
            latents,
            resolution_scale=resolution_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            overlap=overlap,
            tiles_batch_size=tiles_batch_size,
            kvae_bridge=kvae_bridge,
            cached_text_embeds=cached_text_embeds,
            save_path=save_path,
            fps=fps,
            audio=audio,
            source_video=source_video,
            audio_sample_rate=audio_sample_rate,
            output_type=output_type,
            return_dict=return_dict,
        )  # type: ignore[arg-type]


def _to_tchw(value: Tensor | np.ndarray | Sequence[Any]) -> Tensor:  # noqa: PLR0912, PLR2004
    """Normalize common frame layouts to contiguous uint8 ``[T,C,H,W]``."""
    if isinstance(value, Tensor):
        tensor = value.detach()
    else:
        if isinstance(value, Sequence) and not isinstance(value, np.ndarray):
            value = np.stack([np.asarray(frame) for frame in value])
        tensor = torch.as_tensor(value)
    if tensor.ndim == VIDEO_RANK:
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] not in (1, 3, 4):
            raise ValueError("4D video must be [T,C,H,W] or [T,H,W,C]")
    else:
        raise ValueError(f"video sample must have rank 4, got shape {tuple(tensor.shape)}")
    if tensor.is_floating_point():
        if tensor.numel() and float(tensor.detach().amax()) <= 1.0:
            tensor = tensor * 255.0
        tensor = tensor.round()
    return tensor.clamp(0, 255).to(torch.uint8).contiguous()


def _coerce_sr_params(
    value: Any | None,
) -> Any | None:
    if value is not None:
        if isinstance(value, Mapping) and "sr" in value:
            value = value["sr"]
        if isinstance(value, Mapping):
            values = dict(value)
            if "scale_factor" in values and isinstance(values["scale_factor"], Mapping):
                values["scale_factor"] = {
                    int(key): [float(item) for item in items] for key, items in values["scale_factor"].items()
                }
            if "visual_size" in values and isinstance(values["visual_size"], int):
                values["visual_size"] = [values["visual_size"]]
            return SimpleNamespace(**values)
        return value
    return None


def _decode_source_latent_video(
    raw_latents: Tensor,
    source_vae: nn.Module,
    device: str | torch.device,
) -> Tensor:
    """Decode a base-VAE latent to ``[T,C,H,W]`` pixels for the KVAE bridge."""
    if raw_latents.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"raw_latents must have rank 4 [T,C,H,W], got {tuple(raw_latents.shape)}")
    latent_5d = raw_latents.permute(1, 0, 2, 3).unsqueeze(0).to(device=device)
    latent_5d = cast_to_module_dtype(source_vae, latent_5d)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        try:
            decoded = _call_module_method(source_vae, "decode", latent_5d, alternative_fwd=True).sample
        except TypeError as exc:
            if "alternative_fwd" not in str(exc):
                raise
            decoded = _call_module_method(source_vae, "decode", latent_5d).sample
    return (
        ((decoded.squeeze(0).float().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 0, 2, 3)
        .cpu()
    )


__all__ = ["Kandinsky6SRPipeline"]
