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

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import Kandinsky6SRPipelineOutput


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:

        ```python
        >>> import torch
        >>> from diffusers import Kandinsky6SRPipeline, Kandinsky6TI2VAPipeline

        >>> t2va = Kandinsky6TI2VAPipeline.from_pretrained(
        ...     "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> t2va.enable_model_cpu_offload()
        >>> generated = t2va(prompt="A cat and a dog baking a cake together in a kitchen.", height=480, width=864)

        >>> pipe = Kandinsky6SRPipeline.from_pretrained(
        ...     "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> output = pipe(video=generated.frames, resolution_scale=2.25)
        >>> output.frames.shape
        torch.Size([1, 3, 121, 1080, 1944])
        ```
"""


# These values are the SR data contract and must stay local to this pipeline.
VAE_SPATIAL_FACTOR = 16
RESOLUTIONS: dict[int, list[tuple[int, int]]] = {512: [(512, 512), (512, 768), (768, 512)]}
MAX_NUM_FRAMES = 121
ResolutionScale = Literal[2, 4]


# Shared tiling utilities for spatial video tiling with Hanning-window blending.


class TileGrid(NamedTuple):
    """Spatial tile grid parameters.

    ``tops`` and ``lefts`` are the authoritative per-axis tile start positions.
    They cover ``[0, length - tile]``, evenly spread so every gap keeps at
    least the configured overlap (see :func:`axis_positions`).
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


def axis_positions(length: int, tile: int, min_overlap: float, snap: int) -> tuple[int, ...]:
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


def compute_tile_grid(h: int, w: int, tile_hw: tuple[int, int], min_overlap: float, snap: int) -> TileGrid:
    """Build a :class:`TileGrid` covering ``(h, w)`` with the given tile size.

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
        tops=axis_positions(h, tile_h, min_overlap, snap),
        lefts=axis_positions(w, tile_w, min_overlap, snap),
    )


def extract_all_tiles(video: torch.Tensor, grid: TileGrid) -> list[torch.Tensor]:
    """Extract all spatial tiles from a video tensor.

    Args:
        video: ``[C, T, H, W]`` tensor (only the last two dims are read).
        grid: Tile grid parameters from ``compute_tile_grid``.

    Returns:
        List of ``[C, T, tile_h, tile_w]`` tensors in row-major order
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


def resolve_scale_request(scale: float) -> tuple[int, float]:
    """Map a requested ``resolution_scale`` to ``(tiling_scale, pre_upscale)``.

    Args:
        scale: The requested total upscale — ``2``, ``4`` or ``2.25``.

    Returns:
        ``(tiling_scale, pre_upscale)``; ``pre_upscale`` is ``1.0`` for the
        integer scales.
    """
    requested = float(scale)
    if requested == 2.25:
        return 2, 1.125
    if requested in (2.0, 4.0):
        return int(requested), 1.0
    raise ValueError("SR supports total scales 2, 4, and 2.25")


def pre_upscale_video(video: torch.Tensor, factor: float, spatial_multiple: int) -> torch.Tensor:
    """Bilinear-upscale a ``[C, T, H, W]`` uint8 video by ``factor`` in pixel space.

    Target dims are rounded to the nearest multiple of ``spatial_multiple``
    (the VAE spatial factor) so the whole-video encode and the latent tile
    grid stay integer-aligned (``latent_tile_grid_from_pixel_grid`` rejects
    unaligned grids). For sources whose scaled dims already land on the
    factor (512x768 x1.125 -> 576x864) the rounding is a no-op and the total
    scale is exact.

    Args:
        video: ``[C, T, H, W]`` uint8 source video (only the last two dims are read).
        factor: Pixel upscale factor (> 1).
        spatial_multiple: VAE spatial factor to align the target dims to.

    Returns:
        ``[C, T, H', W']`` uint8 video with ``H' ~= H * factor`` aligned.
    """
    if video.ndim != 4:
        raise ValueError(f"video must have rank 4 [C,T,H,W], got {tuple(video.shape)}")
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


def get_visual_size(x: torch.Tensor, spatial_factor: int) -> int:
    """Return the resolution key matching a visual latent tensor's spatial dims.

    Args:
        x: Visual latent tensor of shape ``(T, H, W, C)``.
        spatial_factor: The VAE's spatial compression factor.

    Returns:
        The matching resolution key.

    Raises:
        ValueError: If tensor dimensions do not match any known resolution.
    """
    actual_size = (x.shape[1] * spatial_factor, x.shape[2] * spatial_factor)
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
    eps = randn_tensor(lq_latent.shape, generator=generator, device=lq_latent.device, dtype=lq_latent.dtype)
    if noise_type == "ddpm":
        return (1 - noise_scale**2) ** 0.5 * lq_latent + noise_scale * eps
    return (1 - noise_scale) * lq_latent + noise_scale * eps


def cast_to_module_dtype(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Cast floating-point inputs to the module's parameter dtype."""
    try:
        dtype = next(module.parameters()).dtype
    except StopIteration:
        return value
    if value.is_floating_point() and value.dtype != dtype:
        return value.to(dtype=dtype)
    return value


def encode_pixels_to_latent(vae: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """Encode ``(B, C, T, H, W)`` pixels in ``[0, 255]`` with the KVAE.

    Normalizes with the KVAE's own convention, then returns the latent
    (``(latent, split_list)[0]`` — the regularizer mode; the KVAE always
    returns this mode, there is no sampling toggle).

    Args:
        vae: Causal video KVAE.
        pixels: ``(B, C, T, H, W)`` tensor in ``[0, 255]``.

    Returns:
        Latent ``(B, C, T', H', W')`` — not yet scaled by ``scaling_factor``.
    """
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
    float_pixels = (vae.denormalize_data(decoded) / 255.0).clamp(0.0, 1.0)
    return (float_pixels * 255.0).to(torch.uint8)


def encode_lq_video_to_lr_latent(lq_video: torch.Tensor, vae: torch.nn.Module, device: str | torch.device) -> Tensor:
    """Encode one whole ``[C, T, H, W]`` pixel video into a raw ``[C, T', H', W']`` latent.

    Unlike :func:`Kandinsky6SRPipeline._encode_lq_videos` (a batch of spatial
    tiles, channels-last, pre-scaled for the latent upscaler), this encodes a
    single un-tiled video and leaves the result unscaled — the caller tiles it
    in latent space and applies ``scaling_factor`` per tile (see
    ``_upscale_lr_latent_tile``).

    Args:
        lq_video: ``[C, T, H, W]`` uint8 video in ``[0, 255]``.
        vae: Causal video KVAE.
        device: Target device.

    Returns:
        Unscaled latent ``[C, T', H', W']``.

    Raises:
        ValueError: If ``lq_video`` is not rank 4.
    """
    if lq_video.ndim != 4:
        raise ValueError(f"lq_video must have rank 4 [C,T,H,W], got {tuple(lq_video.shape)}")
    pixel = lq_video.unsqueeze(0).to(device=device)
    latent = encode_pixels_to_latent(vae, pixel)
    return latent.squeeze(0).float()


@dataclass
class RunConfig:
    """Parameters for one tiled SR run."""

    device: str
    num_steps: int = 5
    seed: int = 42
    min_overlap: float = 0.20
    tiles_batch_size: int = 1
    resolution_scale: ResolutionScale = 4

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        if not 0.0 <= self.min_overlap < 1.0:
            raise ValueError("min_overlap must satisfy 0 <= min_overlap < 1")
        if self.tiles_batch_size <= 0:
            raise ValueError("tiles_batch_size must be positive")
        if self.resolution_scale not in (2, 4):
            raise ValueError("resolution_scale must be 2 or 4")


def _closest_base_resolution(h: int, w: int, visual_size: int) -> tuple[int, int]:
    if visual_size not in RESOLUTIONS:
        raise ValueError(f"Unsupported SR visual_size={visual_size}; known sizes: {sorted(RESOLUTIONS)}")
    ratio = w / h if h else 1.0
    return min(RESOLUTIONS[visual_size], key=lambda hw: abs(hw[1] / hw[0] - ratio))


def _tile_geometry(
    h: int,
    w: int,
    visual_size: int,
    scale: int,
    min_overlap: float,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
) -> tuple[tuple[int, int], tuple[int, int], TileGrid]:
    base_h, base_w = _closest_base_resolution(h, w, visual_size)
    if base_h % scale or base_w % scale:
        raise ValueError(f"resolution_scale={scale} does not divide the base resolution {base_h}x{base_w} exactly")
    tile_hw = (base_h // scale, base_w // scale)
    grid = compute_tile_grid(h, w, tile_hw, min_overlap, spatial_factor)
    return (base_h, base_w), tile_hw, grid


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


def _upsample_tiles_to_base(raw_tiles: list[torch.Tensor], base_h: int, base_w: int) -> list[torch.Tensor]:
    """Bilinear-upscale ``[C,T,h,w]`` pixel tiles to ``[T,base_h,base_w,C]`` (channels-last, for ``_encode_lq_videos``)."""
    return [
        functional.interpolate(tile.float(), size=(base_h, base_w), mode="bilinear", align_corners=False).permute(
            1, 2, 3, 0
        )
        for tile in raw_tiles
    ]


def _stitch_batch(
    outputs: list[Tensor],
    grid: TileGrid,
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


def _prepare_batch(value: Tensor, name: str, channels: int, *, dtype: torch.dtype | None = None) -> Tensor:
    """Add a batch dim to a single ``(C,T,H,W)`` sample; validate a ``(B,C,T,H,W)`` batch."""
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5:
        raise ValueError(f"`{name}` must have shape (C,T,H,W) or (B,C,T,H,W), got {tuple(value.shape)}")
    if value.shape[1] != channels:
        raise ValueError(f"`{name}` must have {channels} channels at dim 1, got shape {tuple(value.shape)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"`{name}` must be {dtype}, got {value.dtype}")
    return value


@contextmanager
def _execution_device_context(device: torch.device | str):
    """Scope SR model execution to Diffusers' resolved CUDA device."""
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        with torch.cuda.device(resolved_device):
            yield
    else:
        yield


def _decode_source_latent_video(raw_latents: Tensor, source_vae: nn.Module, device: str | torch.device) -> Tensor:
    """Decode a base-VAE latent to ``[C,T,H,W]`` pixels for the KVAE bridge."""
    if raw_latents.ndim != 4:
        raise ValueError(f"raw_latents must have rank 4 [C,T,H,W], got {tuple(raw_latents.shape)}")
    latent_5d = raw_latents.unsqueeze(0).to(device=device)
    latent_5d = cast_to_module_dtype(source_vae, latent_5d)
    # `source_vae.decode` is decorated with `@apply_forward_hook` (the standard convention on every
    # AutoencoderKL-family `encode`/`decode` in this repo), so a Diffusers CPU-offload hook on
    # `source_vae` already fires correctly here without any manual hook dispatch.
    device_type = torch.device(device).type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        decoded = source_vae.decode(latent_5d).sample
    return ((decoded.squeeze(0).float().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()


class Kandinsky6SRPipeline(DiffusionPipeline):
    r"""Standalone Diffusers pipeline for Kandinsky 6 video super-resolution.

    Components are supplied to the constructor or loaded by Diffusers through
    ``from_pretrained``.  This pipeline deliberately has no config-path
    factory: model construction belongs to the Diffusers component package,
    while this class owns only SR orchestration.

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

    def __init__(
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
        if not getattr(transformer, "sr_params", None):
            raise ValueError("SR transformer config must contain 'sr_params'")
        vae_config = getattr(vae, "config", None)
        self.spatial_factor = int(
            getattr(vae, "spatial_factor", None) or getattr(vae_config, "spatial_factor", None) or VAE_SPATIAL_FACTOR
        )

    def check_inputs(
        self,
        video: Tensor | None = None,
        latents: Tensor | None = None,
        *,
        resolution_scale: float,
        num_inference_steps: int,
        min_overlap: float,
        tiles_batch_size: int,
        kvae_bridge: bool = False,
        output_type: str = "pt",
    ) -> None:
        """Validate SR input, scale, tiling, and output arguments.

        Args:
            video: Pixel video batch, already normalized to rank 5.
            latents: Latent batch, already normalized to rank 5.
            resolution_scale: Total output scale, including the 2.25x
                fractional route.
            num_inference_steps: Number of denoising steps.
            min_overlap: Minimum tile overlap fraction in ``[0, 1)``.
            tiles_batch_size: Number of tiles processed together.
            kvae_bridge: Whether ``latents`` come from a different VAE and
                must be bridged through ``self.source_vae``.
            output_type: ``pt``/``torch`` or ``np``/``numpy``.

        Raises:
            ValueError: If inputs conflict or a value is outside the SR
                pipeline's supported range.
        """
        if (video is None) == (latents is None):
            raise ValueError("pass exactly one of `video` or `latents`")
        if output_type not in ("pt", "torch", "np", "numpy"):
            raise ValueError(f"unsupported output_type={output_type!r}")
        if video is not None:
            num_frames = video.shape[2]
            if (num_frames - 1) % 8 != 0:
                raise ValueError(
                    f"`video` must have 1 + 8k frames (1, 9, 17, ..., {MAX_NUM_FRAMES}, ...), got {num_frames}"
                )
            if num_frames > MAX_NUM_FRAMES:
                raise ValueError(f"`video` must have at most {MAX_NUM_FRAMES} frames (~5s at 24fps), got {num_frames}")

        resolve_scale_request(float(resolution_scale))
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not 0.0 <= min_overlap < 1.0:
            raise ValueError("min_overlap must be in [0, 1)")
        if tiles_batch_size < 1:
            raise ValueError("tiles_batch_size must be positive")
        if latents is not None:
            if float(resolution_scale) == 2.25 and not kvae_bridge:
                raise ValueError(
                    "2.25x latent SR requires kvae_bridge=True and source_vae; "
                    "ordinary latent input supports only x2/x4"
                )
            if kvae_bridge and self.source_vae is None:
                raise ValueError("KVAE bridge latent SR requires source_vae")

    @staticmethod
    def _format_frames(frames: Tensor, output_type: str) -> Tensor | np.ndarray:
        if output_type in ("pt", "torch"):
            return frames
        if output_type in ("np", "numpy"):
            return frames.detach().cpu().numpy()
        raise ValueError(f"Unsupported output_type={output_type!r}; use 'pt'/'torch' or 'np'/'numpy'")

    def _sparse_params(self, visual: torch.Tensor, visual_cu_seqlens: torch.Tensor) -> dict[str, Any] | None:
        """Build NABLA sparse-attention parameters for ``self.transformer`` and the given visual tensor.

        Returns ``None`` for dense (``"flash"``) attention. The target ``P`` is used directly (the
        training-time linear warmup schedule does not apply at inference).
        """
        model = self.transformer
        if model.patch_size[0] != 1:
            raise ValueError(f"Expected transformer.patch_size[0] == 1, got {model.patch_size[0]}")
        t, h, w, _ = visual.shape
        t, h, w = (t // model.patch_size[0], h // model.patch_size[1], w // model.patch_size[2])
        visual_size = get_visual_size(visual, self.spatial_factor)
        attention_configs = model.attention_params
        try:
            attention_params = attention_configs[visual_size]
        except KeyError:
            # JSON object keys are always strings, while the native YAML config
            # uses integer resolution keys.
            attention_params = attention_configs[str(visual_size)]

        if attention_params.get("type") != "nabla":
            return None
        return {
            "to_fractal": True,
            "P": attention_params.get("P"),
            "wT": attention_params.get("wT"),
            "wW": attention_params.get("wW"),
            "wH": attention_params.get("wH"),
            "add_sta": attention_params.get("add_sta"),
            "visual_shape": (t, h, w),
            "visual_seqlens": visual_cu_seqlens,
        }

    @torch.no_grad()
    def _encode_lq_videos(self, lq_videos: list[torch.Tensor], device: str | torch.device) -> torch.Tensor:
        """VAE-encode a batch of ``[T,H,W,C]`` spatial tiles into a stacked ``[sum(T'),H',W',C]`` latent.

        Each tile is encoded independently (rather than as one batched VAE
        call) to mirror the reference implementation exactly; ``self.vae``'s
        ``scaling_factor`` is applied here, since these latents feed the
        denoising loop directly (unlike ``encode_lq_video_to_lr_latent``,
        whose unscaled output is scaled later by ``_upscale_lr_latent_tile``).
        """
        vae = self.vae
        scaling_factor = float(vae.config.scaling_factor)
        latents = []
        for lq in lq_videos:
            pixel = lq.permute(3, 0, 1, 2).unsqueeze(0).to(device=device)
            latent = encode_pixels_to_latent(vae, pixel)
            latents.append(latent.squeeze(0).permute(1, 2, 3, 0).float() * scaling_factor)
        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def _upscale_lr_latent_tile(
        self, lr_latent_tile: torch.Tensor, scale: int, device: str | torch.device
    ) -> torch.Tensor:
        """Upscale one raw ``[C,T,H,W]`` latent tile via ``self.latent_upscaler``'s ``scale``x entry."""
        scaling_factor = float(getattr(getattr(self.vae, "config", None), "scaling_factor", 1.0))
        tile = lr_latent_tile.unsqueeze(0).to(device=device, dtype=torch.float32)
        tile = cast_to_module_dtype(self.latent_upscaler, tile)
        device_type = torch.device(device).type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            upscaled = self.latent_upscaler(tile * scaling_factor, entry=f"{scale}x", return_intermediates=False)
        return upscaled.squeeze(0).permute(1, 2, 3, 0).float()

    def _build_initial_latent(
        self,
        lq_latent: torch.Tensor,
        bs: int,
        duration: int,
        height: int,
        width: int,
        device: str | torch.device,
        generator: torch.Generator | None,
        lq_noise_scale: float,
        lq_noise_type: Literal["linear", "ddpm"],
        lq_channel_noise_scale: float,
    ) -> torch.Tensor:
        """Build the initial latent tensor for the SR denoising loop, per ``self.transformer.instruct_type``.

        Constructs ``[starting | cond | mask]`` depending on ``instruct_type``:

        - ``"noise"``: degraded LQ as starting point, zero LQ cond + zero mask.
        - ``"channel"``: random Gaussian noise as starting point, LQ cond + ones mask.
        - ``"hybrid"``: degraded LQ as starting point, LQ cond + ones mask.
        - ``"hybrid_anchor"``: degraded LQ as starting point, zeroed (anchor-free)
          HR anchor + zeroed anchor_mask — the only mode Kandinsky 6 SR ships with.

        Returns:
            Initial latent tensor ready for the denoising loop (bs*T, H, W, 33).

        Raises:
            ValueError: If ``instruct_type`` is not supported.
        """
        dit = self.transformer
        lq_latent = lq_latent.to(device)
        lq_latent = cast_to_module_dtype(dit, lq_latent)
        if dit.instruct_type == "noise":
            degraded_lq = degrade_lq_latent(lq_latent, lq_noise_scale, lq_noise_type, generator=generator)
            if not getattr(dit, "visual_cond", False):
                return degraded_lq
            zero_cond = torch.zeros_like(degraded_lq)
            zero_mask = torch.zeros([*degraded_lq.shape[:-1], 1], dtype=degraded_lq.dtype, device=device)
            return torch.cat([degraded_lq, zero_cond, zero_mask], dim=-1)
        if dit.instruct_type in ("channel", "hybrid"):
            if dit.instruct_type == "hybrid":
                starting = degrade_lq_latent(lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=generator)
            else:
                starting = randn_tensor(
                    (bs * duration, height, width, dit.in_visual_dim), generator=generator, device=device
                )
            channel_lq = degrade_lq_latent(lq_latent, lq_channel_noise_scale, noise_type="linear")
            mask = torch.ones_like(lq_latent[..., :1])
            return torch.cat([starting, channel_lq, mask], dim=-1)
        if dit.instruct_type == "hybrid_anchor":
            # Kandinsky 6 SR ships anchor-free: the public API has no way to supply a real HR
            # anchor, so the anchor and its mask are always zeroed.
            anchor_latent = torch.zeros_like(lq_latent)
            anchor_mask = torch.zeros_like(lq_latent[..., :1])
            starting = degrade_lq_latent(lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=generator)
            anchor = degrade_lq_latent(anchor_latent, lq_channel_noise_scale, noise_type="linear")
            return torch.cat([starting, anchor, anchor_mask], dim=-1)
        raise ValueError(f"Kandinsky6SRPipeline does not support instruct_type={dit.instruct_type!r}")

    def _prepare_sr_latents(
        self,
        *,
        lq_videos: list[torch.Tensor] | None,
        lq_latents: torch.Tensor | None,
        n_samples: int | None,
        device: str | torch.device,
        generator: torch.Generator | None,
        lq_noise_scale: float,
        lq_noise_type: str,
        lq_channel_noise_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """Encode LQ input and build the initial latent for one SR tile batch.

        Returns:
            ``(lq_latent, image, batch_size, duration, height, width)``.
        """
        if lq_latents is not None:
            lq_latent = lq_latents.to(device)
            if n_samples is None:
                raise ValueError("n_samples is required when lq_latents is provided")
            batch_size = n_samples
        elif lq_videos is not None:
            batch_size = len(lq_videos)
            lq_latent = self._encode_lq_videos(lq_videos, device)
        else:
            raise ValueError("Either lq_videos or lq_latents must be provided")
        duration = lq_latent.shape[0] // batch_size
        height, width = (lq_latent.shape[1], lq_latent.shape[2])
        image = self._build_initial_latent(
            lq_latent,
            bs=batch_size,
            duration=duration,
            height=height,
            width=width,
            device=device,
            generator=generator,
            lq_noise_scale=lq_noise_scale,
            lq_noise_type=lq_noise_type,
            lq_channel_noise_scale=lq_channel_noise_scale,
        )
        return lq_latent, image, batch_size, duration, height, width

    def _encode_sr_text(
        self,
        batch_size: int,
        device: str | torch.device,
        cached_text_embeds: dict[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build the SR text state: empty placeholders, or cached empty-caption embeddings.

        Kandinsky 6 SR ships with a text-free transformer (``use_text=False``) and has no
        text encoder registered on the pipeline, so this never runs a real text encoder;
        it only replicates precomputed ``cached_text_embeds`` to the batch size, or returns
        placeholders when the transformer ignores text entirely.

        Raises:
            ValueError: If the transformer needs text but no ``cached_text_embeds`` were given.
        """
        if not getattr(self.transformer, "use_text", True):
            # A use_text=False DiT ignores every text input (encoder, cross-attention, and pooled-text
            # conditioning are all absent), so these zero-size tensors only satisfy the indexing and
            # cu_seqlens bookkeeping in the denoising loop without ever loading a cached empty-caption file.
            embeds = {
                "text_embeds": torch.zeros(0, 1, device=device),
                "pooled_embed": torch.zeros(batch_size, 1, device=device),
            }
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            return embeds, cu_seqlens
        if cached_text_embeds is None:
            raise ValueError("this SR transformer requires text; pass cached_text_embeds")
        single_text = cached_text_embeds["text_embeds"]
        single_pooled = cached_text_embeds["pooled_embed"]
        seq_len = int(cached_text_embeds["cu_seqlens"][-1].item())
        text_embeds = {
            "text_embeds": single_text.repeat(batch_size, 1).to(device),
            "pooled_embed": single_pooled.repeat(batch_size, 1).to(device),
        }
        cu_seqlens = seq_len * torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        return text_embeds, cu_seqlens

    def _text_positions(
        self,
        image: torch.Tensor,
        batch_size: int,
        duration: int,
        height: int,
        width: int,
        text_cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Build the packed visual/text RoPE positions ``self.transformer`` needs for one SR call."""
        device = image.device
        visual_cu_seqlens = duration * torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        visual_rope_pos = [
            torch.cat([torch.arange(int(end), device=device) for end in torch.diff(visual_cu_seqlens).cpu()]),
            torch.arange(height // self.transformer.patch_size[1], device=device),
            torch.arange(width // self.transformer.patch_size[2], device=device),
        ]
        text_cu_seqlens = text_cu_seqlens.to(device)
        text_rope_pos = torch.cat([torch.arange(int(end), device=device) for end in torch.diff(text_cu_seqlens).cpu()])
        return visual_cu_seqlens, visual_rope_pos, text_rope_pos

    def _predict(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: dict[str, torch.Tensor],
        visual_cu_seqlens: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        visual_rope_pos: list[torch.Tensor],
        text_rope_pos: torch.Tensor,
        scale_factor: tuple[float, ...],
        sparse_params: dict | None,
    ) -> torch.Tensor:
        """Run one ``self.transformer`` forward. SR has no guidance: a single conditional pass."""
        model_time = (t * 1000).to(dtype=x.dtype)
        motion_score = torch.full((1,), 900.0, device=x.device, dtype=x.dtype)
        return self.transformer(
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

    @torch.no_grad()
    def _denoise_native(
        self,
        img: torch.Tensor,
        device: str | torch.device,
        num_steps: int,
        text_embeds: dict[str, torch.Tensor],
        visual_cu_seqlens: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        visual_rope_pos: list[torch.Tensor],
        text_rope_pos: torch.Tensor,
        scale_factor: tuple[float, ...],
        scheduler_scale: float,
        start_timestep: float = 1.0,
        progress_bar: Any = None,
    ) -> torch.Tensor:
        """Run the native Euler loop from ``start_timestep`` to 0 for the non-piflow SR transformer.

        This intentionally does not go through ``self.scheduler``: the SR transformer predicts
        velocity directly and this loop applies the same shift formula
        ``FlowMatchEulerDiscreteScheduler`` uses (``scheduler_scale`` mirrors its ``shift``), matching
        the reference implementation bit-for-bit.
        """
        img = img.to(device)
        visual_cu_seqlens = visual_cu_seqlens.to(device)
        text_cu_seqlens = text_cu_seqlens.to(device)
        visual_rope_pos = [position.to(device) for position in visual_rope_pos]
        text_rope_pos = text_rope_pos.to(device)
        text_embeds = {key: value.to(device) for key, value in text_embeds.items()}
        sparse_params = self._sparse_params(img, visual_cu_seqlens)
        timesteps = torch.linspace(start_timestep, 0, num_steps, device=device)
        timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)
        out_channels = self.transformer.in_visual_dim
        for timestep, timestep_diff in zip(timesteps[:-1], torch.diff(timesteps), strict=False):
            time = timestep.unsqueeze(0).repeat(visual_cu_seqlens.shape[0] - 1)
            velocity = self._predict(
                img,
                time,
                text_embeds,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                scale_factor,
                sparse_params,
            )
            out_channels = velocity.shape[-1]
            img[..., :out_channels] += timestep_diff * velocity
            if progress_bar is not None:
                progress_bar.update()
                progress_bar.refresh()
        return img[..., :out_channels]

    @torch.no_grad()
    def _denoise_piflow(
        self,
        img: torch.Tensor,
        text_embeds: dict[str, torch.Tensor],
        visual_cu_seqlens: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        visual_rope_pos: list[torch.Tensor],
        text_rope_pos: torch.Tensor,
        scale_factor: tuple[float, ...],
        *,
        nfe: int,
        out_dim: int,
        device: str | torch.device | None = None,
        progress_bar: Any = None,
    ) -> torch.Tensor:
        """Few-step π-Flow denoise: ``nfe`` transformer calls, via ``self.scheduler``.

        ``self.scheduler`` (a ``PiflowScheduler``) owns the segment schedule and the
        network-free policy integration between transformer calls entirely from its own config.
        """
        device = device if device is not None else img.device
        img = img.to(device)
        visual_cu_seqlens = visual_cu_seqlens.to(device)
        text_cu_seqlens = text_cu_seqlens.to(device)
        visual_rope_pos = [position.to(device) for position in visual_rope_pos]
        text_rope_pos = text_rope_pos.to(device)
        text_embeds = {key: value.to(device) for key, value in text_embeds.items()}
        sparse_params = self._sparse_params(img, visual_cu_seqlens)
        if nfe < 1:
            raise ValueError(f"piflow denoising requires nfe >= 1, got {nfe}")
        self.scheduler.set_timesteps(nfe, device=device)
        if self.scheduler.timesteps.shape[0] != nfe:
            raise ValueError(f"Piflow scheduler returned {self.scheduler.timesteps.shape[0]} steps, expected {nfe}")
        x = img
        for step_index in range(nfe):
            timestep = self.scheduler.timesteps[step_index]
            n_objects = visual_cu_seqlens.shape[0] - 1
            t_src = torch.full((n_objects,), float(timestep.item()), device=device, dtype=x.dtype)
            v0 = self._predict(
                x,
                t_src,
                text_embeds,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                scale_factor,
                sparse_params,
            )
            x_pred = self.scheduler.step(v0, timestep, x[..., :out_dim], return_dict=False)[0]
            x = torch.cat([x_pred, x[..., out_dim:]], dim=-1)
            if progress_bar is not None:
                progress_bar.update()
                progress_bar.refresh()
        return x[..., :out_dim]

    def _scheduler_nfe(self) -> int | None:
        """Return the configured π-Flow step count, or ``None`` for a regular scheduler."""
        if not bool(getattr(self.scheduler, "is_piflow", False)):
            return None
        nfe = getattr(self.scheduler.config, "nfe", None)
        if nfe is None:
            raise ValueError("PiflowScheduler used for SR must define nfe in scheduler_config.json")
        return int(nfe)

    def _denoise(
        self,
        image: torch.Tensor,
        text_embeds: dict[str, torch.Tensor],
        text_cu_seqlens: torch.Tensor,
        batch_size: int,
        duration: int,
        height: int,
        width: int,
        *,
        scale_factor: tuple[float, ...],
        num_steps: int,
        scheduler_scale: float,
        lq_noise_scale: float,
        cap_noise_timestep: bool,
        device: str | torch.device,
        progress_bar: Any = None,
    ) -> torch.Tensor:
        """Run either the Euler or π-Flow SR denoising stage on ``self.transformer``/``self.scheduler``."""
        nfe = self._scheduler_nfe()
        visual_cu_seqlens, visual_rope_pos, text_rope_pos = self._text_positions(
            image, batch_size, duration, height, width, text_cu_seqlens
        )
        if nfe is not None:
            if cap_noise_timestep and self.transformer.instruct_type in ("noise", "hybrid"):
                raise NotImplementedError(
                    "piflow sampler does not support cap_noise_timestep for noise/hybrid instruct"
                )
            out_dim = int(getattr(self.transformer, "base_out_visual_dim", self.transformer.in_visual_dim))
            return self._denoise_piflow(
                image,
                text_embeds,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                scale_factor,
                nfe=nfe,
                out_dim=out_dim,
                device=device,
                progress_bar=progress_bar,
            )
        start_timestep = (
            lq_noise_scale
            if cap_noise_timestep and self.transformer.instruct_type in ("noise", "hybrid", "hybrid_anchor")
            else 1.0
        )
        return self._denoise_native(
            image,
            device,
            num_steps,
            text_embeds,
            visual_cu_seqlens,
            text_cu_seqlens,
            visual_rope_pos,
            text_rope_pos,
            scale_factor,
            scheduler_scale,
            start_timestep=start_timestep,
            progress_bar=progress_bar,
        )

    @torch.no_grad()
    def _vae_decode(
        self,
        latent_visual: torch.Tensor,
        batch_size: int,
        duration: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Decode denoised SR latents into ``[batch, 3, T, H, W]`` uint8 frames with ``self.vae``."""
        device_type = latent_visual.device.type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            all_latents = latent_visual.reshape(batch_size, duration, height, width, -1)
            all_latents = (all_latents / self.vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            decoded = [decode_latent_to_uint8(self.vae, all_latents[i : i + 1]) for i in range(batch_size)]
            return torch.cat(decoded, dim=0)

    @torch.no_grad()
    def _run_stages(
        self,
        *,
        scale_factor: tuple[float, ...],
        num_steps: int,
        scheduler_scale: float,
        device: str | torch.device,
        generator: torch.Generator | None,
        lq_noise_scale: float,
        lq_noise_type: str,
        lq_channel_noise_scale: float,
        cap_noise_timestep: bool,
        cached_text_embeds: dict[str, torch.Tensor] | None,
        lq_videos: list[torch.Tensor] | None = None,
        lq_latents: torch.Tensor | None = None,
        n_samples: int | None = None,
        progress_bar: Any = None,
    ) -> torch.Tensor:
        """Run the four SR stages — prepare latents, encode text, denoise, decode — for one tile batch."""
        lq_latent, image, batch_size, duration, height, width = self._prepare_sr_latents(
            lq_videos=lq_videos,
            lq_latents=lq_latents,
            n_samples=n_samples,
            device=device,
            generator=generator,
            lq_noise_scale=lq_noise_scale,
            lq_noise_type=lq_noise_type,
            lq_channel_noise_scale=lq_channel_noise_scale,
        )
        text_embeds, text_cu_seqlens = self._encode_sr_text(batch_size, device, cached_text_embeds)
        latent_visual = self._denoise(
            image,
            text_embeds,
            text_cu_seqlens,
            batch_size,
            duration,
            height,
            width,
            scale_factor=scale_factor,
            num_steps=num_steps,
            scheduler_scale=scheduler_scale,
            lq_noise_scale=lq_noise_scale,
            cap_noise_timestep=cap_noise_timestep,
            device=device,
            progress_bar=progress_bar,
        )
        return self._vae_decode(latent_visual, batch_size, duration, height, width)

    def _tile_batch_kwargs(
        self,
        run_config: RunConfig,
        scale_factor: tuple[float, ...],
        cached_text_embeds: dict[str, torch.Tensor] | None,
        seed: int,
        **inputs: Any,
    ) -> dict[str, Any]:
        # A fresh per-tile Generator, seeded deterministically from the run's base seed plus this
        # tile batch's offset, so a tile's noise never depends on how many other tiles ran before it.
        generator = torch.Generator(device=run_config.device)
        generator.manual_seed(seed)
        transformer = self.transformer
        return {
            **inputs,
            "scale_factor": scale_factor,
            "num_steps": run_config.num_steps,
            "scheduler_scale": transformer.scheduler_scale,
            "generator": generator,
            "device": run_config.device,
            "lq_noise_scale": transformer.lq_noise_scale,
            "lq_noise_type": transformer.lq_noise_type,
            "lq_channel_noise_scale": transformer.lq_channel_noise_scale,
            "cap_noise_timestep": transformer.cap_noise_timestep,
            "cached_text_embeds": cached_text_embeds,
        }

    def _loaded_lu_scales(self) -> tuple[int, ...]:
        if self.latent_upscaler is None:
            return ()
        return tuple(self.latent_upscaler.scales)

    def _run_tile_batches(
        self,
        tile_inputs: list[torch.Tensor],
        run_config: RunConfig,
        scale_factor: tuple[float, ...],
        *,
        cached_text_embeds: dict[str, torch.Tensor] | None,
        prepare_chunk: Any | None = None,
        progress_bar: Any = None,
    ) -> list[torch.Tensor]:
        """Run the batched sampler loop for latent or pixel tile inputs.

        The transformer and SR VAE stay resident for the complete tile loop. Only the
        optional latent-upscaler is streamed per chunk.
        """
        outputs: list[torch.Tensor] = []
        for start in range(0, len(tile_inputs), run_config.tiles_batch_size):
            raw_chunk = tile_inputs[start : start + run_config.tiles_batch_size]
            chunk = prepare_chunk(raw_chunk) if prepare_chunk is not None else raw_chunk
            input_kwargs: dict[str, Any] = (
                {"lq_videos": chunk}
                if prepare_chunk is None
                else {"lq_latents": torch.cat(chunk, dim=0), "n_samples": len(raw_chunk)}
            )
            kwargs = self._tile_batch_kwargs(
                run_config,
                scale_factor,
                cached_text_embeds,
                run_config.seed + start,
                **input_kwargs,
            )
            sr_batch = self._run_stages(**kwargs, progress_bar=progress_bar)
            if sr_batch.ndim != 5:
                raise ValueError(f"SR sampler must return [batch,C,T,H,W], got {tuple(sr_batch.shape)}")
            outputs.extend(sample.float().cpu() for sample in sr_batch)
        return outputs

    def _run_batched_tiles(
        self,
        samples: list[Tensor],
        run_config: RunConfig,
        *,
        latent_input: bool,
        cached_text_embeds: dict[str, torch.Tensor] | None,
        progress_bar: Any,
    ) -> Tensor:
        """Run equal-shape samples tile-major and restore the sample batch."""
        visual_size = self.transformer.visual_size
        batch_size = len(samples)
        scale_factor = self.transformer.scale_factor[visual_size]
        if latent_input:
            spatial_factor = self.spatial_factor
            height = samples[0].shape[-2] * spatial_factor
            width = samples[0].shape[-1] * spatial_factor
            _, _, pixel_grid = _tile_geometry(
                height,
                width,
                visual_size,
                run_config.resolution_scale,
                run_config.min_overlap,
                spatial_factor,
            )
            tile_grid = latent_tile_grid_from_pixel_grid(pixel_grid, spatial_factor)
            input_grid = tile_grid
            stitch_grid = pixel_grid
            tile_sets = [extract_all_tiles(sample, tile_grid) for sample in samples]
            if run_config.resolution_scale not in self._loaded_lu_scales():
                raise ValueError(
                    f"The latent-upscaler path has no {run_config.resolution_scale}x model "
                    f"(loaded LU scales: {self._loaded_lu_scales()})"
                )

            def prepare_chunk(chunk: list[Tensor]) -> list[Tensor]:
                return [
                    self._upscale_lr_latent_tile(tile, run_config.resolution_scale, run_config.device)
                    for tile in chunk
                ]
        else:
            height, width = samples[0].shape[-2:]
            spatial_factor = self.spatial_factor
            base, _tile_hw, grid = _tile_geometry(
                height,
                width,
                visual_size,
                run_config.resolution_scale,
                run_config.min_overlap,
                spatial_factor,
            )
            tile_sets = [
                _upsample_tiles_to_base(extract_all_tiles(sample, grid), base[0], base[1]) for sample in samples
            ]
            input_grid = grid
            stitch_grid = grid
            prepare_chunk = None

        tile_inputs = [
            tile_sets[sample_index][tile_index]
            for tile_index in range(input_grid.total_tiles)
            for sample_index in range(batch_size)
        ]
        outputs = self._run_tile_batches(
            tile_inputs,
            run_config,
            scale_factor,
            cached_text_embeds=cached_text_embeds,
            prepare_chunk=prepare_chunk,
            progress_bar=progress_bar,
        )
        return _stitch_batch(outputs, stitch_grid, batch_size, height, width, run_config.resolution_scale)

    def _sr_progress_plan(
        self,
        value: Tensor,
        run_config: RunConfig,
        *,
        latent_input: bool,
        batch_size: int = 1,
    ) -> tuple[int, int, int]:
        """Return total updates, denoising steps, and tile batches."""
        visual_size = self.transformer.visual_size
        spatial_factor = self.spatial_factor
        height, width = value.shape[-2:]
        if latent_input:
            height *= spatial_factor
            width *= spatial_factor
        _, _, grid = _tile_geometry(
            height, width, visual_size, run_config.resolution_scale, run_config.min_overlap, spatial_factor
        )
        tile_count = grid.total_tiles * batch_size
        chunk_count = max(1, math.ceil(tile_count / run_config.tiles_batch_size))
        nfe = self._scheduler_nfe()
        denoise_steps = nfe if nfe is not None else run_config.num_steps - 1
        denoise_steps = max(1, denoise_steps)
        return chunk_count * denoise_steps, denoise_steps, chunk_count

    @staticmethod
    def _seed_from_generator(generator: torch.Generator | None, device: torch.device) -> int:
        if generator is None:
            return int(torch.randint(0, 2**31, (1,), device=device).item())
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        return int(torch.randint(0, 2**31, (1,), generator=generator, device=generator_device).item())

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: Tensor | None = None,
        latents: Tensor | None = None,
        resolution_scale: float = 2.25,
        num_inference_steps: int = 5,
        generator: torch.Generator | None = None,
        min_overlap: float = 0.20,
        tiles_batch_size: int = 1,
        kvae_bridge: bool = False,
        cached_text_embeds: dict[str, Tensor] | None = None,
        output_type: str = "pt",
        return_dict: bool = True,
    ) -> Kandinsky6SRPipelineOutput | tuple[Tensor]:
        r"""Super-resolve one or more equal-size pixel videos or latent inputs.

        Args:
            video: LQ pixel video(s) to super-resolve, ``uint8`` in ``[0, 255]``,
                shape ``(3, T, H, W)`` or a batch ``(B, 3, T, H, W)`` — the same
                layout [`Kandinsky6TI2VAPipeline`] returns with
                ``output_type="pt"``. ``T`` must be ``1 + 8k`` frames (align or
                truncate longer/misaligned clips yourself) and at most
                ``MAX_NUM_FRAMES`` (121, ~5s at 24fps). Mutually exclusive with
                ``latents``.
            latents: Pre-encoded SR-VAE latent(s) already at the target
                (post-upscale) resolution, shape ``(C, T, H, W)`` or a batch
                ``(B, C, T, H, W)``, **unscaled** (not yet multiplied by
                ``self.vae.config.scaling_factor``). Mutually exclusive with
                ``video``. Pass ``kvae_bridge=True`` when these latents
                instead come from a different (base) VAE at the *source*
                resolution — they are then decoded through ``self.source_vae``
                and re-enter the normal ``video`` path (pre-upscale, encode,
                latent-upscale) rather than being used directly.
            resolution_scale: Total spatial upscale factor. Supported values
                are ``2``, ``4``, and ``2.25``.
            num_inference_steps: Number of denoising steps for the SR model.
            generator: Optional random generator used to derive the sampling seed.
            min_overlap: Minimum fraction of overlap between adjacent tiles.
            tiles_batch_size: Number of tiles processed in one sampler batch.
            kvae_bridge: Whether latent inputs come from a base VAE and need to
                be decoded before SR.
            cached_text_embeds: Optional cached empty-caption embeddings.
            output_type: ``pt``/``torch`` or ``np``/``numpy``.
            return_dict: Whether to return ``Kandinsky6SRPipelineOutput``.

        Examples:

        Returns:
            ``Kandinsky6SRPipelineOutput`` or its tuple representation when
            ``return_dict=False``.
        """
        if (video is None) == (latents is None):
            raise ValueError("pass exactly one of `video` or `latents`")
        if video is not None:
            video = _prepare_batch(video, "video", 3, dtype=torch.uint8)
        if latents is not None:
            latents = _prepare_batch(latents, "latents", int(self.transformer.in_visual_dim))

        self.check_inputs(
            video=video,
            latents=latents,
            resolution_scale=resolution_scale,
            num_inference_steps=num_inference_steps,
            min_overlap=min_overlap,
            tiles_batch_size=tiles_batch_size,
            kvae_bridge=kvae_bridge,
            output_type=output_type,
        )
        tiling_scale, pre_upscale = resolve_scale_request(float(resolution_scale))
        batch_size = latents.shape[0] if latents is not None else video.shape[0]
        sr_video = list(video.unbind(0)) if video is not None else None
        sr_latents = list(latents.unbind(0)) if latents is not None else None

        with _execution_device_context(self._execution_device):
            try:
                if sr_latents is not None and kvae_bridge:
                    sr_video = [
                        _decode_source_latent_video(latent, self.source_vae, self._execution_device)
                        for latent in sr_latents
                    ]
                    sr_latents = None
                if sr_video is not None and pre_upscale != 1.0:
                    spatial_factor = self.spatial_factor
                    sr_video = [pre_upscale_video(item, pre_upscale, spatial_factor) for item in sr_video]

                run_config = RunConfig(
                    device=str(self._execution_device),
                    num_steps=num_inference_steps,
                    seed=self._seed_from_generator(generator, self._execution_device),
                    min_overlap=min_overlap,
                    tiles_batch_size=tiles_batch_size * batch_size,
                    resolution_scale=tiling_scale,
                )
                progress_input = sr_latents[0] if sr_latents is not None else sr_video[0]
                progress_total, progress_steps, progress_tiles = self._sr_progress_plan(
                    progress_input,
                    run_config,
                    latent_input=sr_latents is not None,
                    batch_size=batch_size,
                )
                with self.progress_bar(total=progress_total) as progress_bar:
                    progress_bar.set_description(f"SR [{progress_steps} steps x {progress_tiles} tiles]")
                    has_latent_upscaler = run_config.resolution_scale in self._loaded_lu_scales()
                    if sr_latents is not None:
                        frames = self._run_batched_tiles(
                            sr_latents,
                            run_config,
                            latent_input=True,
                            cached_text_embeds=cached_text_embeds,
                            progress_bar=progress_bar,
                        )
                    elif has_latent_upscaler:
                        lr_latents = [
                            encode_lq_video_to_lr_latent(item, self.vae, self._execution_device) for item in sr_video
                        ]
                        frames = self._run_batched_tiles(
                            lr_latents,
                            run_config,
                            latent_input=True,
                            cached_text_embeds=cached_text_embeds,
                            progress_bar=progress_bar,
                        )
                    else:
                        frames = self._run_batched_tiles(
                            sr_video,
                            run_config,
                            latent_input=False,
                            cached_text_embeds=cached_text_embeds,
                            progress_bar=progress_bar,
                        )
            finally:
                # Diffusers expects custom pipelines to restore the offloaded
                # modules after a call.  This also keeps the next invocation from
                # observing a partially resident model chain.
                self.maybe_free_model_hooks()

        output = Kandinsky6SRPipelineOutput(frames=self._format_frames(frames, output_type))
        if not return_dict:
            return (output.frames,)
        return output
