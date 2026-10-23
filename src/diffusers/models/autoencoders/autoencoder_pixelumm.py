# Copyright 2026 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.accelerate_utils import apply_forward_hook
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput


TUBELET = (4, 16, 16)
TUBE_DIM = TUBELET[0] * TUBELET[1] * TUBELET[2] * 3
LATENT_CHANNELS = 32
# A DCAE latent cell is (4, 32, 32) pixels: 1 tubelet in time, 2 x 2 in space.
LATENT_TOKEN_GRID = 2


@dataclass(frozen=True)
class PixelUMMArchitecture:
    """Qwen3-8B (hidden 4096) with PixelUMM's SenseNova T/H/W MRoPE."""

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta_t: float = 1_000_000.0
    rope_theta_hw: float = 10_000.0

    def __post_init__(self):
        if self.head_dim % 4 or self.num_heads % self.num_kv_heads:
            raise ValueError("head_dim must be divisible by 4 and heads by kv heads")

    @property
    def rope_sections(self) -> tuple[int, int, int]:
        """Head-dim slices rotated by the T, H and W position respectively."""
        return self.head_dim // 2, self.head_dim // 4, self.head_dim // 4


def patchify(video: torch.Tensor) -> torch.Tensor:
    """[B,3,T,H,W] -> [B, (T/4)(H/16)(W/16), 3072]; token order T,H,W, values (t,p,q,c)."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")
    pt, ph, pw = TUBELET
    batch, channels, frames, height, width = video.shape
    if frames % pt or height % ph or width % pw:
        raise ValueError(f"video {tuple(video.shape[2:])} must be a multiple of {TUBELET}")
    x = video.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
    return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(batch, -1, TUBE_DIM)


def unpatchify(tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    batch, count, dim = tokens.shape
    nt, nh, nw = grid
    pt, ph, pw = TUBELET
    if count != nt * nh * nw or dim != TUBE_DIM:
        raise ValueError(f"tokens {tuple(tokens.shape)} do not match grid {grid}")
    x = tokens.reshape(batch, nt, nh, nw, pt, ph, pw, 3)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch, 3, nt * pt, nh * ph, nw * pw)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def mrope_tables(arch: PixelUMMArchitecture, grid: tuple[int, int, int], device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [1, T*H*W, 1, head_dim] for a THW-ordered grid, laid out [T | H | W]."""
    positions = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    thetas = (arch.rope_theta_t, arch.rope_theta_hw, arch.rope_theta_hw)
    cos_parts, sin_parts = [], []
    for position, dim, theta in zip(positions, arch.rope_sections, thetas):
        inv_freq = theta ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        angles = position.flatten().float()[:, None] * inv_freq[None]
        angles = torch.cat((angles, angles), dim=-1)
        cos_parts.append(angles.cos())
        sin_parts.append(angles.sin())
    return (
        torch.cat(cos_parts, dim=-1)[None, :, None, :],
        torch.cat(sin_parts, dim=-1)[None, :, None, :],
    )


def apply_mrope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sections) -> torch.Tensor:
    """x [B,N,H,D]; each section is rotated within itself (rotate_half per axis)."""
    rotated = torch.cat([_rotate_half(part) for part in x.split(tuple(sections), dim=-1)], dim=-1)
    return x * cos + rotated * sin


def local_attention_layout(grid, tile, device):
    frames, height, width = grid
    tiles_h, tiles_w = -(-height // tile), -(-width // tile)
    if tiles_h <= 3 and tiles_w <= 3:
        return None
    time, row, column = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    tile_h, tile_w = (row // tile).flatten(), (column // tile).flatten()
    order = (((tile_h * tiles_w + tile_w) * frames + time.flatten()) * tile + (row % tile).flatten()) * tile
    permutation = torch.argsort(order + (column % tile).flatten())
    ranges = []
    offset = 0
    for row_index in range(tiles_h):
        for column_index in range(tiles_w):
            count = frames * min(tile, height - row_index * tile) * min(tile, width - column_index * tile)
            ranges.append((offset, offset + count))
            offset += count
    neighborhoods = []
    for row_index in range(tiles_h):
        for column_index in range(tiles_w):
            center_h = min(max(row_index, 1), max(tiles_h - 2, 1))
            center_w = min(max(column_index, 1), max(tiles_w - 2, 1))
            neighbors = [
                ranges[h * tiles_w + w]
                for h in range(max(0, center_h - 1), min(tiles_h, center_h + 2))
                for w in range(max(0, center_w - 1), min(tiles_w, center_w + 2))
            ]
            neighborhoods.append((ranges[row_index * tiles_w + column_index], neighbors))
    block_mask = None
    if torch.device(device).type == "cuda":
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(batch, head, query_index, key_index):
            # The permutation groups tiles row by row, including partial edge tiles.
            row_tokens = frames * tile * width
            query_row, key_row = query_index // row_tokens, key_index // row_tokens
            query_height = (height - query_row * tile).clamp(1, tile)
            key_height = (height - key_row * tile).clamp(1, tile)
            query_column = (query_index % row_tokens) // (frames * query_height * tile)
            key_column = (key_index % row_tokens) // (frames * key_height * tile)
            center_row = query_row.clamp(1, max(tiles_h - 2, 1))
            center_column = query_column.clamp(1, max(tiles_w - 2, 1))
            return ((center_row - key_row).abs() <= 1) & ((center_column - key_column).abs() <= 1)

        count = frames * height * width
        block_mask = create_block_mask(
            mask_mod, None, None, count, count, device=device, _compile=not torch.compiler.is_compiling()
        )
    return permutation, torch.argsort(permutation), (neighborhoods, block_mask)


def _pixelumm_flex_attention(query, key, value, block_mask, parallel_config):
    return dispatch_attention_fn(
        query, key, value, attn_mask=block_mask, enable_gqa=True, backend="flex", parallel_config=parallel_config
    )


class PixelUMMRMSNorm(nn.Module):
    """Qwen3 RMSNorm: normalize in fp32, cast back to the input dtype, then scale."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class PixelUMMAttnProcessor:
    _attention_backend = None
    _parallel_config = None
    _compiled_flex_attention = None

    def __call__(self, attn, hidden_states, cos, sin, neighborhoods=None):
        batch, count, _ = hidden_states.shape
        arch = attn.arch
        query = attn.q_proj(hidden_states).view(batch, count, arch.num_heads, arch.head_dim)
        key = attn.k_proj(hidden_states).view(batch, count, arch.num_kv_heads, arch.head_dim)
        value = attn.v_proj(hidden_states).view(batch, count, arch.num_kv_heads, arch.head_dim)
        query_time, query_spatial = query.float().chunk(2, dim=-1)
        key_time, key_spatial = key.float().chunk(2, dim=-1)
        query = torch.cat((attn.q_norm_t(query_time), attn.q_norm_hw(query_spatial)), dim=-1)
        key = torch.cat((attn.k_norm_t(key_time), attn.k_norm_hw(key_spatial)), dim=-1)
        query = apply_mrope(query, cos, sin, arch.rope_sections).to(value.dtype)
        key = apply_mrope(key, cos, sin, arch.rope_sections).to(value.dtype)
        regions, block_mask = neighborhoods if neighborhoods is not None else (None, None)
        if block_mask is not None and self._attention_backend in (None, "flex"):
            if torch.compiler.is_compiling():
                hidden_states = _pixelumm_flex_attention(query, key, value, block_mask, self._parallel_config)
            else:
                if self._compiled_flex_attention is None:
                    self._compiled_flex_attention = torch.compile(_pixelumm_flex_attention, dynamic=False)
                hidden_states = self._compiled_flex_attention(query, key, value, block_mask, self._parallel_config)
        else:
            repeat = arch.num_heads // arch.num_kv_heads
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)
            if regions is None:
                hidden_states = dispatch_attention_fn(
                    query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config
                )
            else:
                outputs = []
                for (start, end), neighbors in regions:
                    local_key = torch.cat([key[:, first:last] for first, last in neighbors], dim=1)
                    local_value = torch.cat([value[:, first:last] for first, last in neighbors], dim=1)
                    outputs.append(
                        dispatch_attention_fn(
                            query[:, start:end].contiguous(),
                            local_key,
                            local_value,
                            backend=self._attention_backend,
                            parallel_config=self._parallel_config,
                        )
                    )
                hidden_states = torch.cat(outputs, dim=1)
        return attn.o_proj(hidden_states.reshape(batch, count, arch.num_heads * arch.head_dim))


class PixelUMMAttention(nn.Module, AttentionModuleMixin):
    _supports_qkv_fusion = False
    _default_processor_cls = PixelUMMAttnProcessor
    _available_processors = [PixelUMMAttnProcessor]

    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        hidden, heads, kv, dim = arch.hidden_size, arch.num_heads, arch.num_kv_heads, arch.head_dim
        self.q_proj = nn.Linear(hidden, heads * dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv * dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv * dim, bias=False)
        self.o_proj = nn.Linear(heads * dim, hidden, bias=False)
        self.q_norm_t = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.q_norm_hw = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_t = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_hw = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.set_processor(PixelUMMAttnProcessor())

    def forward(self, hidden_states, cos, sin, neighborhoods=None) -> torch.Tensor:
        return self.processor(self, hidden_states, cos, sin, neighborhoods)


class PixelUMMMLP(nn.Module):
    def __init__(self, arch: PixelUMMArchitecture):
        super().__init__()
        self.gate_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
        self.up_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
        self.down_proj = nn.Linear(arch.intermediate_size, arch.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PixelUMMBlock(nn.Module):
    def __init__(self, arch: PixelUMMArchitecture):
        super().__init__()
        self.input_norm = PixelUMMRMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.attn = PixelUMMAttention(arch)
        self.post_attn_norm = PixelUMMRMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.mlp = PixelUMMMLP(arch)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, neighborhoods=None) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin, neighborhoods)
        return x + self.mlp(self.post_attn_norm(x))


class PixelUMMHead(nn.Module):
    """Normalize hidden states and project each tubelet to RGB pixels."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.linear = nn.Linear(hidden, TUBE_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class PixelUMMDecoderModel(nn.Module):
    """`forward(latent [B,32,L,h,w], xt [B,3,4L,32h,32w]) -> x0 prediction`."""

    def __init__(self, arch: PixelUMMArchitecture, *, local_attention_tile: int = 0):
        super().__init__()
        if local_attention_tile < 0:
            raise ValueError(f"local_attention_tile must be >= 0 tokens, got {local_attention_tile}")
        self.arch = arch
        self.local_attention_tile = int(local_attention_tile)
        self.gradient_checkpointing = False
        hidden = arch.hidden_size
        self.embed = nn.Linear(TUBE_DIM, hidden)
        self.cond_proj = nn.Linear(LATENT_CHANNELS, LATENT_TOKEN_GRID**2 * hidden)
        self.layers = nn.ModuleList(PixelUMMBlock(arch) for _ in range(arch.num_layers))
        self.final_norm = PixelUMMRMSNorm(hidden, arch.rms_norm_eps)
        self.head = PixelUMMHead(hidden)

    def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"expected DCAE latent [B,{LATENT_CHANNELS},L,h,w], got {tuple(latent.shape)}")
        batch, _, frames, height, width = latent.shape
        pixel_shape = (batch, 3, frames * TUBELET[0], height * 32, width * 32)
        if tuple(xt.shape) != pixel_shape:
            raise ValueError(f"xt {tuple(xt.shape)} != expected {pixel_shape}")
        dtype = latent.dtype
        context = (
            torch.autocast(device_type=latent.device.type, dtype=dtype)
            if dtype in (torch.bfloat16, torch.float16)
            else nullcontext()
        )
        grid = (frames, height * LATENT_TOKEN_GRID, width * LATENT_TOKEN_GRID)
        with context:
            tokens = self.embed(patchify(xt.to(dtype)))
            condition = self.cond_proj(latent.to(dtype).permute(0, 2, 3, 4, 1))
            condition = condition.view(
                batch, frames, height, width, LATENT_TOKEN_GRID, LATENT_TOKEN_GRID, self.arch.hidden_size
            )
            condition = condition.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, -1, self.arch.hidden_size)
            tokens = tokens + condition.to(tokens.device)
            cos, sin = mrope_tables(self.arch, grid, tokens.device)
            layout = (
                local_attention_layout(grid, self.local_attention_tile, tokens.device)
                if self.local_attention_tile
                else None
            )
            neighborhoods = None
            if layout is not None:
                perm, inverse, neighborhoods = layout
                tokens, cos, sin = tokens[:, perm], cos[:, perm], sin[:, perm]
            for layer in self.layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    tokens = self._gradient_checkpointing_func(layer, tokens, cos, sin, neighborhoods)
                else:
                    tokens = layer(tokens, cos, sin, neighborhoods)
            tokens = self.final_norm(tokens)
            if layout is not None:
                tokens = tokens[:, inverse.to(tokens.device)]
        # The source runs the head after leaving autocast (forward_features returns first).
        return unpatchify(self.head(tokens), grid)


@dataclass
class StreamingPreviewUpdate(BaseOutput):
    """One complete segment preview plus any corrected preceding tail."""

    video: torch.Tensor
    pixel_start: int
    pixel_end: int
    replace_pixel_frames: int
    stable_pixel_frames: int
    provisional_pixel_frames: int


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    if window < 1 or stride < 1 or stride > window:
        raise ValueError(f"expected 1 <= stride <= window, got stride={stride}, window={window}")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    tail = length - window
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _slice_along(tensor: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]


def _merge_overlapping(parts: list[torch.Tensor], starts: list[int], *, dim: int) -> torch.Tensor:
    if len(parts) != len(starts) or not parts:
        raise ValueError("parts and starts must be non-empty and have equal length")
    result = parts[0]
    result_end = starts[0] + parts[0].shape[dim]
    for part, start in zip(parts[1:], starts[1:]):
        part_end = start + part.shape[dim]
        overlap = result_end - start
        if overlap < 0 or part_end <= result_end:
            raise ValueError("overlap windows must be ordered and extend the range")
        overlap = min(overlap, result.shape[dim], part.shape[dim])
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = 0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))
        shape = [1] * part.ndim
        shape[dim] = overlap
        weight = weight.view(shape).to(dtype=part.dtype)
        result_head = _slice_along(result, dim, 0, result.shape[dim] - overlap)
        result_tail = _slice_along(result, dim, result.shape[dim] - overlap, result.shape[dim])
        part_head = _slice_along(part, dim, 0, overlap)
        part_tail = _slice_along(part, dim, overlap, part.shape[dim])
        result = torch.cat([result_head, result_tail * (1.0 - weight) + part_head * weight, part_tail], dim=dim)
        result_end = part_end
    return result


class _TemporalBlender:
    """Incremental twin of `_merge_overlapping(parts, starts, dim=2)` into one buffer.

    Same arithmetic as the source's `_merge_into_buffer`: one preallocated clip buffer, each window cosine-blended in
    as soon as it is complete.
    """

    def __init__(self, total_frames: int):
        self.total = int(total_frames)
        self.out: torch.Tensor | None = None
        self.end = 0

    def add(self, part: torch.Tensor, start: int) -> None:
        length = part.shape[2]
        if self.out is None:
            self.out = part.new_empty((*part.shape[:2], self.total, *part.shape[3:]))
            self.out[:, :, :length] = part
            self.end = length
            return
        overlap = self.end - start
        if overlap < 0 or start + length <= self.end:
            raise ValueError(
                f"overlap windows must be ordered and extend the covered range: start={start}, result_end={self.end}, part_end={start + length}"
            )
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = (0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))).view(1, 1, -1, 1, 1)
        weight = weight.to(dtype=part.dtype)
        out = self.out
        previous = out[:, :, start : self.end]
        out[:, :, start : self.end] = previous * (1.0 - weight) + part[:, :, :overlap] * weight
        out[:, :, self.end : start + length] = part[:, :, overlap:]
        self.end = start + length

    def result(self) -> torch.Tensor:
        if self.out is None or self.end != self.total:
            raise RuntimeError(f"temporal merge incomplete: covered {self.end} of {self.total}")
        return self.out


class PixelUMMDecoder(ModelMixin, ConfigMixin, AttentionMixin):
    """
    PixelUMM DMD2 decoder for DCAE-V video latents.

    This decoder predicts RGB pixels from 32-channel latents with spatial compression 32 and temporal compression 4.
    Local attention covers three neighboring spatial tiles along each axis and the complete temporal window. The model
    supports standard [`ModelMixin.save_pretrained`] and [`ModelMixin.from_pretrained`] serialization.

    Parameters:
        hidden_size (`int`, defaults to 4096):
            Transformer hidden dimension.
        intermediate_size (`int`, defaults to 12288):
            Gated feed-forward intermediate dimension.
        num_layers (`int`, defaults to 36):
            Number of transformer layers.
        num_heads (`int`, defaults to 32):
            Number of query heads.
        num_kv_heads (`int`, defaults to 8):
            Number of key and value heads.
        head_dim (`int`, defaults to 128):
            Dimension of each attention head.
        rms_norm_eps (`float`, defaults to 1e-6):
            RMS normalization epsilon.
        rope_theta_t (`float`, defaults to 1000000.0):
            Rotary embedding frequency base for the temporal axis.
        rope_theta_hw (`float`, defaults to 10000.0):
            Rotary embedding frequency base for both spatial axes.
        local_attention_tile_pixels (`int`, defaults to 128):
            Spatial attention tile size in pixels. Zero enables full attention.
        denoising_step_list (`tuple[float]`, defaults to `(0.999, 0.866, 0.634, 0.342)`):
            Distilled noise schedule, in decreasing order.
        num_steps (`int`, defaults to 1):
            Number of entries to use from the distilled schedule.
        seed (`int`, defaults to 42):
            Pixel noise seed used when no generator is supplied.
        temporal_window_latents (`int`, defaults to 10):
            Number of latent frames decoded in each temporal window.
        temporal_overlap_latents (`int`, defaults to 5):
            Overlap between temporal windows, in latent frames.
        decoder_tile_pixels (`int`, defaults to 0):
            Spatial decode tile size in pixels. Zero decodes the full spatial field.
        decoder_tile_overlap_pixels (`int`, defaults to 64):
            Overlap between spatial decode tiles, in pixels.
        tail_hold_latents (`int`, defaults to 1):
            Number of tail latents withheld by preview consumers when a later segment will correct the tail.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["PixelUMMBlock"]
    _repeated_blocks = ["PixelUMMBlock"]
    _skip_layerwise_casting_patterns = ["embed", "cond_proj", "norm", "head"]

    spatial_compression_ratio = 32
    time_compression_ratio = 4
    temporal_compression_ratio = 4
    scaling_factor = 1.0

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_layers: int = 36,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta_t: float = 1_000_000.0,
        rope_theta_hw: float = 10_000.0,
        local_attention_tile_pixels: int = 128,
        denoising_step_list: tuple[float, ...] = (0.999, 0.866, 0.634, 0.342),
        num_steps: int = 1,
        seed: int = 42,
        temporal_window_latents: int = 10,
        temporal_overlap_latents: int = 5,
        decoder_tile_pixels: int = 0,
        decoder_tile_overlap_pixels: int = 64,
        tail_hold_latents: int = 1,
    ):
        super().__init__()
        if local_attention_tile_pixels < 0 or local_attention_tile_pixels % 16:
            raise ValueError("local_attention_tile_pixels must be zero or a multiple of 16")
        if temporal_window_latents < 1 or not 0 <= temporal_overlap_latents < temporal_window_latents:
            raise ValueError("temporal overlap must be nonnegative and smaller than the window")
        if decoder_tile_pixels < 0 or decoder_tile_pixels % 32:
            raise ValueError("decoder_tile_pixels must be zero or a multiple of 32")
        if decoder_tile_overlap_pixels < 0 or decoder_tile_overlap_pixels % 32:
            raise ValueError("decoder_tile_overlap_pixels must be a nonnegative multiple of 32")
        if decoder_tile_pixels and decoder_tile_overlap_pixels >= decoder_tile_pixels:
            raise ValueError("decoder overlap must be smaller than the tile")
        if tail_hold_latents < 0:
            raise ValueError("tail_hold_latents must be nonnegative")
        if not denoising_step_list or any(not 0 < value <= 1 for value in denoising_step_list):
            raise ValueError("denoising_step_list must contain values in (0, 1]")
        if any(first <= second for first, second in zip(denoising_step_list, denoising_step_list[1:])):
            raise ValueError("denoising_step_list must be strictly decreasing")
        if not 1 <= num_steps <= len(denoising_step_list):
            raise ValueError("num_steps must be between 1 and the distilled schedule length")
        arch = PixelUMMArchitecture(
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta_t,
            rope_theta_hw,
        )
        self.decoder = PixelUMMDecoderModel(arch, local_attention_tile=local_attention_tile_pixels // 16)
        self.temporal_window_latents = temporal_window_latents
        self.temporal_overlap_latents = temporal_overlap_latents
        self.tail_hold_latents = tail_hold_latents

    def forward(
        self, latent: torch.Tensor, sample: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | tuple[torch.Tensor]:
        """
        Predict a clean RGB video from a noisy video and its latent conditioning.

        Args:
            latent (`torch.Tensor`):
                Conditioning with shape `(batch, 32, frames, height, width)`.
            sample (`torch.Tensor`):
                Noisy RGB pixels with shape `(batch, 3, frames * 4, height * 32, width * 32)`.
            return_dict (`bool`, defaults to `True`):
                Return a [`~models.autoencoders.vae.DecoderOutput`] instead of a one-element tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                Predicted clean pixels with the same shape as `sample`.
        """
        output = self.decoder(latent, sample)
        return DecoderOutput(sample=output) if return_dict else (output,)

    def _make_noise(self, latent, start, end, seed):
        batch, _, _, height, width = latent.shape
        frames = []
        for index in range(start, end):
            generator = torch.Generator(device=latent.device).manual_seed((seed * 1_000_003 + index) % (2**64))
            frames.append(
                torch.randn(
                    (batch, 3, 4, height * 32, width * 32),
                    generator=generator,
                    device=latent.device,
                    dtype=torch.float32,
                )
            )
        return torch.cat(frames, dim=2)

    def _noise_tile(self, latent, seed):
        cached_range = None
        cached_noise = None

        def source(first, last, row, column, height, width):
            nonlocal cached_range, cached_noise
            if cached_range != (first, last):
                cached_noise = self._make_noise(latent, first, last, seed)
                cached_range = (first, last)
            return cached_noise[..., row * 32 : (row + height) * 32, column * 32 : (column + width) * 32]

        return source

    def _predict_x0(self, latent, xt_source, seed, t_starts=None, pad_short=False):
        frames, height, width = latent.shape[2:]
        window = self.config.temporal_window_latents
        stride = window - self.config.temporal_overlap_latents
        t_starts = _window_starts(frames, window, stride) if t_starts is None else t_starts
        length = min(window, frames)
        cells = self.config.decoder_tile_pixels // 32
        if cells <= 0 or (height <= cells and width <= cells):
            rows, columns, tile_h, tile_w = [0], [0], height, width
        else:
            spatial_stride = cells - self.config.decoder_tile_overlap_pixels // 32
            tile_h, tile_w = min(cells, height), min(cells, width)
            rows = _window_starts(height, tile_h, min(spatial_stride, tile_h))
            columns = _window_starts(width, tile_w, min(spatial_stride, tile_w))
        pad_noise = None
        if pad_short and frames >= window and t_starts[-1] + window > frames:
            variance, mean = torch.var_mean(latent.detach().float(), dim=(2, 3, 4), correction=0, keepdim=True)
            generator = torch.Generator(device=latent.device).manual_seed((seed * 1_000_003 + frames - 1) % (2**64))
            pad_noise = mean + variance.sqrt() * torch.randn(
                (latent.shape[0], latent.shape[1], window, height, width),
                generator=generator,
                device=latent.device,
                dtype=torch.float32,
            )
        blender = _TemporalBlender((t_starts[-1] + length - t_starts[0]) * 4)
        for first in t_starts:
            row_outputs = []
            for row in rows:
                tile_outputs = []
                for column in columns:
                    real = latent[:, :, first : first + length, row : row + tile_h, column : column + tile_w]
                    if pad_noise is not None and real.shape[2] < length:
                        fill = length - real.shape[2]
                        offset = (first // stride) % (window - fill + 1)
                        tile = pad_noise[:, :, offset : offset + fill, row : row + tile_h, column : column + tile_w]
                        real = torch.cat([real, tile.to(real.dtype)], dim=2)
                    noise = xt_source(first, first + length, row, column, tile_h, tile_w)
                    tile_outputs.append(self.decoder(real.contiguous(), noise.contiguous()))
                row_outputs.append(_merge_overlapping(tile_outputs, [column * 32 for column in columns], dim=-1))
            decoded = _merge_overlapping(row_outputs, [row * 32 for row in rows], dim=-2)
            blender.add(decoded, (first - t_starts[0]) * 4)
        return blender.result()

    @torch.no_grad()
    @apply_forward_hook
    def decode(
        self,
        latents: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        pixel_frames: int | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        """
        Decode a latent clip with overlapping temporal windows and optional spatial tiles.

        Args:
            latents (`torch.Tensor`):
                Latent video with shape `(batch, 32, frames, height, width)`.
            generator (`torch.Generator`, *optional*):
                Its initial seed selects the pixel noise stream. Each absolute latent-frame index has its own seed, so
                overlapping windows receive identical noise. The generator state is not advanced.
            return_dict (`bool`, defaults to `True`):
                Return a [`~models.autoencoders.vae.DecoderOutput`] instead of a one-element tuple.
            pixel_frames (`int`, *optional*):
                Keep this many leading decoded frames; otherwise return `frames * 4` frames.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                RGB pixels with shape `(batch, 3, pixel_frames, height * 32, width * 32)`.
        """
        if latents.ndim != 5 or latents.shape[1] != 32 or min(latents.shape[2:]) < 1:
            raise ValueError("latents must have nonempty shape (batch, 32, frames, height, width)")
        seed = self.config.seed if generator is None else generator.initial_seed()
        steps = self.config.denoising_step_list[: self.config.num_steps]
        if len(steps) == 1:
            video = self._predict_x0(latents, self._noise_tile(latents, seed), seed)
        else:
            state = self._make_noise(latents, 0, latents.shape[2], seed)

            def source(first, last, row, column, height, width):
                return state[
                    :, :, first * 4 : last * 4, row * 32 : (row + height) * 32, column * 32 : (column + width) * 32
                ]

            for index, current in enumerate(steps):
                video = self._predict_x0(latents, source, seed)
                if index + 1 < len(steps):
                    ratio = steps[index + 1] / current
                    for offset in range(0, state.shape[2], 40):
                        anchor = video[:, :, offset : offset + 40].to(device=state.device, dtype=torch.float32)
                        state[:, :, offset : offset + 40] = anchor + ratio * (
                            state[:, :, offset : offset + 40] - anchor
                        )
        if pixel_frames is not None:
            if not 1 <= pixel_frames <= video.shape[2]:
                raise ValueError("pixel_frames must be positive and no larger than the decoded frame count")
            video = video[:, :, :pixel_frames]
        return DecoderOutput(sample=video) if return_dict else (video,)

    @torch.no_grad()
    @apply_forward_hook
    def _decode_windows(
        self, latent: torch.Tensor, starts: list[int], seed: int, pad_short: bool = False
    ) -> torch.Tensor:
        latent = latent.to(device=self.device, dtype=self.dtype)
        return self._predict_x0(latent, self._noise_tile(latent, seed), seed, t_starts=starts, pad_short=pad_short)

    def create_streaming_decode_session(
        self, *, to_cpu: bool = True, retain_history: bool = True, generator: torch.Generator | None = None
    ) -> "PixelUMMStreamingDecodeSession":
        """
        Create independent state for append-only decoding, tail correction, and rollback.

        Streaming requires `num_steps=1`. Set `session.final_segment=False` before a nonfinal push to enable padded
        tail correction. Returned previews include their replacement range on the visible pixel timeline. `to_cpu`
        controls where returned pixels are stored. `retain_history` enables `slice_pixels` and `full_video`. The
        optional generator selects the same absolute-frame noise stream as [`~PixelUMMDecoder.decode`].
        """
        if self.config.num_steps != 1:
            raise ValueError("Streaming PixelUMM decoding requires num_steps=1")
        seed = self.config.seed if generator is None else generator.initial_seed()
        return PixelUMMStreamingDecodeSession(self, to_cpu=to_cpu, retain_history=retain_history, seed=seed)


class PixelUMMStreamingDecodeSession:
    """Independent latent timeline for stable previews, tail correction, and rollback."""

    def __init__(self, model, *, to_cpu: bool = True, retain_history: bool = True, seed: int = 42):
        self.model = model
        self.to_cpu = bool(to_cpu)
        self.retain_history = bool(retain_history)
        self.window = model.temporal_window_latents
        self.stride = self.window - model.temporal_overlap_latents
        self.temporal_ratio = model.time_compression_ratio
        self.buffer: torch.Tensor | None = None
        self.available_latent_frames = 0
        self.available_pixel_frames = 0
        self.next_window_start = 0
        self.pending: torch.Tensor | None = None
        self.pending_pixel_start = 0
        self.committed: list[tuple[int, torch.Tensor]] = []
        self.committed_raw_pixel_frames = 0
        self.committed_pixel_frames = 0
        self.push_latent_frames: list[int] = []
        self.push_pixel_frames: list[int] = []
        self.push_stable_pixel_frames: list[int] = []
        self.finished = False
        self.seed = seed
        self.final_segment = True

    def _append_decoded_windows(self, video: torch.Tensor, *, first_window_start: int) -> None:
        clip_start = first_window_start * self.temporal_ratio
        if self.pending is None:
            self.pending = video
            self.pending_pixel_start = clip_start
            return
        relative_start = clip_start - self.pending_pixel_start
        self.pending = _merge_overlapping([self.pending, video], [0, relative_start], dim=2)

    @property
    def available_raw_pixel_frames(self) -> int:
        return self.available_latent_frames * self.temporal_ratio

    def _segment_raw_visible_spans(self):
        raw_start = 0
        visible_start = 0
        for latent_frames, pixel_frames in zip(self.push_latent_frames, self.push_pixel_frames):
            raw_end = raw_start + latent_frames * self.temporal_ratio
            visible_end = visible_start + pixel_frames
            yield (raw_start, raw_start + pixel_frames, raw_end, visible_start, visible_end)
            raw_start = raw_end
            visible_start = visible_end

    def _visible_frames_before_raw(self, raw_end: int) -> int:
        raw_end = max(0, min(int(raw_end), self.available_raw_pixel_frames))
        visible = 0
        for (
            raw_start,
            raw_visible_end,
            _raw_segment_end,
            _visible_start,
            visible_end,
        ) in self._segment_raw_visible_spans():
            if raw_end <= raw_start:
                break
            visible += max(0, min(raw_end, raw_visible_end) - raw_start)
            if raw_end < raw_visible_end:
                break
            if raw_end >= raw_visible_end:
                visible = visible_end
        return visible

    def _visible_slice_from_raw(self, video: torch.Tensor, *, raw_start: int, raw_end: int) -> torch.Tensor:
        """Drop per-segment decoder padding from one raw-pixel tensor."""
        raw_start = int(raw_start)
        raw_end = int(raw_end)
        expected = raw_end - raw_start
        if expected < 0 or int(video.shape[2]) != expected:
            raise RuntimeError(
                f"PixelUMM raw preview range mismatch: range=[{raw_start}, {raw_end}), video={video.shape[2]}"
            )
        parts = []
        for (
            segment_raw_start,
            segment_visible_raw_end,
            _segment_raw_end,
            _segment_visible_start,
            _segment_visible_end,
        ) in self._segment_raw_visible_spans():
            overlap_start = max(raw_start, segment_raw_start)
            overlap_end = min(raw_end, segment_visible_raw_end)
            if overlap_end <= overlap_start:
                continue
            parts.append(video[:, :, overlap_start - raw_start : overlap_end - raw_start])
        if parts:
            return torch.cat(parts, dim=2)
        return video[:, :, :0]

    def _provisional_window_starts(self) -> list[int]:
        if self.model.tail_hold_latents and (not self.final_segment):
            length = self.available_latent_frames
            return [0] if length <= self.window else list(range(0, length, self.stride))
        return _window_starts(self.available_latent_frames, self.window, self.stride)

    def _decode_provisional_raw(self) -> torch.Tensor | None:
        """Decode the exact current tail from the raw stable boundary onward."""
        raw_start = self.committed_raw_pixel_frames
        raw_end = self.available_raw_pixel_frames
        if raw_end <= raw_start:
            assert self.buffer is not None
            if self.pending is not None:
                return self.pending[:, :, :0]
            return self.buffer.new_empty((0,))
        regular_decoded_latent_end = (
            self.next_window_start - self.stride + self.window if self.next_window_start > 0 else 0
        )
        regular_decoded_raw_end = regular_decoded_latent_end * self.temporal_ratio
        if raw_end <= regular_decoded_raw_end:
            if self.pending is not None:
                pending_end = self.pending_pixel_start + int(self.pending.shape[2])
                if not (self.pending_pixel_start <= raw_start and pending_end >= raw_end):
                    raise RuntimeError(
                        f"PixelUMM pending output does not cover the regular decoded range [{raw_start}, {raw_end}); pending=[{self.pending_pixel_start}, {pending_end})"
                    )
                return self.pending[
                    :, :, raw_start - self.pending_pixel_start : raw_end - self.pending_pixel_start
                ].contiguous()
            raise RuntimeError("PixelUMM session is missing pending regular-window output")
        assert self.buffer is not None
        starts = self._provisional_window_starts()
        committed_latent = raw_start // self.temporal_ratio
        first_index = next(
            (index for index, start in enumerate(starts) if start + self.window > committed_latent), len(starts) - 1
        )
        selected_starts = starts[first_index:]
        decoded = self._decode_windows(selected_starts)
        assert decoded is not None
        decoded_raw_start = selected_starts[0] * self.temporal_ratio
        local_start = raw_start - decoded_raw_start
        local_end = raw_end - decoded_raw_start
        if not 0 <= local_start <= local_end <= decoded.shape[2]:
            raise RuntimeError(
                f"PixelUMM provisional decode does not cover the requested raw range: decoded_start={decoded_raw_start}, requested=[{raw_start}, {raw_end}), decoded_frames={decoded.shape[2]}"
            )
        return decoded[:, :, local_start:local_end].contiguous()

    def _commit_until(self, stable_raw_until: int) -> torch.Tensor | None:
        stable_raw_until = int(stable_raw_until)
        if stable_raw_until <= self.committed_raw_pixel_frames:
            return None
        if self.pending is None:
            raise RuntimeError("PixelUMM session has no decoded pixels to commit")
        visible_start = self.committed_pixel_frames
        stable_local_end = stable_raw_until - self.pending_pixel_start
        if not 0 < stable_local_end <= self.pending.shape[2]:
            raise RuntimeError(
                f"Invalid PixelUMM stable-prefix boundary: local_end={stable_local_end}, pending={self.pending.shape[2]}"
            )
        stable_raw = self.pending[:, :, :stable_local_end].contiguous()
        stable = self._visible_slice_from_raw(stable_raw, raw_start=self.pending_pixel_start, raw_end=stable_raw_until)
        if self.to_cpu:
            stable = stable.cpu()
        if self.retain_history and stable.shape[2] > 0:
            self.committed.append((visible_start, stable))
        self.pending = self.pending[:, :, stable_local_end:].contiguous()
        self.pending_pixel_start = stable_raw_until
        self.committed_raw_pixel_frames = stable_raw_until
        self.committed_pixel_frames = self._visible_frames_before_raw(stable_raw_until)
        return stable

    def _decode_ready_regular_windows(self) -> torch.Tensor | None:
        starts = []
        while self.next_window_start + self.window <= self.available_latent_frames:
            starts.append(self.next_window_start)
            self.next_window_start += self.stride
        if not starts:
            return None
        decoded = self._decode_windows(starts)
        assert decoded is not None
        self._append_decoded_windows(decoded, first_window_start=starts[0])
        tail_start = self.available_latent_frames - self.window
        stable_latent_frames = (
            tail_start + 1 if tail_start >= 0 and tail_start % self.stride == 0 else max(0, tail_start)
        )
        return self._commit_until(stable_latent_frames * self.temporal_ratio)

    @property
    def stable_pixel_frames(self) -> int:
        return self.committed_pixel_frames

    @property
    def withheld_pixel_frames(self) -> int:
        return self.available_pixel_frames - self.committed_pixel_frames

    def slice_pixels(self, start: int, end: int) -> torch.Tensor:
        """Read a future-stable pixel range from retained history."""
        if not self.retain_history:
            raise RuntimeError("PixelUMM streaming history retention is disabled for this session.")
        if not 0 <= start < end <= self.committed_pixel_frames:
            raise ValueError(
                f"Requested PixelUMM pixels are not stable yet: range=[{start}, {end}), stable={self.committed_pixel_frames}."
            )
        selected = []
        cursor = start
        for part_start, part in self.committed:
            part_end = part_start + int(part.shape[2])
            overlap_start = max(start, part_start)
            overlap_end = min(end, part_end)
            if overlap_end <= overlap_start:
                continue
            if overlap_start != cursor:
                raise RuntimeError(
                    f"PixelUMM streaming output has a pixel gap: cursor={cursor}, next={overlap_start}."
                )
            selected.append(part[:, :, overlap_start - part_start : overlap_end - part_start])
            cursor = overlap_end
        if cursor != end or not selected:
            raise RuntimeError(
                f"PixelUMM streaming output does not cover requested range: [{start}, {end}), covered_to={cursor}."
            )
        return torch.cat(selected, dim=2)

    @torch.no_grad()
    def push(self, latent: torch.Tensor, *, pixel_frames: int | None = None) -> torch.Tensor:
        """Append one arbitrary-length segment and return newly stable pixels."""
        if self.finished:
            raise RuntimeError("Cannot push PixelUMM latents after finish().")
        if latent.ndim != 5 or latent.shape[2] <= 0:
            raise ValueError(f"expected non-empty [B,C,T,H,W], got {tuple(latent.shape)}")
        latent_frames = int(latent.shape[2])
        segment_pixel_frames = latent_frames * self.temporal_ratio if pixel_frames is None else int(pixel_frames)
        if segment_pixel_frames <= 0:
            raise ValueError("pixel_frames must be positive")
        min_pixel_frames = (latent_frames - 1) * self.temporal_ratio + 1
        max_pixel_frames = latent_frames * self.temporal_ratio
        if not min_pixel_frames <= segment_pixel_frames <= max_pixel_frames:
            raise ValueError(
                f"PixelUMM streaming segment pixel length is inconsistent with its latent length: latent_frames={latent_frames}, pixel_frames={segment_pixel_frames}, expected_range=[{min_pixel_frames}, {max_pixel_frames}]."
            )
        previous_stable = self.committed_pixel_frames
        self.buffer = latent.contiguous() if self.buffer is None else torch.cat([self.buffer, latent], dim=2)
        self.available_latent_frames += latent_frames
        self.available_pixel_frames += segment_pixel_frames
        self.push_latent_frames.append(latent_frames)
        self.push_pixel_frames.append(segment_pixel_frames)
        stable = self._decode_ready_regular_windows()
        self.push_stable_pixel_frames.append(self.committed_pixel_frames)
        if stable is None or self.committed_pixel_frames <= previous_stable:
            assert self.buffer is not None
            return self.buffer.new_empty((0,))
        return stable

    @torch.no_grad()
    def push_preview(self, latent: torch.Tensor, *, pixel_frames: int | None = None) -> StreamingPreviewUpdate:
        """Append a segment and publish its complete provisional preview.

        `push()` publishes only future-stable pixels. Interactive web playback instead needs the complete segment
        immediately. This method additionally exposes the current provisional overlap. On the next push, the preceding
        provisional tail is merged with the newly decoded cross-boundary PixelUMM window and returned again for
        in-place replacement.

        Segment boundaries do not need to align to the PixelUMM stride. Callers copy the returned tensor over the
        preceding provisional tail, then append the remaining new frames.
        """
        previous_available = int(self.available_pixel_frames)
        previous_stable = int(self.committed_pixel_frames)
        stable = self.push(latent, pixel_frames=pixel_frames)
        replace_pixel_frames = previous_available - previous_stable
        pixel_start = previous_stable
        pixel_end = int(self.available_pixel_frames)
        provisional_pixel_frames = pixel_end - int(self.committed_pixel_frames)
        provisional_raw = None
        if provisional_pixel_frames > 0:
            provisional_raw = self._decode_provisional_raw()
        parts = []
        if stable.numel() > 0:
            parts.append(stable)
        if provisional_pixel_frames > 0:
            assert provisional_raw is not None
            pending = self._visible_slice_from_raw(
                provisional_raw, raw_start=self.committed_raw_pixel_frames, raw_end=self.available_raw_pixel_frames
            )
            if self.to_cpu:
                pending = pending.cpu()
            parts.append(pending)
        if not parts:
            assert self.buffer is not None
            video = self.buffer.new_empty((0,))
        elif len(parts) == 1:
            video = parts[0]
        else:
            video = torch.cat(parts, dim=2)
        expected = pixel_end - pixel_start
        if int(video.shape[2]) != expected:
            raise RuntimeError(
                f"PixelUMM preview update has the wrong pixel range: start={pixel_start}, end={pixel_end}, video={video.shape[2]}"
            )
        return StreamingPreviewUpdate(
            video=video,
            pixel_start=pixel_start,
            pixel_end=pixel_end,
            replace_pixel_frames=replace_pixel_frames,
            stable_pixel_frames=int(self.committed_pixel_frames),
            provisional_pixel_frames=provisional_pixel_frames,
        )

    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        """Publish the withheld tail and close the append-only timeline."""
        if self.finished:
            raise RuntimeError("PixelUMM streaming decode is already finished.")
        if self.buffer is None:
            raise RuntimeError("Cannot finish an empty PixelUMM streaming session.")
        previous_stable = self.committed_pixel_frames
        provisional_raw = self._decode_provisional_raw()
        assert provisional_raw is not None
        stable = self._visible_slice_from_raw(
            provisional_raw, raw_start=self.committed_raw_pixel_frames, raw_end=self.available_raw_pixel_frames
        )
        if self.to_cpu:
            stable = stable.cpu()
        if self.retain_history and stable.shape[2] > 0:
            self.committed.append((previous_stable, stable))
        if self.pending is not None:
            self.pending = self.pending[:, :, :0]
        self.pending_pixel_start = self.available_raw_pixel_frames
        self.committed_raw_pixel_frames = self.available_raw_pixel_frames
        self.committed_pixel_frames = self.available_pixel_frames
        self.finished = True
        if self.committed_pixel_frames <= previous_stable:
            assert self.buffer is not None
            return self.buffer.new_empty((0,))
        return stable

    @torch.no_grad()
    def rollback_last_push(self) -> None:
        """Discard the latest segment and rebuild the exact stable prefix."""
        if self.finished:
            raise RuntimeError("Cannot roll back a finalized PixelUMM session.")
        if not self.push_latent_frames:
            raise RuntimeError("There is no PixelUMM streaming segment to roll back.")
        latent_frames = self.push_latent_frames.pop()
        self.push_pixel_frames.pop()
        self.push_stable_pixel_frames.pop()
        assert self.buffer is not None
        keep = self.available_latent_frames - latent_frames
        retained = self.buffer[:, :, :keep].contiguous() if keep > 0 else None
        retained_latent_frames = list(self.push_latent_frames)
        retained_pixel_frames = list(self.push_pixel_frames)
        retained_stable_pixel_frames = list(self.push_stable_pixel_frames)
        self.buffer = retained
        self.available_latent_frames = keep
        self.available_pixel_frames = sum(retained_pixel_frames)
        self.next_window_start = 0
        self.pending = None
        self.pending_pixel_start = 0
        self.committed = []
        self.committed_raw_pixel_frames = 0
        self.committed_pixel_frames = 0
        self.push_latent_frames = retained_latent_frames
        self.push_pixel_frames = retained_pixel_frames
        self.push_stable_pixel_frames = retained_stable_pixel_frames
        if retained is not None:
            self._decode_ready_regular_windows()
        if self.push_stable_pixel_frames and self.push_stable_pixel_frames[-1] != self.committed_pixel_frames:
            raise RuntimeError(
                f"Rebuilt PixelUMM stable prefix differs after rollback: expected={self.push_stable_pixel_frames[-1]}, rebuilt={self.committed_pixel_frames}."
            )

    def full_video(self) -> torch.Tensor:
        """Return the finalized full video from retained history."""
        if not self.finished:
            raise RuntimeError("PixelUMM streaming decode is not finished")
        if not self.retain_history:
            raise RuntimeError("PixelUMM streaming history retention is disabled for this session.")
        return self.slice_pixels(0, self.available_pixel_frames)

    def _decode_windows(self, starts):
        video = self.model._decode_windows(
            self.buffer, starts, self.seed, pad_short=bool(self.model.tail_hold_latents) and (not self.final_segment)
        )
        return video.cpu() if self.to_cpu else video
