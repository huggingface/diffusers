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

"""Kandinsky 6 SR transformer Diffusers component."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin
from .transformer_kandinsky6 import Kandinsky6AttnProcessor


class Kandinsky6SRAttentionProcessor(Kandinsky6AttnProcessor):
    """Diffusers attention processor used by the SR transformer."""

    def __call__(
        self,
        attn: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        *,
        sparse_params: dict[str, Any] | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        if sparse_params is not None:
            return attn.attention_flex(query, key, value, sparse_params=sparse_params)

        if getattr(attn, "cached_k", None) is not None:
            key, value, cu_seqlens_q, cu_seqlens_k = self.assemble_cached_attention_inputs(
                key, value, attn.cached_k, attn.cached_v, cu_seqlens_q, attn.cached_cu_seqlens
            )
        if getattr(attn, "return_kv", False):
            attn.cached_k, attn.cached_v, attn.cached_cu_seqlens = key, value, cu_seqlens_k

        outputs = []
        batch_size = cu_seqlens_q.numel() - 1
        for index in range(batch_size):
            query_start, query_end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
            key_start, key_end = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
            outputs.append(
                dispatch_attention_fn(
                    query[query_start:query_end].unsqueeze(0),
                    key[key_start:key_end].unsqueeze(0),
                    value[key_start:key_end].unsqueeze(0),
                    attn_mask=self._packed_attention_mask(
                        attn_mask,
                        index,
                        batch_size,
                        query_end - query_start,
                        key_end - key_start,
                    ),
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )[0]
            )
        return torch.cat(outputs, dim=0).flatten(-2, -1)

    @staticmethod
    def assemble_cached_attention_inputs(
        new_key: Tensor,
        new_value: Tensor,
        cached_key: Tensor,
        cached_value: Tensor,
        cu_seqlens_new: Tensor,
        cached_cu_seqlens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Merge cached and new K/V into packed varlen attention inputs."""
        num_seqs = cu_seqlens_new.numel() - 1
        key_parts: list[Tensor] = []
        value_parts: list[Tensor] = []
        for index in range(num_seqs):
            cache_start, cache_end = int(cached_cu_seqlens[index]), int(cached_cu_seqlens[index + 1])
            new_start, new_end = int(cu_seqlens_new[index]), int(cu_seqlens_new[index + 1])
            key_parts.append(cached_key[cache_start:cache_end])
            key_parts.append(new_key[new_start:new_end])
            value_parts.append(cached_value[cache_start:cache_end])
            value_parts.append(new_value[new_start:new_end])

        key = torch.cat(key_parts, dim=0)
        value = torch.cat(value_parts, dim=0)
        merged_lens = torch.diff(cached_cu_seqlens) + torch.diff(cu_seqlens_new)
        cu_seqlens_k = torch.cat([cu_seqlens_new.new_zeros(1), torch.cumsum(merged_lens, dim=0)]).to(
            cu_seqlens_new.dtype
        )
        return key, value, cu_seqlens_new, cu_seqlens_k

    @staticmethod
    def _packed_attention_mask(
        attn_mask: Tensor | None,
        index: int,
        batch_size: int,
        query_length: int,
        key_length: int,
    ) -> Tensor | None:
        if attn_mask is None:
            return None
        if attn_mask.ndim == 1:
            return attn_mask[:key_length].unsqueeze(0)
        if attn_mask.ndim == 2:
            if attn_mask.shape[0] in (1, batch_size):
                return attn_mask[index : index + 1, :key_length]
            return attn_mask[:query_length, :key_length]
        if attn_mask.shape[0] in (1, batch_size):
            attn_mask = attn_mask[index : index + 1]
        return attn_mask[..., :query_length, :key_length]


# Side of the local 8x8 token block used by fractal (NABLA) attention.
# Independent of the VAE spatial compression — do not swap for VAE_SPATIAL_FACTOR.
FRACTAL_BLOCK_SIZE = 8


def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    """Compute sinusoidal frequency schedule for rotary embeddings.

    Args:
        dim: Embedding dimension.
        max_period: Maximum period of the sinusoidal frequencies.

    Returns:
        Frequency tensor of shape ``(dim,)``.
    """
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def fractal_flatten(
    x: Tensor,
    rope: Tensor,
    cu_seqlens: Tensor,
    shape: tuple[int, int, int],
    *,
    fractal: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten spatial dimensions with optional fractal (local block) patching.

    Args:
        x: Input tensor with spatial dimensions.
        rope: Rotary position embedding tensor matching ``x`` layout.
        cu_seqlens: Cumulative sequence lengths (per-frame counts).
        shape: Spatial shape as ``(length, height, width)``.
        fractal: If True, apply local 8x8 block patching before flattening.

    Returns:
        Tuple of (flattened ``x``, flattened ``rope``, scaled ``cu_seqlens``).
    """
    _, height, width = shape
    if fractal:
        block = FRACTAL_BLOCK_SIZE
        x = local_patching(x, shape, (1, block, block), dim=0)
        rope = local_patching(rope, shape, (1, block, block), dim=0)
        x = x.flatten(0, 1)
        rope = rope.flatten(0, 1)
    else:
        x = x.flatten(0, 2)
        rope = rope.flatten(0, 2)
    cu_seqlens = cu_seqlens * (height * width)
    return x, rope, cu_seqlens


def fractal_unflatten(
    x: Tensor,
    cu_seqlens: Tensor,
    shape: tuple[int, int, int],
    *,
    fractal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Unflatten spatial dimensions, inverse of ``fractal_flatten``.

    Args:
        x: Flattened input tensor.
        cu_seqlens: Scaled cumulative sequence lengths.
        shape: Target spatial shape as ``(length, height, width)``.
        fractal: If True, reverse local 8x8 block patching.

    Returns:
        Tuple of (unflattened ``x``, rescaled ``cu_seqlens``).
    """
    _, height, width = shape
    if fractal:
        block = FRACTAL_BLOCK_SIZE
        x = x.reshape(-1, block**2, *x.shape[1:])
        x = local_merge(x, shape, (1, block, block), dim=0)
    else:
        x = x.reshape(*shape, *x.shape[1:])
    cu_seqlens = cu_seqlens // (height * width)
    return x, cu_seqlens


def local_patching(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    """Rearrange tensor into local spatial patches.

    Groups neighboring elements along each spatial axis into patches,
    producing a tensor with patch-count and intra-patch dimensions.

    Args:
        x: Input tensor with spatial dimensions starting at ``dim``.
        shape: Spatial shape as ``(duration, height, width)``.
        group_size: Patch size per axis ``(g1, g2, g3)``.
        dim: First spatial dimension index.

    Returns:
        Patched tensor with shape ``(..., num_patches, patch_elems, ...)``.
    """
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration // g1, g1, height // g2, g2, width // g3, g3, *x.shape[dim + 3 :])
    x = x.permute(
        *range(len(x.shape[:dim])), dim, dim + 2, dim + 4, dim + 1, dim + 3, dim + 5, *range(dim + 6, len(x.shape))
    )
    return x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)


def local_merge(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    """Reverse local patching, restoring the original spatial layout.

    Args:
        x: Patched tensor with ``(num_patches, patch_elems)`` at ``dim``.
        shape: Original spatial shape as ``(duration, height, width)``.
        group_size: Patch size per axis ``(g1, g2, g3)``.
        dim: First spatial dimension index.

    Returns:
        Tensor with restored spatial dimensions.
    """
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration // g1, height // g2, width // g3, g1, g2, g3, *x.shape[dim + 2 :])
    x = x.permute(
        *range(len(x.shape[:dim])), dim, dim + 3, dim + 1, dim + 4, dim + 2, dim + 5, *range(dim + 6, len(x.shape))
    )
    return x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)


@torch.compile()
@torch.no_grad()
def fast_sta_nabla(T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3, device: str = "cuda") -> Tensor:
    """Build a static spatiotemporal neighborhood attention mask.

    Each token attends to spatial and temporal neighbors within
    the given window sizes.

    Args:
        T: Temporal dimension (number of frames).
        H: Height dimension.
        W: Width dimension.
        wT: Temporal window size (odd).
        wH: Height window size (odd).
        wW: Width window size (odd).
        device: Device for the output tensor.

    Returns:
        Boolean mask of shape ``(T*H*W, T*H*W)``.
    """
    max_dim = max(T, H, W)
    r = torch.arange(0, max_dim, 1, dtype=torch.int16, device=device)
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = mat[:T, :T].flatten(), mat[:H, :H].flatten(), mat[:W, :W].flatten()
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
    sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
    return sta.reshape(T * H * W, T * H * W)


@torch.compile(dynamic=True)
@torch.no_grad()
def nablaT_v2_doc(
    q: Tensor,
    k: Tensor,
    seq: Tensor,
    T: int,
    H: int,
    W: int,
    *,
    wT: int = 3,
    wH: int = 3,
    wW: int = 3,
    thr: float = 0.9,
    add_sta: bool = True,
) -> BlockMask:
    """Build a dynamic sparse attention BlockMask with document boundaries.

    Estimates an approximate attention map from block-averaged queries and
    keys, then binarizes it (CDF cutoff) to select the most relevant blocks per query.

    Args:
        q: Query tensor of shape ``(B, heads, seq_len, dim)``.
        k: Key tensor of shape ``(B, heads, seq_len, dim)``.
        seq: Cumulative document lengths (e.g. [0, 31, 51, 66, 97] - video boundaries in seq_len).
        T: Temporal dimension.
        H: Height dimension.
        W: Width dimension.
        wT: Temporal window for static attention.
        wH: Height window for static attention.
        wW: Width window for static attention.
        thr: CDF threshold for binarization.
        add_sta: Whether to add static local attention.

    Returns:
        Sparse ``BlockMask`` with block size 64.
    """
    # Q/K are the authoritative execution tensors when this function is
    # reached through attention. Keep every mask intermediate on their device.
    device = q.device
    seq = seq.to(device=device)

    # Map estimation
    B, h, S, D = q.shape
    qa = q.reshape(B, h, S // 64, 64, D).mean(-2)
    ka = k.reshape(B, h, S // 64, 64, D).mean(-2).transpose(-2, -1)
    attn_map = qa @ ka

    d = torch.diff(seq)
    doc = (
        torch.eye(d.numel(), dtype=torch.bool, device=device)
        .repeat_interleave(d * H * W, dim=0)
        .repeat_interleave(d * H * W, dim=1)
    )
    attn_map += doc.log()
    attn_map = torch.softmax(attn_map / math.sqrt(D), dim=-1)

    # Map binarization
    vals, inds = attn_map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))

    if add_sta:
        sta = fast_sta_nabla(T, H, W, wT, wH, wW, device=device).unsqueeze_(0).unsqueeze_(0)
        mask = torch.logical_or(mask, sta)
    mask = torch.logical_and(mask, doc)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None)


# Neural network building blocks for the diffusion transformer.


# `mode="max-autotune-no-cudagraphs"` autotunes Triton kernels for CUDA; fall back to eager
# `flex_attention` on CPU/MPS (tests, non-CUDA inference) instead of attempting CUDA autotuning.
_compiled_flex = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs", dynamic=True)


def flex(query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tensor:
    if query.device.type == "cuda":
        return _compiled_flex(query, key, value, **kwargs)
    return flex_attention(query, key, value, **kwargs)


# Without FA3 the only varlen entry points available are FA2's
# ``flash_attn_varlen_*_func``, whose C++ binding declares ``max_seqlen_q``/
# ``max_seqlen_k`` as ``SymInt`` but does not support FakeTensors → dynamo
# can't trace through it.  Skip dynamo on the affected methods so the
# surrounding code in ``TransformerDecoderBlock.forward`` (norms, QKV
# projections, modulation, FFN) still gets ``@torch.compile``-fused while
# only the FA call runs in eager.  When FA3 is available the dedicated
# ``flash_attn_interface.flash_attn_varlen_func`` does register a custom op
# and traces cleanly, so this decorator becomes a no-op.


def apply_scale_shift_norm(norm: nn.Module, x: Tensor, scale: Tensor, shift: Tensor, idx: Tensor) -> Tensor:
    """Apply adaptive normalization with indexed scale and shift, forcing fp32 math."""
    with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=x.device.type == "cuda"):
        return norm(x) * (scale.index_select(0, idx) + 1.0) + shift.index_select(0, idx)


def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor, idx: Tensor) -> Tensor:
    """Add gated residual output to input using indexed gate values, forcing fp32 math."""
    with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=x.device.type == "cuda"):
        return x + gate.index_select(0, idx) * out


def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """Apply rotary position embeddings to input tensor."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
        x_out = rope[..., 0] * x_[..., 0] + rope[..., 1] * x_[..., 1]
        return x_out.reshape(*x.shape)


class TimeEmbeddings(nn.Module):
    """Sinusoidal time step embeddings projected through a two-layer MLP."""

    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0) -> None:
        """Initialize time embeddings.

        Args:
            model_dim: Dimension of sinusoidal encoding. Must be even.
            time_dim: Output dimension after MLP projection.
            max_period: Maximum period for frequency computation.
        """
        super().__init__()
        if model_dim % 2 != 0:
            msg = "model_dim must be even"
            raise ValueError(msg)
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)

        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    def forward(self, time: Tensor) -> tuple[Tensor, Tensor]:
        """Encode time steps into embeddings.

        Args:
            time: Diffusion time steps.

        Returns:
            Tuple of (time embeddings, index tensor mapping tokens to embeddings).
        """
        with torch.autocast(device_type=time.device.type, dtype=torch.float32, enabled=time.device.type == "cuda"):
            args = torch.outer(time, self.freqs.to(device=time.device))
            time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        time_embed_idx = torch.arange(time_embed.shape[0], device=time_embed.device, dtype=torch.int32)
        return time_embed, time_embed_idx


class TextEmbeddings(nn.Module):
    """Linear projection with layer normalization for text encoder outputs."""

    def __init__(self, text_dim: int, model_dim: int) -> None:
        """Initialize text embeddings.

        Args:
            text_dim: Input dimension of text encoder hidden states.
            model_dim: Output dimension after projection.
        """
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed: Tensor) -> Tensor:
        """Project and normalize text embeddings.

        Args:
            text_embed: Raw text encoder hidden states.

        Returns:
            Projected and normalized text embeddings.
        """
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):
    """Patchify and project visual tokens into the transformer hidden space."""

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]) -> None:
        """Initialize visual embeddings.

        Args:
            visual_dim: Number of input visual channels.
            model_dim: Output dimension after projection.
            patch_size: Patch size as (temporal, height, width).
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def _patchify(self, x: Tensor, visual_cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
        """Shared patchification logic for both full input and LQ-only slices.

        Args:
            x: Input tensor of shape ``(duration, height, width, channels)``.
            visual_cu_seqlens: Cumulative sequence lengths.

        Returns:
            Tuple of (patchified tensor, updated cumulative sequence lengths).
        """
        if self.patch_size[0] > 1:
            idxs = torch.ones(x.shape[0], dtype=torch.int32, device=visual_cu_seqlens.device)
            idxs[visual_cu_seqlens[:-1]] += self.patch_size[0] - 1
            x = torch.repeat_interleave(x, idxs, dim=0)
            visual_cu_seqlens = visual_cu_seqlens + torch.arange(
                visual_cu_seqlens.shape[0], device=visual_cu_seqlens.device, dtype=torch.int32
            )

        duration, height, width, dim = x.shape
        # [T, H, W, C] -> [T/pt, H/ph, W/pw, pt*ph*pw*C]
        # Groups of (pt, ph, pw) neighboring pixels are concatenated into one token vector
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        visual_cu_seqlens = visual_cu_seqlens // self.patch_size[0]
        return x, visual_cu_seqlens

    def forward(self, x: Tensor, visual_cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
        """Patchify input and project to model dimension.

        Args:
            x: Visual input of shape ``(duration, height, width, channels)``.
            visual_cu_seqlens: Cumulative sequence lengths for packed sequences.

        Returns:
            ``(projected_patches, cu_seqlens)``.
        """
        patches, cu = self._patchify(x, visual_cu_seqlens)
        return self.in_layer(patches), cu


class RoPE1D(nn.Module):
    """1D rotary position embeddings for text tokens."""

    def __init__(self, dim: int, max_pos: int = 1024, max_period: float = 10000.0) -> None:
        """Initialize 1D rotary position embeddings.

        Args:
            dim: Embedding dimension. Must be even.
            max_pos: Maximum number of positions to precompute.
            max_period: Maximum period for frequency computation.
        """
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos: Tensor) -> Tensor:
        """Compute rotary embeddings for given positions.

        Args:
            pos: Position indices.

        Returns:
            Rotary embedding matrix with shape ``(*pos.shape, 1, dim//2, 2, 2)``.
        """
        with torch.autocast(device_type=pos.device.type, enabled=False):
            args = self.args[pos]
            rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
            rope = rope.view(*rope.shape[:-1], 2, 2)
            return rope.unsqueeze(-4)


class RoPE3D(nn.Module):
    """3D rotary position embeddings for visual tokens along temporal, height, and width axes."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ) -> None:
        """Initialize 3D rotary position embeddings.

        Args:
            axes_dims: Per-axis embedding dimensions (temporal, height, width).
            max_pos: Maximum positions per axis.
            max_period: Maximum period for frequency computation.
        """
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos, strict=False)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    def forward(
        self,
        shape: tuple[int, int, int],
        pos: tuple[Tensor, Tensor, Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        """Compute 3D rotary embeddings for given positions.

        Args:
            shape: Spatial dimensions as (duration, height, width).
            pos: Position indices per axis (temporal, height, width).
            scale_factor: Frequency scaling per axis.

        Returns:
            Rotary embedding matrix broadcast to (duration, height, width, 1, dim//2, 2, 2).
        """
        with torch.autocast(device_type=pos[0].device.type, enabled=False):
            duration, height, width = shape
            args_t = self.args_0[pos[0]] / scale_factor[0]
            args_h = self.args_1[pos[1]] / scale_factor[1]
            args_w = self.args_2[pos[2]] / scale_factor[2]

            args = torch.cat(
                [
                    args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                    args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                    args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
                ],
                dim=-1,
            )
            rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
            rope = rope.view(*rope.shape[:-1], 2, 2)
            return rope.unsqueeze(-4)


class Modulation(nn.Module):
    """Adaptive modulation layer producing scale, shift, and gate parameters from time embeddings."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int) -> None:
        """Initialize modulation layer.

        Args:
            time_dim: Input dimension of time embeddings.
            model_dim: Per-parameter output dimension.
            num_params: Number of modulation parameters to produce.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Compute modulation parameters from time embeddings.

        Args:
            x: Time embedding tensor.

        Returns:
            Concatenated modulation parameters.
        """
        with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=x.device.type == "cuda"):
            return self.out_layer(self.activation(x))


class MultiheadSelfAttention(nn.Module, AttentionModuleMixin):
    """Multi-head self-attention with flash attention and optional sparse flex attention."""

    _default_processor_cls = Kandinsky6SRAttentionProcessor
    _available_processors = [Kandinsky6SRAttentionProcessor]

    def __init__(self, num_channels: int, head_dim: int) -> None:
        """Initialize multi-head self-attention.

        Args:
            num_channels: Total number of channels. Must be divisible by head_dim.
            head_dim: Dimension per attention head.
        """
        super().__init__()
        if num_channels % head_dim != 0:
            msg = "num_channels must be divisible by head_dim"
            raise ValueError(msg)
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

        self.cached_k: Tensor | None = None
        self.cached_v: Tensor | None = None
        self.cached_cu_seqlens: Tensor | None = None
        self.return_kv = False
        self.set_processor(self._default_processor_cls())

    def get_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project input into query, key, and value tensors.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (query, key, value) reshaped to ``(seq_len, num_heads, head_dim)``.
        """
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]  # for TP compatibility
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RMS normalization to query and key.

        Args:
            q: Query tensor.
            k: Key tensor.

        Returns:
            Tuple of (normalized query, normalized key).
        """
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def reset_kv_cache(self) -> None:
        """Clear the rolling KV-cache and stop emitting K/V."""
        self.cached_k = None
        self.cached_v = None
        self.cached_cu_seqlens = None
        self.return_kv = False

    def attention_flex(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sparse_params: dict[str, Any] | None = None,
        *,
        return_sparsity: bool = False,
    ) -> Tensor | tuple[Tensor, float]:
        """Compute sparse self-attention using flex attention with block masks.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            sparse_params: Sparse attention configuration.
            return_sparsity: Whether to return sparsity percentage.

        Returns:
            Attention output, or tuple of (output, sparsity percentage) if requested.
        """
        if self.cached_k is not None:
            msg = "KV-cache is not supported on the flex/NABLA attention path; only dense flash attention supports it."
            raise NotImplementedError(msg)

        query = query.unsqueeze(0).transpose(1, 2).contiguous()
        key = key.unsqueeze(0).transpose(1, 2).contiguous()
        value = value.unsqueeze(0).transpose(1, 2).contiguous()

        t, h, w = sparse_params["visual_shape"]
        # Token grid -> fractal 8x8 block grid; the block size matches
        # fractal_flatten and is unrelated to the VAE spatial compression.
        h, w = h // FRACTAL_BLOCK_SIZE, w // FRACTAL_BLOCK_SIZE
        visual_seqlens = sparse_params["visual_seqlens"].to(device=query.device)
        block_mask = nablaT_v2_doc(
            query,
            key,
            visual_seqlens,
            t,
            h,
            w,
            wT=sparse_params["wT"],
            wH=sparse_params["wH"],
            wW=sparse_params["wW"],
            thr=sparse_params["P"],
            add_sta=sparse_params["add_sta"],
        )
        out = (
            flex(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={"BLOCK_M": 64, "BLOCK_N": 64},
            )
            .transpose(1, 2)
            .squeeze(0)
            .contiguous()
        )
        out = out.flatten(-2, -1)

        if return_sparsity:
            sparsity = 100.0 * (1 - (1 - block_mask.sparsity() / 100) * (sparse_params["visual_seqlens"].shape[0] - 1))
            return out, sparsity
        return out

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        sparse_params: dict[str, Any] | None = None,
    ) -> Tensor:
        """Run self-attention with rotary embeddings.

        Args:
            x: Input tensor.
            rope: Rotary position embeddings.
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            sparse_params: Optional sparse attention parameters.

        Returns:
            Self-attention output.
        """
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        out = self.processor(self, query, key, value, cu_seqlens, cu_seqlens, sparse_params=sparse_params)

        return self.out_layer(out)


class MultiheadCrossAttention(nn.Module, AttentionModuleMixin):
    """Multi-head cross-attention with flash attention."""

    _default_processor_cls = Kandinsky6SRAttentionProcessor
    _available_processors = [Kandinsky6SRAttentionProcessor]

    def __init__(self, num_channels: int, head_dim: int) -> None:
        """Initialize multi-head cross-attention.

        Args:
            num_channels: Total number of channels. Must be divisible by head_dim.
            head_dim: Dimension per attention head.
        """
        super().__init__()
        if num_channels % head_dim != 0:
            msg = "num_channels must be divisible by head_dim"
            raise ValueError(msg)
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)
        self.set_processor(self._default_processor_cls())

    def get_qkv(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project input and condition into query, key, and value tensors.

        Args:
            x: Input tensor for query projection.
            cond: Conditioning tensor for key and value projections.

        Returns:
            Tuple of (query, key, value) reshaped to ``(seq_len, num_heads, head_dim)``.
        """
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape, cond_shape = query.shape[:-1], key.shape[:-1]  # for TP compatibility
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RMS normalization to query and key.

        Args:
            q: Query tensor.
            k: Key tensor.

        Returns:
            Tuple of (normalized query, normalized key).
        """
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def forward(self, x: Tensor, cond: Tensor, cu_seqlens: Tensor, cond_cu_seqlens: Tensor) -> Tensor:
        """Run cross-attention between input and conditioning.

        Args:
            x: Input tensor for query computation.
            cond: Conditioning tensor for key and value computation.
            cu_seqlens: Cumulative sequence lengths for input sequences.
            cond_cu_seqlens: Cumulative sequence lengths for conditioning sequences.

        Returns:
            Cross-attention output.
        """
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.processor(self, query, key, value, cu_seqlens, cond_cu_seqlens)
        return self.out_layer(out)


class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, dim: int, ff_dim: int) -> None:
        """Initialize feed-forward network.

        Args:
            dim: Input and output dimension.
            ff_dim: Hidden layer dimension.
        """
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        return self.out_layer(self.activation(self.in_layer(x)))


class OutLayer(nn.Module):
    """Final output layer that unpatchifies and projects visual tokens to pixel space."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]) -> None:
        """Initialize output layer.

        Args:
            model_dim: Input dimension from the transformer.
            time_dim: Dimension of time embeddings for modulation.
            visual_dim: Number of output visual channels.
            patch_size: Patch size as (temporal, height, width).
        """
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(
        self,
        visual_embed: Tensor,
        _text_embed: Tensor,
        time_embed: Tensor,
        visual_cu_seqlens: Tensor,
        time_embed_idx: Tensor,
    ) -> Tensor:
        """Unpatchify and project visual embeddings to output channels.

        Args:
            visual_embed: Visual token embeddings from the transformer.
            _text_embed: Text embeddings (unused, reserved for interface compatibility).
            time_embed: Time step embeddings for adaptive modulation.
            visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
            time_embed_idx: Index mapping visual tokens to their time embeddings.

        Returns:
            Denoised visual output tensor.
        """
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm, visual_embed, scale[:, None, None], shift[:, None, None], time_embed_idx
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        duration, height, width, _dim = x.shape
        x = (
            x.view(
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        visual_cu_seqlens = visual_cu_seqlens * self.patch_size[0]

        if self.patch_size[0] > 1:
            idxs = torch.ones(duration * self.patch_size[0], dtype=torch.int32, device=visual_cu_seqlens.device)
            idxs[visual_cu_seqlens[:-1]] -= self.patch_size[0] - 1
            x = torch.repeat_interleave(x, idxs, dim=0)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention, feed-forward, and adaptive modulation."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int) -> None:
        """Initialize encoder block layers.

        Args:
            model_dim: Hidden dimension of the transformer.
            time_dim: Dimension of time step embeddings.
            ff_dim: Inner dimension of the feed-forward network.
            head_dim: Per-head dimension for self-attention.
        """
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(
        self,
        x: Tensor,
        time_embed: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        time_embed_idx: Tensor,
    ) -> Tensor:
        """Run self-attention and feed-forward with time-conditioned modulation.

        Args:
            x: Input text embeddings.
            time_embed: Time step embeddings for modulation.
            rope: Rotary position embeddings.
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            time_embed_idx: Index mapping tokens to their time embeddings.

        Returns:
            Modulated text embeddings.
        """
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.self_attention(out, rope, cu_seqlens)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.feed_forward(out)
        return apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with self-attention, cross-attention, and feed-forward."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        *,
        use_text: bool = True,
    ) -> None:
        """Initialize decoder block layers.

        Args:
            model_dim: Hidden dimension of the transformer.
            time_dim: Dimension of time step embeddings.
            ff_dim: Inner dimension of the feed-forward network.
            head_dim: Per-head dimension for attention layers.
            use_text: When False, the block has no text cross-attention: no
                ``cross_attention``/``cross_attention_norm`` modules and the
                modulation produces 6 params (self-attn + FFN) instead of 9.
                Used by the text-free SR variant.
        """
        super().__init__()
        self.use_text = use_text
        self.visual_modulation = Modulation(time_dim, model_dim, 9 if use_text else 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        if use_text:
            self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
            self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(
        self,
        visual_embed: Tensor,
        text_embed: Tensor,
        time_embed: Tensor,
        rope: Tensor,
        visual_cu_seqlens: Tensor,
        text_cu_seqlens: Tensor,
        time_embed_idx: Tensor,
        sparse_params: dict[str, Any] | None,
    ) -> Tensor:
        """Run self-attention, cross-attention, and feed-forward with modulation.

        Args:
            visual_embed: Visual token embeddings.
            text_embed: Encoded text embeddings used as cross-attention context.
            time_embed: Time step embeddings for modulation.
            rope: Rotary position embeddings for visual tokens.
            visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
            text_cu_seqlens: Cumulative sequence lengths for text tokens.
            time_embed_idx: Index mapping visual tokens to their time embeddings.
            sparse_params: Optional sparse attention parameters.

        Returns:
            Updated visual embeddings.
        """
        dtype = visual_embed.dtype
        if self.use_text:
            self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        else:
            self_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 2, dim=-1)
            cross_attn_params = None

        # --- Self-attention ---
        # Norm -> modulate by (scale + 1, shift) -> SelfAttn -> gate
        shift_t, scale_t, gate_t = torch.chunk(self_attn_params, 3, dim=-1)
        with torch.autocast(
            device_type=visual_embed.device.type, dtype=torch.float32, enabled=visual_embed.device.type == "cuda"
        ):
            normed = self.self_attention_norm(visual_embed)
            scale = scale_t.index_select(0, time_embed_idx)
            shift = shift_t.index_select(0, time_embed_idx)
            visual_out = normed * (scale + 1.0) + shift
        visual_out = visual_out.type_as(visual_embed)
        visual_out = self.self_attention(visual_out, rope, visual_cu_seqlens, sparse_params)
        with torch.autocast(
            device_type=visual_embed.device.type, dtype=torch.float32, enabled=visual_embed.device.type == "cuda"
        ):
            visual_embed = visual_embed + gate_t.index_select(0, time_embed_idx) * visual_out
        visual_embed = visual_embed.to(dtype=dtype)

        # --- Cross-attention --- (skipped when text-free)
        # Norm -> modulate by (scale + 1, shift) -> CrossAttn -> gate
        if self.use_text:
            shift_t, scale_t, gate_t = torch.chunk(cross_attn_params, 3, dim=-1)
            with torch.autocast(
                device_type=visual_embed.device.type, dtype=torch.float32, enabled=visual_embed.device.type == "cuda"
            ):
                normed = self.cross_attention_norm(visual_embed)
                scale = scale_t.index_select(0, time_embed_idx)
                shift = shift_t.index_select(0, time_embed_idx)
                visual_out = normed * (scale + 1.0) + shift
            visual_out = visual_out.type_as(visual_embed)
            visual_out = self.cross_attention(visual_out, text_embed, visual_cu_seqlens, text_cu_seqlens)
            with torch.autocast(
                device_type=visual_embed.device.type, dtype=torch.float32, enabled=visual_embed.device.type == "cuda"
            ):
                visual_embed = visual_embed + gate_t.index_select(0, time_embed_idx) * visual_out
            visual_embed = visual_embed.to(dtype=dtype)

        # --- Feed-forward ---
        # Norm -> modulate by (scale + 1, shift) -> FF -> gate
        shift_t, scale_t, gate_t = torch.chunk(ff_params, 3, dim=-1)
        with torch.autocast(
            device_type=visual_embed.device.type, dtype=torch.float32, enabled=visual_embed.device.type == "cuda"
        ):
            normed = self.feed_forward_norm(visual_embed)
            scale = scale_t.index_select(0, time_embed_idx)
            shift = shift_t.index_select(0, time_embed_idx)
            visual_out = normed * (scale + 1.0) + shift
        visual_out = visual_out.type_as(visual_embed)
        visual_out = self.feed_forward(visual_out)
        with torch.autocast(
            device_type=visual_embed.device.type, dtype=torch.float32, enabled=visual_embed.device.type == "cuda"
        ):
            visual_embed = visual_embed + gate_t.index_select(0, time_embed_idx) * visual_out
        return visual_embed.to(dtype=dtype)


class Kandinsky6SRTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    """3D diffusion transformer with text-conditioned visual generation for Kandinsky 6 super-resolution.

    Unlike `Kandinsky6Transformer3DModel` (the TI2VA transformer), which batches over a padded
    `(B, T, H, W, C)` tensor, this transformer packs variable-length visual/text sequences into a single
    sequence per axis using `cu_seqlens` (the flash-attention varlen convention). This lets the SR pipeline
    batch many spatial tiles per forward pass without padding overhead. Visual tokens are processed through
    encoder (text) and decoder (visual) transformer blocks with RoPE, adaptive modulation, and optional
    NABLA sparse attention.

    Args:
        in_visual_dim (`int`, *optional*, defaults to 4): Number of input visual channels.
        in_text_dim (`int`, *optional*, defaults to 3584): Dimension of text encoder hidden states.
        in_text_dim2 (`int`, *optional*, defaults to 768): Dimension of pooled text embeddings.
        time_dim (`int`, *optional*, defaults to 512): Dimension of time step embeddings.
        out_visual_dim (`int`, *optional*, defaults to 4): Number of output visual channels.
        patch_size (`tuple[int, int, int]`, *optional*, defaults to `(1, 2, 2)`): Patch size as
            (temporal, height, width).
        model_dim (`int`, *optional*, defaults to 2048): Hidden dimension of the transformer.
        ff_dim (`int`, *optional*, defaults to 5120): Inner dimension of feed-forward networks.
        num_text_blocks (`int`, *optional*, defaults to 2): Number of text encoder blocks.
        num_visual_blocks (`int`, *optional*, defaults to 32): Number of visual decoder blocks.
        axes_dims (`tuple[int, int, int]`, *optional*, defaults to `(16, 24, 24)`): RoPE dimension split
            per axis (temporal, height, width).
        visual_cond (`bool`, *optional*, defaults to `False`): Whether to use visual conditioning input.
        instruct_type (`str`, *optional*): Conditioning mode: `"noise"`, `"channel"`, `"hybrid"`, or
            `"hybrid_anchor"` (the mode Kandinsky 6 SR ships with).
        attention_params (`dict`, *optional*): Per-resolution attention configuration, keyed by base tile
            resolution; each entry selects dense (`type="flash"`) or sparse NABLA (`type="nabla"`) attention.
        use_motion_score (`bool`, *optional*, defaults to `False`): Whether to add motion score embeddings
            to time conditioning.
        use_text (`bool`, *optional*, defaults to `True`): When `False`, builds a text-free model with no
            text encoder, text RoPE/blocks, pooled-text conditioning, or per-block cross-attention.
        sr_params (`dict`, *optional*): SR-specific configuration: `scheduler_scale`, `lq_noise_scale`,
            `lq_noise_type`, `lq_channel_noise_scale`, `cap_noise_timestep`, `fps`, `scale_factor` (RoPE
            frequency scaling per base resolution), and `visual_size` (base tile resolution). Read by
            [`Kandinsky6SRPipeline`] via the attributes this unpacks into.
        attribute_overrides (`dict`, *optional*): Arbitrary extra attributes set on the model after
            construction.
    """

    _no_split_modules = ["TransformerEncoderBlock", "TransformerDecoderBlock"]
    _repeated_blocks = ["TransformerEncoderBlock", "TransformerDecoderBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_visual_dim: int = 4,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 512,
        out_visual_dim: int = 4,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        model_dim: int = 2048,
        ff_dim: int = 5120,
        num_text_blocks: int = 2,
        num_visual_blocks: int = 32,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        *,
        visual_cond: bool = False,
        instruct_type: str | None = None,
        attention_params: dict[str, Any] | None = None,
        use_motion_score: bool = False,
        use_text: bool = True,
        sr_params: Mapping[str, Any] | None = None,
        attribute_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.instruct_type = instruct_type
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.attention_params = attention_params

        sr_params = dict(sr_params) if sr_params is not None else {}
        self.sr_params = sr_params
        self.scheduler_scale = float(sr_params.get("scheduler_scale", 5.0))
        self.lq_noise_scale = float(sr_params.get("lq_noise_scale", 0.7))
        self.lq_noise_type = sr_params.get("lq_noise_type", "ddpm")
        self.lq_channel_noise_scale = float(sr_params.get("lq_channel_noise_scale", 0.0))
        self.cap_noise_timestep = bool(sr_params.get("cap_noise_timestep", False))
        self.fps = int(sr_params.get("fps", 24))
        visual_size = sr_params.get("visual_size")
        self.visual_size = (
            int(visual_size[0] if isinstance(visual_size, (list, tuple)) else visual_size)
            if visual_size is not None
            else None
        )
        scale_factor = sr_params.get("scale_factor")
        self.scale_factor = (
            {int(key): tuple(float(value) for value in values) for key, values in scale_factor.items()}
            if scale_factor is not None
            else None
        )

        visual_embed_dim = (
            2 * in_visual_dim + 1
            if visual_cond or instruct_type in ("channel", "hybrid", "hybrid_anchor")
            else in_visual_dim
        )
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        if use_motion_score:
            self.motion_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.use_motion_score = use_motion_score

        self.use_text = use_text
        if use_text:
            self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
            self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        else:
            # Text-free: under the empty caption the pooled-text contribution
            # `pooled_text_embeddings("")` is a constant added to the time embedding.
            # build_text_free_dit.py bakes it into this bias; dropping it instead corrupts
            # the time conditioning, so the model still needs it (cross-attention does not).
            self.pooled_bias = nn.Parameter(torch.zeros(time_dim))
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        if use_text:
            self.text_rope_embeddings = RoPE1D(head_dim)
            self.text_transformer_blocks = nn.ModuleList(
                [TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_text_blocks)]
            )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, use_text=use_text)
                for _ in range(num_visual_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)
        for name, value in (attribute_overrides or {}).items():
            setattr(self, str(name), value)

        self.gradient_checkpointing = False

    def forward(
        self,
        x: Tensor,
        text_embed: Tensor,
        pooled_text_embed: Tensor,
        time: Tensor,
        visual_cu_seqlens: Tensor,
        text_cu_seqlens: Tensor,
        visual_rope_pos: tuple[Tensor, Tensor, Tensor],
        text_rope_pos: Tensor,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        motion_score: Tensor | None = None,
    ) -> Tensor:
        """Run the full diffusion transformer forward pass.

        Args:
            x: Noisy visual input tensor.
            text_embed: Raw text encoder hidden states.
            pooled_text_embed: Pooled text embeddings added to time conditioning.
            time: Diffusion time steps.
            visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
            text_cu_seqlens: Cumulative sequence lengths for text tokens.
            visual_rope_pos: 3D positional indices (temporal, height, width) for visual RoPE.
            text_rope_pos: 1D positional indices for text RoPE.
            scale_factor: RoPE frequency scaling per axis.
            sparse_params: Optional sparse/fractal attention parameters.
            motion_score: Optional motion score for temporal conditioning.

        Returns:
            Denoised visual output tensor.
        """
        checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing

        text_embed = self.text_embeddings(text_embed) if self.use_text else None
        time_embed, time_embed_idx = self.time_embeddings(time)
        if motion_score is not None and self.use_motion_score:
            ms_embed, _ = self.motion_embeddings(motion_score)
            time_embed = time_embed + ms_embed

        if self.use_text:
            time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        else:
            time_embed = time_embed + self.pooled_bias

        visual_embed, visual_cu_seqlens = self.visual_embeddings(x, visual_cu_seqlens)

        if self.use_text:
            text_rope = self.text_rope_embeddings(text_rope_pos)
            text_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(text_cu_seqlens), dim=0)
            for text_transformer_block in self.text_transformer_blocks:
                args = (text_embed, time_embed, text_rope, text_cu_seqlens, text_time_embed_idx)
                text_embed = (
                    self._gradient_checkpointing_func(text_transformer_block, *args)
                    if checkpoint
                    else text_transformer_block(*args)
                )

        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        # [T, 32, 32, 1792] -> [T*32*32, 1792]
        visual_embed, visual_rope, visual_cu_seqlens = fractal_flatten(
            visual_embed, visual_rope, visual_cu_seqlens, visual_shape, fractal=to_fractal
        )
        visual_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(visual_cu_seqlens), dim=0)

        for visual_transformer_block in self.visual_transformer_blocks:
            args = (
                visual_embed,
                text_embed,
                time_embed,
                visual_rope,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_time_embed_idx,
                sparse_params,
            )
            visual_embed = (
                self._gradient_checkpointing_func(visual_transformer_block, *args)
                if checkpoint
                else visual_transformer_block(*args)
            )
        visual_embed, visual_cu_seqlens = fractal_unflatten(
            visual_embed, visual_cu_seqlens, visual_shape, fractal=to_fractal
        )

        visual_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(visual_cu_seqlens), dim=0)
        return self.out_layer(visual_embed, text_embed, time_embed, visual_cu_seqlens, visual_time_embed_idx)


__all__ = ["Kandinsky6SRTransformer3DModel"]
