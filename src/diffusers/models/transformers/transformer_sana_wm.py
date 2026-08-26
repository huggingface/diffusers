# Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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
#
# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..activations import get_activation
from ..embeddings import get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Mlp(nn.Module):
    """Two-layer feed-forward block (`fc1` -> activation -> `fc2`)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, hidden_states: torch.Tensor, HW: tuple[int, int] | None = None) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop2(hidden_states)
        return hidden_states


class ShortConvolution(nn.Module):
    """Depthwise causal 1D convolution over the temporal axis.

    SANA-WM's GDN attention applies a short causal depthwise conv to Q/K/V before the linear-attention kernel. This is
    a self-contained PyTorch implementation of the `fla.modules.ShortConvolution` layer the reference implementation
    used (with `activation=None`), so the model needs no `fla-core` dependency and can be built on any device.

    Args:
        hidden_size (`int`): Number of channels (the conv is depthwise, one group per channel).
        kernel_size (`int`): Temporal kernel width.
        bias (`bool`, defaults to `False`): Whether to add a per-channel bias.
    """

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False, activation: str | None = None) -> None:
        super().__init__()
        if activation is not None:
            raise ValueError(f"SANA-WM only uses `activation=None` short convolutions, got {activation!r}.")
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        # Same parameter layout as the reference implementation: (C, 1, K).
        self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Apply the causal conv.

        Args:
            x (`torch.Tensor`): Input of shape `(batch, seq_len, hidden_size)`.

        Returns:
            `tuple[torch.Tensor, None]`: `(output, cache)`; the cache slot is kept for signature compatibility with the
            reference implementation but is unused for the bidirectional (non-streaming) forward SANA-WM runs.
        """
        seq_len = x.shape[1]
        # Left-pad by (K - 1) and drop the tail so output[t] only sees inputs <= t.
        y = F.conv1d(
            x.transpose(1, 2),
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
            groups=self.hidden_size,
            padding=self.kernel_size - 1,
        )[..., :seq_len]
        return y.transpose(1, 2), None


# ============================================================================
# Helpers (norms / chunk / weight utilities)
# ============================================================================


# NOTE: kept local instead of `..normalization.RMSNorm` because SANA-WM needs `scale_factor` (the released config
# initializes `attention_y_norm` at `ones * 0.01`) and normalizes fully in fp32, which the shared class does not do.
class RMSNorm(torch.nn.Module):
    """Root-mean-square layer norm with a scaled weight initialization.

    Args:
        dim (`int`): Size of the normalized dimension.
        scale_factor (`float`, defaults to 1.0): Initial value of every weight entry.
        eps (`float`, defaults to 1e-6): Added to the mean square for numerical stability.
        norm_dim (`int`, defaults to -1): Dimension to normalize over.
    """

    def __init__(self, dim: int, scale_factor: float = 1.0, eps: float = 1e-6, norm_dim: int = -1):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)
        self.norm_dim = norm_dim

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(self.norm_dim, keepdim=True) + self.eps)

    def forward(self, x):
        weight_shape = [1] * x.dim()
        weight_shape[self.norm_dim] = -1
        weight = self.weight.view(*weight_shape)
        return (weight * self._norm(x.float())).type_as(x)


def chunk_index_from_chunk_size(
    T: int,
    chunk_size: int,
    strategy: str = "uniform",
) -> List[int]:
    """Convert chunk_size to chunk_index list with a split strategy.

    Args:
        T: Number of latent frames.
        chunk_size: Base chunk size for the temporal dimension.
        strategy: Chunk split strategy. Supported values:
            - "uniform" (default): uniform chunks with optional remainder Example: T=21, chunk_size=4 →
              [0,4,8,12,16,20] → sizes [4,4,4,4,4,1]
            - "first_frame": first chunk is 1 frame, then uniform chunk_size Example: T=21, chunk_size=4 →
              [0,1,5,9,13,17] → sizes [1,4,4,4,4,4]
            - "first_plus_one": first chunk is chunk_size + 1, then uniform chunk_size Example: T=21, chunk_size=4 →
              [0,5,9,13,17] → sizes [5,4,4,4,4]

    Returns:
        List of chunk start indices (not including the final T).

    Raises:
        ValueError: If chunk_size or T are invalid, or strategy is unknown.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}.")

    if strategy is None:
        strategy = "uniform"
    strategy = str(strategy).lower()

    if strategy in ("uniform", "default"):
        indices = list(range(0, T, chunk_size))
        # Absorb small remainder into last chunk to avoid degenerate chunks
        # (e.g., causal_conv1d crashes on length=1 sequences).
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_frame", "first_frame_alone", "first_frame_only"):
        if T <= 1:
            return [0]
        indices = [0] + list(range(1, T, chunk_size))
        if len(indices) > 2 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_plus_one", "first_chunk_plus_one"):
        if T <= chunk_size + 1:
            return [0]
        indices = [0] + list(range(chunk_size + 1, T, chunk_size))
        # Absorb small remainder into last chunk to avoid degenerate chunks
        # (e.g., T_latent=41 with chunk_size=3 → last chunk would be 1 frame,
        # which crashes causal_conv1d). Merge it into the previous chunk instead.
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    raise ValueError(f"Unknown chunk_split_strategy '{strategy}'. Supported: uniform, first_frame, first_plus_one.")


def compute_chunk_sizes(chunk_index: List[int], T: int) -> List[int]:
    """Compute actual chunk sizes from chunk_index.

    Args:
        chunk_index: List of chunk start indices (e.g., [0, 4, 8, 12]).
        T: Total number of frames.

    Returns:
        List of chunk sizes (e.g., [4, 4, 4, 1] if T=13).

    Example:
        >>> compute_chunk_sizes([0, 4, 8, 12], T=13) [4, 4, 4, 1] >>> compute_chunk_sizes([0, 1, 5, 9], T=13) [1, 4, 4,
        4]
    """
    if not chunk_index:
        return []

    # Ensure chunk_index is clean
    chunk_index = [idx for idx in chunk_index if 0 <= idx < T]
    if not chunk_index:
        return []

    # Add T as the final boundary if not present
    if chunk_index[-1] != T:
        chunk_index = chunk_index + [T]

    # Compute sizes
    sizes = [chunk_index[i + 1] - chunk_index[i] for i in range(len(chunk_index) - 1)]
    return sizes


def is_uniform_chunking(
    chunk_index: List[int],
    T: int,
    chunk_size: int,
) -> bool:
    """Check if chunk_index represents uniform chunking.

    Returns True if all chunks are equal to chunk_size except possibly the last chunk which may be smaller (the
    remainder). This is the pattern that allows safe vectorized padding with: pad_t = chunk_size - (T % chunk_size).

    Uniform patterns (return True):
        - [0,4,8,12,16,20] with T=21, chunk_size=4 → sizes [4,4,4,4,4,1] ✓
        - [0,4,8,12,16] with T=20, chunk_size=4 → sizes [4,4,4,4,4] ✓
        - [0,4,8] with T=10, chunk_size=4 → sizes [4,4,2] ✓

    Non-uniform patterns (return False):
        - [0,1,5,9,13,17] with T=21, chunk_size=4 → sizes [1,4,4,4,4,4] ✗
        - [0,5,9,13,17] with T=21, chunk_size=4 → sizes [5,4,4,4,4] ✗

    Args:
        chunk_index: List of chunk start indices.
        T: Total number of frames.
        chunk_size: Expected uniform chunk size.

    Returns:
        True if chunking is uniform, False otherwise.
    """
    if chunk_size <= 0:
        return False

    # Compute actual chunk sizes
    sizes = compute_chunk_sizes(chunk_index, T)

    if not sizes:
        return True  # Empty is trivially uniform

    # Check that all chunks except possibly the last are equal to chunk_size
    for i, size in enumerate(sizes):
        is_last = i == len(sizes) - 1
        if is_last:
            # Last chunk can be <= chunk_size (remainder)
            if size > chunk_size:
                return False
        else:
            # All other chunks must be exactly chunk_size
            if size != chunk_size:
                return False

    return True


def normalize_chunk_index(
    chunk_index: Optional[List[int]],
    T: int,
    chunk_size: Optional[int] = None,
    chunk_split_strategy: str = "uniform",
) -> Tuple[List[int], bool]:
    """Normalize chunk_index and detect if uniform.

    This function handles all the complex logic for:
    1. Converting chunk_size + strategy → chunk_index (if needed)
    2. Cleaning and validating chunk_index
    3. Detecting if the result is uniform (safe for vectorized padding)

    Args:
        chunk_index: Optional pre-computed chunk indices.
        T: Total number of frames.
        chunk_size: Chunk size (required if chunk_index is None or for uniformity check).
        chunk_split_strategy: Strategy to use if generating chunk_index from chunk_size.

    Returns:
        (normalized_chunk_index, is_uniform):
            - normalized_chunk_index: Clean list of chunk start indices
            - is_uniform: True if safe to use vectorized path with padding

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    # Case 1: chunk_index provided explicitly
    if chunk_index is not None:
        normalized_chunk_index = list(chunk_index)

        # Clean up: ensure starts with 0 and ends with T
        if not normalized_chunk_index or normalized_chunk_index[0] != 0:
            normalized_chunk_index = [0] + [idx for idx in normalized_chunk_index if idx > 0]
        normalized_chunk_index = [idx for idx in normalized_chunk_index if idx < T]
        if not normalized_chunk_index:
            normalized_chunk_index = [0]
        if normalized_chunk_index[-1] != T:
            normalized_chunk_index = normalized_chunk_index + [T]

        # Check if uniform (requires chunk_size for comparison)
        if chunk_size is None:
            # Can't verify uniformity without chunk_size, assume non-uniform (safe)
            is_uniform = False
        else:
            is_uniform = is_uniform_chunking(normalized_chunk_index, T, chunk_size)

        return normalized_chunk_index, is_uniform

    # Case 2: Generate chunk_index from chunk_size + strategy
    if chunk_size is None:
        raise ValueError("Either chunk_index or chunk_size must be provided.")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

    # Normalize strategy
    strategy = "uniform" if chunk_split_strategy is None else str(chunk_split_strategy).lower()

    # Generate chunk_index
    chunk_index_gen = chunk_index_from_chunk_size(T, chunk_size, strategy=strategy)

    # Add T as final boundary
    if not chunk_index_gen:
        chunk_index_gen = [0]
    if chunk_index_gen[-1] != T:
        chunk_index_gen = chunk_index_gen + [T]

    # Check if uniform
    is_uniform = is_uniform_chunking(chunk_index_gen, T, chunk_size)

    return chunk_index_gen, is_uniform


# ============================================================================
# Attention blocks (sana / sana-camctrl / GDN / GDN-camctrl / softmax variants)
# ============================================================================

# String-keyed registry for the GDN/softmax attention block variants used by the SANA-WM DiT.
# `SanaWMTransformer3DModel` looks classes up here by its `attn_type` / `camctrl_type` config strings.
# Populated after the class definitions below.
ATTENTION_BLOCKS: dict[str, type] = {}


def _resolve_attention_block(name: str, *, role: str) -> type:
    """Look up a registered attention class by its config string."""
    cls = ATTENTION_BLOCKS.get(name)
    if cls is None:
        raise ValueError(f"Unknown {role}: {name!r}. Available: {sorted(ATTENTION_BLOCKS)}")
    return cls


# Safe element-count threshold for a single conv call: PyTorch's 2D conv kernels (both cuDNN and the ATEN fallback)
# use 32-bit indexing internally, so very large ``(batch * frames, channels, height, width)`` inputs (e.g. minute-scale
# video at default CFG) can overflow. Empirically a single call up to ~1B elements is safe; above that we split along
# the leading dim. Set so short videos stay on the original fused path (no chunking, no overhead).
_INT32_SAFE_CONV_ELEMENTS = 1 << 30  # 1,073,741,824


class SanaWMConvLayer(nn.Module):
    """2D convolution with an optional activation.

    Wraps the convolution in a ``conv`` submodule to keep the checkpoint's parameter names
    (``mlp.inverted_conv.conv.weight``, ...) unchanged.

    Args:
        in_dim (`int`): Input channels.
        out_dim (`int`): Output channels.
        kernel_size (`int`, defaults to 3): Spatial kernel size (odd, so ``same`` padding is exact).
        groups (`int`, defaults to 1): Convolution groups.
        use_bias (`bool`, defaults to `False`): Whether the convolution has a bias.
        act (`str`, *optional*): Activation name resolved through
            [`~models.activations.get_activation`], or `None` for no activation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        groups: int = 1,
        use_bias: bool = False,
        act: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            padding=kernel_size // 2,
            groups=groups,
            bias=use_bias,
        )
        self.act = get_activation(act) if act is not None else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        if self.act is not None:
            hidden_states = self.act(hidden_states)
        return hidden_states


class GLUMBConvTemp(nn.Module):
    """SANA-WM feed-forward block: a gated inverted-bottleneck conv over space plus a residual temporal conv.

    Args:
        in_features (`int`): Input channels.
        hidden_features (`int`): Width of the inverted bottleneck (doubled internally for the GLU gate).
        out_feature (`int`, *optional*): Output channels, defaults to `in_features`.
        kernel_size (`int`, defaults to 3): Spatial kernel size of the depthwise convolution.
        use_bias (`tuple[bool, bool, bool]`, defaults to `(False, False, False)`): Bias flag per convolution.
        act (`tuple`, defaults to `("silu", "silu", None)`): Activation for the inverted conv, the GLU gate and the
            point conv respectively; `None` means no activation.
        t_kernel_size (`int`, defaults to 3): Temporal kernel size of the residual temporal convolution.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature: Optional[int] = None,
        kernel_size: int = 3,
        use_bias: Tuple[bool, bool, bool] = (False, False, False),
        act: Tuple[Optional[str], Optional[str], Optional[str]] = ("silu", "silu", None),
        t_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        out_feature = out_feature or in_features

        self.glu_act = get_activation(act[1])
        self.inverted_conv = SanaWMConvLayer(
            in_features, hidden_features * 2, kernel_size=1, use_bias=use_bias[0], act=act[0]
        )
        self.depth_conv = SanaWMConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=kernel_size,
            groups=hidden_features * 2,
            use_bias=use_bias[1],
            act=None,
        )
        self.point_conv = SanaWMConvLayer(
            hidden_features, out_feature, kernel_size=1, use_bias=use_bias[2], act=act[2]
        )
        self.t_conv = nn.Conv2d(
            out_feature,
            out_feature,
            kernel_size=(t_kernel_size, 1),
            padding=(t_kernel_size // 2, 0),
            bias=False,
        )
        nn.init.zeros_(self.t_conv.weight)

    def forward(self, hidden_states: torch.Tensor, HW: Tuple[int, int, int], **kwargs) -> torch.Tensor:
        batch_size, seq_len, channels = hidden_states.shape
        num_frames, height, width = HW
        hidden_states = hidden_states.reshape(batch_size * num_frames, height, width, channels).permute(0, 3, 1, 2)

        # Split the leading dim so each conv launch stays under PyTorch's 32-bit indexing limit (no-op for short clips).
        rows_per_call = max(1, _INT32_SAFE_CONV_ELEMENTS // (self.inverted_conv.conv.out_channels * height * width))
        spatial_chunks = []
        for start in range(0, hidden_states.shape[0], rows_per_call):
            chunk = self.inverted_conv(hidden_states[start : start + rows_per_call])
            chunk = self.depth_conv(chunk)
            value, gate = torch.chunk(chunk, 2, dim=1)
            spatial_chunks.append(self.point_conv(value * self.glu_act(gate)))
        hidden_states = spatial_chunks[0] if len(spatial_chunks) == 1 else torch.cat(spatial_chunks, dim=0)

        # Residual temporal aggregation over the frame axis.
        hidden_states = hidden_states.view(batch_size, num_frames, channels, height * width).permute(0, 2, 1, 3)
        hidden_states = hidden_states + self.t_conv(hidden_states)
        return hidden_states.permute(0, 2, 3, 1).reshape(batch_size, seq_len, channels)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape
        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(B, -1, 2, C)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(B, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(q.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


#################################################################################
#   AMP attention with fp32 softmax to fix loss NaN problem during training     #
#################################################################################


class T2IFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward_frame_aware(self, x, t):
        # t: B,1,F,D
        B, N, C = x.shape
        num_frames = t.shape[2]
        # shift, scale: 2, hidden_size -> 1,1,2,hidden_size -> B,F,2,hidden_size
        shift, scale = (self.scale_shift_table[None, None, :, :] + t.transpose(1, 2)).chunk(
            2, dim=-2
        )  # each chunk: B,F,1,D
        x = (self.norm_final(x).reshape(B, num_frames, -1, C) * (1 + scale) + shift).reshape(B, N, C)
        x = self.linear(x)
        return x

    def forward(self, x, t):
        if len(t.shape) > 2:
            return self.forward_frame_aware(x, t)
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings. :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output. :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))

    def forward(self, caption):
        return self.y_proj(caption)


class PatchEmbedMS3D(nn.Module):
    """3D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_chans=3,
        embed_dim=768,
        kernel_size=None,
        padding=0,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        kernel_size = tuple(kernel_size or patch_size)
        patch_size = tuple(patch_size)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.flatten = flatten
        if patch_size[0] != 1:
            raise ValueError(f"Patch size for 3D embedding must be (1, *, *), got {patch_size}.")
        if not padding and kernel_size[-1] % 2 > 0:
            padding = tuple(k // 2 for k in kernel_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        x = self.norm(x)
        return x


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        fhw_dim: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        if fhw_dim is not None:
            assert attention_head_dim == sum(fhw_dim), (
                f"attention_head_dim {attention_head_dim} must match sum(fhw_dim) {sum(fhw_dim)}"
            )
            t_dim, h_dim, w_dim = fhw_dim
        else:
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float32
            )
            freqs.append(freq)
        self.register_buffer("freqs", torch.cat(freqs, dim=1), persistent=False)

    def forward(self, fhw: Tuple[int, int, int]) -> torch.Tensor:
        ppf, pph, ppw = fhw

        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


# ---------------------------------------------------------------------------
# UCM (Unified Camera Model) projection / unprojection and per-pixel ray
# transformation (world <-> ray) used by UCPE camera conditioning.
# ---------------------------------------------------------------------------


def compute_fov_from_fx_xi(
    fx: Union[torch.Tensor, float],
    xi: Union[torch.Tensor, float],
    width: int,
    device="cpu",
    dtype=torch.float32,
):
    """Inverse of :func:`compute_fx_from_fov_xi`."""

    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    fx = to_tensor_1d(fx).reshape(-1)
    xi = to_tensor_1d(xi).reshape(-1)
    B = max(fx.shape[0], xi.shape[0])
    fx = fx.expand(B)
    xi = xi.expand(B)
    A = 2.0 * fx / width
    phi = torch.atan(1.0 / A)
    denom = torch.sqrt(A * A + 1.0)
    ratio = (xi / denom).clamp(-1.0, 1.0)
    theta = torch.asin(ratio) + phi
    x_fov = torch.rad2deg(2.0 * theta)
    return x_fov


def ucm_unproject_grid_fov(
    x_fov: Union[float, torch.Tensor],
    y_fov: Union[float, torch.Tensor],
    xi: Union[float, torch.Tensor],
    height: int,
    width: int,
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unproject grid with intrinsics expressed as FoV (degrees) + xi."""
    is_batched = any(torch.is_tensor(p) and p.numel() > 1 for p in [x_fov, y_fov, xi, cx, cy])
    fx = compute_fx_from_fov_xi(x_fov, xi, width, device, dtype)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, device, dtype)
    d_cam = ucm_unproject_grid(
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        xi=xi if torch.is_tensor(xi) else torch.tensor([xi], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        y_down=True,
    )
    if not is_batched:
        d_cam = d_cam[0]
    return d_cam


def world_to_ray_mats(
    d_cam: torch.Tensor,  # [H, W, 3], [B, H, W, 3], or [B, T, H, W, 3]
    c2w: torch.Tensor,  # [B, T, 4, 4]
) -> torch.Tensor:
    """Build per-pixel ``ray<-world`` transforms from camera unit rays + C2W poses."""
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    if d_cam.ndim == 4:
        B, H, W, _ = d_cam.shape
        T = c2w.shape[1]
        d_cam = d_cam.unsqueeze(1).expand(-1, T, -1, -1, -1)
    elif d_cam.ndim == 5:
        B, T, H, W, _ = d_cam.shape
    else:
        raise ValueError(f"Unsupported d_cam shape: {d_cam.shape}")

    device = d_cam.device
    dtype = d_cam.dtype
    R_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", R_cam, d_cam)
    cam_y = R_cam[..., :, 1]
    # (B, T, 3) -> (B, T, H, W, 3)
    cam_y = cam_y[:, :, None, None, :].expand(-1, -1, H, W, -1)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-6)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-6)
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    # (B, T, H, W, 3, 3) — transpose last two dims for the world->local rotation.
    R_w2l = R_l2w.transpose(-1, -2)
    # (B, T, 3) -> (B, T, H, W, 3)
    t_world = t_cam[:, :, None, None, :].expand(-1, -1, H, W, -1)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", R_w2l, t_world)
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)
    return raymats


def create_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a pixel coordinate grid of shape ``(H, W, 3)`` or ``(B, H, W, 3)``."""
    if device.type == "cpu":
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: {dtype} is not supported by {device.type}\nIf device is `cpu`, use float32 or float64"
        )
    _xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    _ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    ys, xs = torch.meshgrid([_ys, _xs], indexing="ij")
    zs = torch.ones_like(xs, dtype=dtype, device=device)
    grid = torch.stack((xs, ys, zs), dim=2)
    if batch is not None:
        # Prepend a batch dim and broadcast.
        grid = grid.unsqueeze(0).expand(batch, *grid.shape)
    return grid


def ucm_unproject_grid(
    height: int,
    width: int,
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
    xi: Union[float, torch.Tensor],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    y_down: bool = True,
) -> torch.Tensor:
    """Unproject pixel grid into a camera-frame direction vector using the UCM."""
    fx_, fy_, cx_, cy_, xi_ = fx, fy, cx, cy, xi

    def to_tensor_flatten(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).reshape(-1)
        return torch.tensor([x], dtype=dtype, device=device)

    fx, fy, cx, cy, xi = map(to_tensor_flatten, (fx, fy, cx, cy, xi))
    B = max(fx.shape[0], fy.shape[0], cx.shape[0], cy.shape[0], xi.shape[0])
    fx = fx.expand(B)
    fy = fy.expand(B)
    cx = cx.expand(B)
    cy = cy.expand(B)
    xi = xi.expand(B)

    grid = create_grid(height=height, width=width, batch=B, dtype=dtype, device=device)
    u = grid[..., 0]
    v = grid[..., 1]
    fx = fx[:, None, None]
    fy = fy[:, None, None]
    cx = cx[:, None, None]
    cy = cy[:, None, None]
    xi = xi[:, None, None]
    x = (u - cx) / fx
    y = (v - cy) / fy
    if not y_down:
        y = -y
    r2 = x * x + y * y
    alpha = xi + torch.sqrt(1 + (1 - xi * xi) * r2)
    gamma = alpha / (1 + r2)
    X = gamma * x
    Y = gamma * y
    Z = gamma - xi
    d_cam = torch.stack([X, Y, Z], dim=-1)
    is_scalar_input = all(not torch.is_tensor(p) for p in (fx_, fy_, cx_, cy_, xi_))
    if is_scalar_input:
        return d_cam[0]
    else:
        return d_cam


def compute_fx_from_fov_xi(
    x_fov: Union[torch.Tensor, float],
    xi: Union[torch.Tensor, float],
    width: int,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Recover focal length ``fx`` from horizontal FoV (degrees) + UCM xi."""

    def to_tensor_flatten(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).view(-1)
        return torch.tensor([x], dtype=dtype, device=device)

    x_fov = to_tensor_flatten(x_fov)
    xi = to_tensor_flatten(xi)
    B = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.expand(B)
    xi = xi.expand(B)
    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / denom
    return fx


def project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi):
    """Project 3D points in camera frame to UCM image plane."""
    r = torch.sqrt(X * X + Y * Y + Z * Z)

    def reshape_param(p, target):
        if torch.is_tensor(p):
            if p.numel() == 1:
                return p
            if p.ndim == 1 and target.ndim == 4:
                return p.view(target.shape[0], target.shape[1], 1, 1)
            while p.ndim < target.ndim:
                p = p.unsqueeze(-1)
        return p

    xi = reshape_param(xi, X)
    fx = reshape_param(fx, X)
    fy = reshape_param(fy, X)
    cx = reshape_param(cx, X)
    cy = reshape_param(cy, X)

    alpha = Z + xi * r
    du = fx * (X / alpha) + cx
    dv = fy * (Y / alpha) + cy
    return du, dv


def project_ucm_points_fov(X, Y, Z, x_fov, y_fov, xi, height, width, cx, cy):
    """Project 3D points in camera frame to UCM image plane using FoV-based intrinsics."""
    fx = compute_fx_from_fov_xi(x_fov, xi, width, X.device, X.dtype)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, X.device, X.dtype)
    return project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi)


def compute_up_lat_map(
    R: torch.Tensor,
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    delta: float = 0.1,
):
    """Compute UCPE absolute embedding maps ``(up_map, lat_map)``.

    ``up_map`` is a 2-channel projected up-direction; ``lat_map`` is a 1-channel latitude. Concatenated they form the
    3-channel absmap consumed by the camera branch.
    """
    B, T, _, _ = R.shape
    dtype = R.dtype
    R = R.float()
    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        y_fov=y_fov,
        xi=xi,
        height=height,
        width=width,
        cx=cx,
        cy=cy,
        device=device,
        dtype=torch.float32,
    )

    if d_cam.ndim == 3:
        # (H, W, C) -> (B, T, H, W, C)
        d_cam_exp = d_cam[None, None].expand(B, T, -1, -1, -1)
    elif d_cam.ndim == 4:
        if d_cam.shape[0] == B * T:
            d_cam_exp = d_cam.view(B, T, height, width, 3)
        else:
            # (B, H, W, C) -> (B, T, H, W, C)
            d_cam_exp = d_cam.unsqueeze(1).expand(-1, T, -1, -1, -1)
    else:
        d_cam_exp = d_cam

    mask_exp = d_cam_exp.isnan().any(dim=-1, keepdim=True)
    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)
    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(-1)
    v = d_world
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    k = torch.cross(v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1)
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)
    delta_t = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps = torch.cos(delta_t)
    sin_eps = torch.sin(delta_t)
    v_rot = (
        v * cos_eps + torch.cross(k, v, dim=-1) * sin_eps + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)
    )
    dirs_cam = torch.einsum("btij,bthwj->bthwi", R.transpose(-1, -2), v_rot)
    Xs, Ys, Zs = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]
    du, dv = project_ucm_points_fov(
        Xs,
        Ys,
        Zs,
        x_fov=x_fov.float(),
        y_fov=y_fov.float(),
        xi=xi.float(),
        height=height,
        width=width,
        cx=cx.float(),
        cy=cy.float(),
    )
    grid = create_grid(
        height=height,
        width=width,
        batch=B,
        dtype=torch.float32,
        device=device,
    )
    grid_x = grid[..., 0].unsqueeze(1)
    grid_y = grid[..., 1].unsqueeze(1)
    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)
    up_map = up_map.to(dtype=dtype)
    lat_map = lat_map.to(dtype=dtype)
    up_map = up_map.masked_fill(mask_exp, 0.0)
    lat_map = lat_map.masked_fill(mask_exp, 0.0)
    return up_map, lat_map


def _process_camera_conditions_ucpe(camera_conditions, B, HW, patch_size):
    """Convert ``(B, F, 20)`` camera conditions (C2W flat + fx,fy,cx,cy) into
    ``(raymats, absmap)``.

    ``raymats`` is ``(B, F, H, W, 4, 4)`` ``ray<-world`` transforms; ``absmap`` is ``(B, F, H, W, 3)`` (up_map 2-ch +
    lat_map 1-ch).
    """
    F_dim = camera_conditions.shape[1]
    c2w_flat = camera_conditions[..., :16]
    C_to_W = c2w_flat.view(B, F_dim, 4, 4)

    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]
    H_dim, W_dim = HW[1], HW[2]
    image_width = W_dim * patch_size[2]
    image_height = H_dim * patch_size[1]

    # xi is fixed at 0 (pinhole) in this stack.
    xi = torch.zeros((B, F_dim), device=camera_conditions.device, dtype=camera_conditions.dtype)
    x_fov = compute_fov_from_fx_xi(
        fx, xi, image_width, device=camera_conditions.device, dtype=camera_conditions.dtype
    ).view(B, F_dim)
    y_fov = compute_fov_from_fx_xi(
        fy, xi, image_height, device=camera_conditions.device, dtype=camera_conditions.dtype
    ).view(B, F_dim)

    d_cam = ucm_unproject_grid_fov(
        x_fov,
        y_fov,
        xi,
        H_dim,
        W_dim,
        cx / patch_size[2],
        cy / patch_size[1],
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    )
    if d_cam.ndim == 4 and d_cam.shape[0] == B * F_dim:
        d_cam = d_cam.view(B, F_dim, H_dim, W_dim, 3)

    raymats = world_to_ray_mats(d_cam, C_to_W)  # [B, F, H, W, 4, 4]

    up_map, lat_map = compute_up_lat_map(
        R=C_to_W[..., :3, :3],
        x_fov=x_fov,
        y_fov=y_fov,
        xi=xi,
        height=image_height,
        width=image_width,
        cx=cx,
        cy=cy,
        device=camera_conditions.device,
    )
    absmap = torch.cat([up_map, lat_map], dim=-1)  # (B, F, H, W, 3)

    return raymats, absmap


# ---------------------------------------------------------------------------
# Block-diagonal apply primitives shared by camera and main branches
# ---------------------------------------------------------------------------


def _apply_ucpe_transform(
    feats: torch.Tensor,
    matrix: torch.Tensor,
    rotary_emb: Optional[torch.Tensor] = None,
    inverse_rope: bool = False,
) -> torch.Tensor:
    """Apply the block-diagonal UCPE transform to per-token features.

    The channel axis is split in half: the first half is rotated by the per-token 4x4 ray matrix (applied to channels
    grouped by 4), the second half gets complex RoPE.

    Args:
        feats (`torch.Tensor`): Features of shape `(batch, heads, seq_len, head_dim)`.
        matrix (`torch.Tensor`): Per-token 4x4 transform of shape `(batch, seq_len, 4, 4)`.
        rotary_emb (`torch.Tensor`, *optional*): Complex RoPE frequencies; `None` leaves the second half unchanged.
        inverse_rope (`bool`, defaults to `False`): Conjugate the frequencies (inverse rotation), used on the output.

    Returns:
        `torch.Tensor`: Transformed features with the same shape as `feats`.
    """
    batch, num_heads, seq_len, head_dim = feats.shape
    half_dim = head_dim // 2
    projected, rotated = feats.split(half_dim, dim=-1)

    matrix_dim = matrix.shape[-1]
    projected = torch.einsum(
        "bnij,bhnkj->bhnki",
        matrix,
        projected.reshape(batch, num_heads, seq_len, -1, matrix_dim),
    ).reshape(batch, num_heads, seq_len, half_dim)

    if rotary_emb is not None:
        rotated_fp32 = rotated.to(torch.float32)
        if rotated_fp32.stride(-1) != 1:
            rotated_fp32 = rotated_fp32.contiguous()
        freqs = rotary_emb.conj() if inverse_rope else rotary_emb
        rotated_complex = torch.view_as_complex(rotated_fp32.unflatten(-1, (-1, 2)))
        rotated = torch.view_as_real(rotated_complex * freqs).flatten(-2, -1).type_as(rotated)

    return torch.cat([projected, rotated], dim=-1)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a 4x4 SE(3) batch."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# UCPE ray-transform preparation
# ---------------------------------------------------------------------------


def _slice_rope_for_cam(
    rotary_emb: Optional[torch.Tensor],
    head_dim: int,
    rope_dim: int,
) -> Optional[torch.Tensor]:
    """Re-slice WAN RoPE frequencies for a smaller rope_dim using the same (T, H, W) split."""
    if rotary_emb is None:
        return None
    orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
    orig_h_size = head_dim // 6
    new_t_size = rope_dim // 2 - 2 * (rope_dim // 6)
    new_h_size = rope_dim // 6
    new_w_size = rope_dim // 6
    t_part = rotary_emb[..., :new_t_size]
    h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
    w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
    return torch.cat([t_part, h_part, w_part], dim=-1)


def _prepare_ucpe_ray_transforms(
    head_dim: int,
    camera_conditions: torch.Tensor,
    HW: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    rotary_emb: Optional[torch.Tensor] = None,
    raymats: Optional[torch.Tensor] = None,
    cam_pos_embeds: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Precompute the UCPE ray matrices once for a batch, shared across all blocks.

    Accepts either precomputed matrices (`cam_pos_embeds` with `P`, `P_inv`, `pos_embeds_cam`) or raw camera conditions
    plus optional `raymats`.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: `(P, P_T, P_inv, rotary_emb_cam)`,
        where `P` is the `ray<-world` transform used on the output and `P_T` / `P_inv` are used on Q and K/V.
    """
    batch_size = camera_conditions.shape[0]

    # Priority 1: use precomputed matrices.
    if cam_pos_embeds is not None:
        P = cam_pos_embeds.get("P")
        P_inv = cam_pos_embeds.get("P_inv")
        rotary_emb_cam = cam_pos_embeds.get("pos_embeds_cam")

        if P is not None and P_inv is not None:
            if P.ndim == 3:
                P = P.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            if P_inv.ndim == 3:
                P_inv = P_inv.unsqueeze(0).repeat(batch_size, 1, 1, 1)

            if rotary_emb_cam is not None and rotary_emb_cam.ndim == 3:
                rotary_emb_cam = rotary_emb_cam.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif rotary_emb_cam is None and rotary_emb is not None:
                rotary_emb_cam = _slice_rope_for_cam(rotary_emb, head_dim, head_dim // 2)
            elif rotary_emb_cam is None:
                rotary_emb_cam = rotary_emb

            return P, P.transpose(-1, -2), P_inv, rotary_emb_cam

    # Priority 2: online path.
    if raymats is None:
        raymats, _ = _process_camera_conditions_ucpe(camera_conditions, batch_size, HW, patch_size)
    P = raymats.reshape(batch_size, -1, 4, 4)
    rotary_emb_cam = _slice_rope_for_cam(rotary_emb, head_dim, head_dim // 2)

    return P, P.transpose(-1, -2), _invert_SE3(P), rotary_emb_cam


OUTPUT_GATE_INIT_BIAS = 1.278464542761074  # silu(x)=1.0


def flip_and_shift(x, dim=2, shift_val=0.0):
    """Flip a sequence and shift it right by one step.

    The operation reverses the sequence, drops the last element, and pads the front with ``shift_val``.

    Example:
        [x0, x1, x2, x3] -> flip [x3, x2, x1, x0] -> shift [v, x3, x2, x1]

    Args:
        x: Input tensor with a time dimension at ``dim``.
        dim: Dimension to flip and shift.
        shift_val: Value used for the padded step.

    Returns:
        Tensor with the same shape as ``x``.
    """
    x_flip = torch.flip(x, dims=[dim])
    x_shifted = x_flip.narrow(dim, 0, x.shape[dim] - 1)
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_val, device=x.device, dtype=x.dtype)
    return torch.cat([padding, x_shifted], dim=dim)


def torch_chunk_sana_gdn(
    q,
    k,
    v,
    q_rot,
    k_rot,
    beta,
    decay,
    recall_gate=None,
    chunk_size: int | None = 21,
    eps: float = 1e-6,
    return_components: bool = False,
):
    del recall_gate  # Accepted so the chunk and fused scan share one signature; unused by this rule.

    B, H, D, N = q.shape
    if beta.ndim not in (3, 4):
        raise ValueError(f"Expected beta.ndim in (3, 4), got {beta.ndim}.")
    T = beta.shape[2]
    if T <= 0:
        raise ValueError(f"Expected T > 0, got T={T}.")
    if N % T != 0:
        raise ValueError(f"Expected N divisible by T, got N={N}, T={T}.")
    S = N // T

    target_z = 1.0
    scale = 1.0

    def to_frame_seq(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q, k, v = to_frame_seq(q), to_frame_seq(k), to_frame_seq(v)
    q_rot, k_rot = to_frame_seq(q_rot), to_frame_seq(k_rot)

    if beta.ndim == 4:
        beta = beta.unsqueeze(3)
    else:
        beta = beta.view(B, H, T, 1, 1)

    decay = decay.view(B, H, T, 1, 1)

    # =========================================================================
    # 1. PARALLEL PRE-PROCESSING
    # =========================================================================

    I = torch.eye(D, device=q.device, dtype=q.dtype).view(1, 1, 1, D, D)

    # KV State Matrices: W = g * (I - c * K @ K^T)
    k_rot_beta = k_rot * beta
    W_kv = decay * (I - scale * torch.matmul(k_rot_beta, k_rot.transpose(-1, -2)))
    U_kv = torch.matmul(v * beta, k_rot.transpose(-1, -2))

    # Z State Matrices: W = g * (I - c * K @ K^T)
    k_beta = k * beta
    W_z = decay * (I - scale * torch.matmul(k_beta, k.transpose(-1, -2)))
    U_z = target_z * k_beta.sum(dim=-1, keepdim=True)  # Equivalent to Kt @ bt^T over spatial dim

    # =========================================================================
    # 2. CHUNKING LOGIC
    # =========================================================================

    valid_chunk_index, _ = normalize_chunk_index(None, T, chunk_size)
    split_sizes = [valid_chunk_index[i + 1] - valid_chunk_index[i] for i in range(len(valid_chunk_index) - 1)]

    W_kv_c = W_kv.split(split_sizes, dim=2)
    U_kv_c = U_kv.split(split_sizes, dim=2)
    W_z_c = W_z.split(split_sizes, dim=2)
    U_z_c = U_z.split(split_sizes, dim=2)

    # =========================================================================
    # 3. FAST INTRA-CHUNK SCAN OVER DxD SPACE
    # =========================================================================

    S_kv = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    S_z = torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)

    out_S_kv = []
    out_S_z = []

    def _chunk_scan(w_kv, u_kv, w_z, u_z, s_kv, s_z):
        c_len = w_kv.shape[2]
        s_kv_list, s_z_list = [], []
        for t in range(c_len):
            s_kv = torch.matmul(s_kv, w_kv[:, :, t]) + u_kv[:, :, t]
            s_z = torch.matmul(w_z[:, :, t], s_z) + u_z[:, :, t]
            s_kv_list.append(s_kv)
            s_z_list.append(s_z)
        return torch.stack(s_kv_list, dim=2), s_kv, torch.stack(s_z_list, dim=2), s_z

    for i in range(len(split_sizes)):
        s_kv_all, S_kv, s_z_all, S_z = _chunk_scan(W_kv_c[i], U_kv_c[i], W_z_c[i], U_z_c[i], S_kv, S_z)
        out_S_kv.append(s_kv_all)
        out_S_z.append(s_z_all)

    S_kv_all = torch.cat(out_S_kv, dim=2)
    S_z_all = torch.cat(out_S_z, dim=2)

    # =========================================================================
    # 4. PARALLEL OUTPUT PROJECTION
    # =========================================================================

    out_num = torch.matmul(S_kv_all, q_rot)
    out_den = torch.matmul(S_z_all.transpose(-1, -2), q)

    def restore_shape(tensor, target_d):
        return tensor.permute(0, 1, 3, 2, 4).reshape(B, H, target_d, N)

    final_num = restore_shape(out_num, D)
    final_den = restore_shape(out_den, 1)

    if return_components:
        return final_num, final_den

    return final_num / (final_den + eps)


# ---------------------------------------------------------------------------
# Helpers for hot-path operations
# ---------------------------------------------------------------------------


def _compute_frame_gates(
    x: torch.Tensor,
    T: int,
    S: int,
    heads: int,
    beta_weight: torch.Tensor,
    beta_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame beta / decay gates."""
    B, N, C = x.shape
    beta = F.linear(x, beta_weight, beta_bias).sigmoid().reshape(B, T, S, heads).permute(0, 3, 1, 2)
    x_frame = x.reshape(B, T, S, C).mean(dim=2)
    a_out = F.linear(x_frame, gate_weight, gate_bias).float()
    dt = dt_bias.float().view(1, 1, -1)
    A_val = A_log.float().exp().view(1, 1, -1)
    decay = (-A_val * F.softplus(a_out + dt)).exp().transpose(1, 2)
    return beta, decay


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to `(batch, heads, dim, seq_len)` features."""
    x_rotated = torch.view_as_complex(
        hidden_states.permute(0, 1, 3, 2).to(torch.float32).unflatten(3, (-1, 2)),
    )
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
    return x_out.type_as(hidden_states)


def _apply_output_gate(
    out: torch.Tensor,
    gate_x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
) -> torch.Tensor:
    """Apply the SiLU output gate."""
    gate = F.silu(F.linear(gate_x, gate_weight, gate_bias).to(torch.float32))
    return out * gate


class GDN(nn.Module):
    """Frame-wise Gated Delta Net attention for Sana video.

    This block follows Sana's vanilla linear attention strategy but upgrades it with a Gated Delta Network mechanism:
    - Apply ReLU kernel to q/k.
    - Apply RoPE only on the numerator (q_rot, k_rot).
    - Denominator (Z stream) uses unrotated q/k to maintain mass conservation.
    - Gated delta rule is applied across time (T). Gates are computed per-frame (shared spatially), but states are
      maintained per-pixel.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-15,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        use_output_gate: bool = True,
        chunk_gdn_chunk_size: int = 21,
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        **kwargs: object,
    ) -> None:
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__()

        # Fused QKV projection and output projection (the `q_norm` / `k_norm`
        # attributes are set further down, depending on `qk_norm`).
        self.num_heads = heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=use_bias)
        self.proj = nn.Linear(in_dim, in_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.k_conv_only = k_conv_only
        self.key_scale_mode = str(kwargs.pop("key_scale_mode", "dim_spatial"))

        self.kernel_func = nn.ReLU(inplace=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(self.in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Gate projections operate on pooled frame features (B, T, D) -> (B, T, H).
        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)

        A = torch.zeros(self.heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # `recall_gate` is unused by the forward; kept as a buffer for checkpoint compatibility.
        self.register_buffer("recall_gate", torch.zeros(1))

        self.use_output_gate = use_output_gate
        if use_output_gate:
            self.output_gate = nn.Linear(in_dim, out_dim, bias=True)
        else:
            self.output_gate = None

        self.chunk_gdn_chunk_size = chunk_gdn_chunk_size

        # Short Convolutions (FLA causal depthwise Conv1d along T)
        self.conv_kernel_size = conv_kernel_size
        if conv_kernel_size > 0:
            self.conv_k = ShortConvolution(
                hidden_size=out_dim,
                kernel_size=conv_kernel_size,
                activation=None,
            )
            if k_conv_only:
                self.conv_q = None
                self.conv_v = None
            else:
                self.conv_q = ShortConvolution(
                    hidden_size=out_dim,
                    kernel_size=conv_kernel_size,
                    activation=None,
                )
                self.conv_v = ShortConvolution(
                    hidden_size=out_dim,
                    kernel_size=conv_kernel_size,
                    activation=None,
                )
        else:
            self.conv_q = None
            self.conv_k = None
            self.conv_v = None

    def _key_scale(self, spatial_tokens: int) -> float:
        """Return the post-ReLU key scale used by frame-wise GDN."""
        if self.key_scale_mode == "dim_spatial":
            return (self.dim**-0.5) * (spatial_tokens**-0.5)
        if self.key_scale_mode == "dim":
            return self.dim**-0.5
        if self.key_scale_mode == "none":
            return 1.0
        raise ValueError(f"Unsupported GDN key_scale_mode: {self.key_scale_mode}")

    def _apply_output_gate(self, out: torch.Tensor, gate_x: torch.Tensor) -> torch.Tensor:
        if not (self.use_output_gate and self.output_gate is not None):
            return out
        return _apply_output_gate(out, gate_x, self.output_gate.weight, self.output_gate.bias)

    @staticmethod
    def _reshape_to_temporal(x: torch.Tensor, HW: tuple[int, int, int]) -> tuple[torch.Tensor, int, int, int]:
        """Reshape (B, T*S, C) to (B*S, T, C) for temporal conv.

        Returns:
            Reshaped tensor and (B, S, T) for later restoration.
        """
        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        # FLA ShortConvolution backward is not reliable on non-contiguous
        # strided layouts produced by this permutation path.
        x = x.reshape(B, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B * S, T, C)
        return x, B, S, T

    @staticmethod
    def _reshape_from_temporal(x: torch.Tensor, B: int, S: int, T: int) -> torch.Tensor:
        """Reshape (B*S, T, C) back to (B, T*S, C)."""
        C = x.shape[-1]
        return x.reshape(B, S, T, C).permute(0, 2, 1, 3).reshape(B, T * S, C)

    @staticmethod
    def _causal_conv_1d(
        x: torch.Tensor,
        conv: ShortConvolution,
    ) -> torch.Tensor:
        """Run causal conv and preserve input dtype.

        Args:
            x: Tensor of shape (batch, seq_len, channels).
            conv: FLA ``ShortConvolution`` module.

        Returns:
            Tensor of same shape and dtype as ``x``.
        """
        dtype_in = x.dtype
        y, _ = conv(x)
        if y.dtype != dtype_in:
            y = y.to(dtype_in)
        return y

    @staticmethod
    def _bidirectional_causal_conv_1d(
        x: torch.Tensor,
        conv: ShortConvolution,
    ) -> torch.Tensor:
        """Simulate non-causal conv by combining forward + backward causal passes.

        A causal depthwise Conv1d with kernel ``[w_0, w_1, ..., w_{k-1}]`` computes at time *t*:

            ``y_fwd[t] = w_0 * x[t-k+1] + ... + w_{k-1} * x[t]``

        Running the same kernel on the time-flipped input and flipping back gives:

            ``y_bwd[t] = w_{k-1} * x[t] + ... + w_0 * x[t+k-1]``

        Both passes include the current timestep ``x[t]`` with the center weight ``w_{k-1}``. To avoid double-counting
        we subtract one copy of the center contribution:

            ``y = y_fwd + y_bwd - w_{k-1} * x``

        The result is a symmetric temporal filter where every position in the window ``[t-k+1, t+k-1]`` is counted
        exactly once.

        Args:
            x: Tensor of shape ``(batch, seq_len, channels)``.
            conv: FLA ``ShortConvolution`` module (depthwise causal Conv1d).

        Returns:
            Tensor of same shape and dtype as ``x``.
        """
        dtype_in = x.dtype

        y_fwd, _ = conv(x)
        y_bwd, _ = conv(x.flip(1))
        y_bwd = y_bwd.flip(1)

        # Subtract the shared center tap (last weight of the causal kernel).
        # ShortConvolution weight shape: (channels, 1, kernel_size).
        # The last element along dim=-1 is the weight applied to x[t].
        w_center = conv.weight[:, 0, -1]  # (channels,)
        center_term = x * w_center.unsqueeze(0).unsqueeze(0)  # broadcast over (B, T)

        y = y_fwd + y_bwd - center_term
        if y.dtype != dtype_in:
            y = y.to(dtype_in)
        return y

    def _apply_temporal_short_conv(
        self,
        x: torch.Tensor,
        conv: ShortConvolution,
        HW: tuple[int, int, int],
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply causal ShortConvolution along T, with S merged into batch.

        Under CP, a causal conv of kernel size K needs K-1 left-context frames from the previous rank at each boundary.
        We use a halo exchange (O(K) communication) instead of a full gather (O(T)).

        Args:
            x: Input tensor of shape (B, N, C) where N = T * S.
            conv: FLA ``ShortConvolution`` module.
            HW: Tuple of (T, H, W) describing the token layout.
            **kwargs: Extra keyword arguments (unused in base; subclasses
                may consume ``chunk_size``, ``chunk_index``, etc.).

        Returns:
            Tensor of shape (B, N, C) after temporal convolution.
        """
        del kwargs  # unused in base class

        x, B, S, T = self._reshape_to_temporal(x, HW)
        x = self._causal_conv_1d(x, conv)
        return self._reshape_from_temporal(x, B, S, T)

    def _compute_frame_gates(
        self,
        x: torch.Tensor,
        hw: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-frame gates shared across spatial positions.

        Delegates to the module-level compiled ``_compute_frame_gates``.
        """
        T, H, W = hw
        S = H * W
        return _compute_frame_gates(
            x,
            T,
            S,
            self.heads,
            self.beta_proj.weight,
            self.beta_proj.bias,
            self.gate_proj.weight,
            self.gate_proj.bias,
            self.dt_bias,
            self.A_log,
        )

    @staticmethod
    def _prepare_frame_valid_masks(
        frame_valid_mask: torch.Tensor | None,
        *,
        B: int,
        T: int,
        S: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Convert frame-valid mask to token/beta/decay masks used by GDN blocks."""
        if frame_valid_mask is None:
            return None, None, None

        m = frame_valid_mask
        if m.ndim == 5:
            # (B, 1, T, 1, 1)
            m = m[:, 0, :, 0, 0]
        elif m.ndim == 3 and m.shape[1] == 1:
            # (B, 1, T)
            m = m[:, 0, :]
        elif m.ndim != 2:
            raise ValueError(
                "frame_valid_mask must be shaped (B, 1, T, 1, 1), (B, 1, T), or (B, T); "
                f"got shape={list(frame_valid_mask.shape)}"
            )

        if m.shape[0] != B or m.shape[1] != T:
            raise ValueError(f"frame_valid_mask shape mismatch: expected (B={B}, T={T}), got {list(m.shape)}")

        m = m.to(device=device, dtype=dtype)
        token_valid_mask = m[:, :, None].expand(B, T, S).reshape(B, T * S)
        beta_valid_mask = m.view(B, 1, T, 1)
        decay_valid_mask = m.view(B, 1, T)
        return token_valid_mask, beta_valid_mask, decay_valid_mask

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply GDN attention to a token sequence.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Unused attention mask (kept for API compatibility).
            HW: Tuple of (T, H, W) describing the token layout.
            rotary_emb: Optional rotary embeddings for q/k.
            block_mask: Unused block mask (kept for API compatibility).
            apply_output_gate: When False, return raw attention output
                before output gate and projection.
            **kwargs: Unused extra arguments.

        Returns:
            Tensor of shape (B, N, C) after attention and projection.
        """
        del mask, block_mask
        frame_valid_mask = kwargs.get("frame_valid_mask", None)

        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for GDN attention.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            frame_valid_mask,
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )
        if token_valid_mask is not None:
            x = x * token_valid_mask.view(B, N, 1)

        # Projections.
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if token_valid_mask is not None:
            token_mask_bnhd = token_valid_mask.view(B, N, 1, 1)
            q = q * token_mask_bnhd
            k = k * token_mask_bnhd
            v = v * token_mask_bnhd

        # Short convolution along T (before norm / kernel activation).
        if self.conv_k is not None:
            if self.conv_q is not None:
                q = self._apply_temporal_short_conv(q.reshape(B, N, C), self.conv_q, HW).reshape(
                    B, N, self.heads, self.dim
                )
            k = self._apply_temporal_short_conv(k.reshape(B, N, C), self.conv_k, HW).reshape(
                B, N, self.heads, self.dim
            )
            if self.conv_v is not None:
                v = self._apply_temporal_short_conv(v.reshape(B, N, C), self.conv_v, HW).reshape(
                    B, N, self.heads, self.dim
                )

        # Apply Q/K norm on flattened channels (B, N, C) then reshape to heads.
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # ReLU kernel.
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        k_scale = self._key_scale(S)
        k = k * k_scale

        # Permute to (B, H, D, N) for processing.
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q = q * token_mask_qkv
            k = k * token_mask_qkv
            v = v * token_mask_qkv

        # RoPE preparation (numerator only).
        if rotary_emb is not None:
            q_rot = _apply_rotary_emb(q, rotary_emb)
            k_rot = _apply_rotary_emb(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_rot = q_rot * token_mask_qkv
            k_rot = k_rot * token_mask_qkv

        # Gate computation (use pre-computed gates when available to avoid
        # redundant work in dual-branch CamCtrl models).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        # Run the frame-wise GDN update.
        # Force FP32 to preserve recurrent stability.
        dtype_orig = x.dtype
        recall_gate = self.recall_gate
        q = q.float()
        k = k.float()
        v = v.float()
        q_rot = q_rot.float()
        k_rot = k_rot.float()
        beta = beta.float()
        decay = decay.float()
        recall_gate = recall_gate.float()

        out = torch_chunk_sana_gdn(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=recall_gate,
            chunk_size=self.chunk_gdn_chunk_size,
            eps=self.eps,
        )

        # Reshape and project output.
        if dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        out = out.permute(0, 3, 1, 2)
        N_out = out.shape[1]
        out = out.reshape(B, N_out, C)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(x.dtype))
            if token_valid_mask is not None:
                out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)
            return out
        return out


class BidirectionalGDN(GDN):
    """Bidirectional GDN attention with forward/backward fusion."""

    def _apply_temporal_short_conv(
        self,
        x: torch.Tensor,
        conv: ShortConvolution,
        HW: tuple[int, int, int],
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply bidirectional (non-causal) ShortConvolution along T.

        Uses the forward+backward causal trick: run the causal conv in both directions and average, yielding a
        symmetric temporal filter with a single set of weights.

        Args:
            x: Input tensor of shape (B, N, C) where N = T * S.
            conv: FLA ``ShortConvolution`` module.
            HW: Tuple of (T, H, W) describing the token layout.
            **kwargs: Unused.

        Returns:
            Tensor of shape (B, N, C) after bidirectional temporal conv.
        """
        del kwargs

        x, B, S, T = self._reshape_to_temporal(x, HW)
        x = self._bidirectional_causal_conv_1d(x, conv)
        return self._reshape_from_temporal(x, B, S, T)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply bidirectional GDN attention to a token sequence.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Unused attention mask (kept for API compatibility).
            HW: Tuple of (T, H, W) describing the token layout.
            rotary_emb: Optional rotary embeddings for q/k.
            block_mask: Unused block mask (kept for API compatibility).
            **kwargs: Unused extra arguments.

        Returns:
            Tensor of shape (B, N, C) after attention and projection.
        """
        del mask, block_mask
        frame_valid_mask = kwargs.get("frame_valid_mask", None)

        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for GDN attention.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            frame_valid_mask,
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )
        if token_valid_mask is not None:
            x = x * token_valid_mask.view(B, N, 1)

        # Projections.
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if token_valid_mask is not None:
            token_mask_bnhd = token_valid_mask.view(B, N, 1, 1)
            q = q * token_mask_bnhd
            k = k * token_mask_bnhd
            v = v * token_mask_bnhd

        # Short convolution along T (before norm / kernel activation).
        if self.conv_k is not None:
            if self.conv_q is not None:
                q = self._apply_temporal_short_conv(q.reshape(B, N, C), self.conv_q, HW).reshape(
                    B, N, self.heads, self.dim
                )
            k = self._apply_temporal_short_conv(k.reshape(B, N, C), self.conv_k, HW).reshape(
                B, N, self.heads, self.dim
            )
            if self.conv_v is not None:
                v = self._apply_temporal_short_conv(v.reshape(B, N, C), self.conv_v, HW).reshape(
                    B, N, self.heads, self.dim
                )

        # Apply Q/K norm on flattened channels (B, N, C) then reshape to heads.
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # ReLU kernel.
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        k_scale = self._key_scale(S)
        k = k * k_scale

        # Permute to (B, H, D, N) for processing.
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q = q * token_mask_qkv
            k = k * token_mask_qkv
            v = v * token_mask_qkv

        # RoPE preparation (numerator only).
        if rotary_emb is not None:
            q_rot = _apply_rotary_emb(q, rotary_emb)
            k_rot = _apply_rotary_emb(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_rot = q_rot * token_mask_qkv
            k_rot = k_rot * token_mask_qkv

        # Gate computation (use pre-computed gates when available).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        H_eff = q.shape[1]
        N_eff = q.shape[3]
        T_eff = N_eff // S

        # Run the frame-wise GDN update.
        # Force FP32 to preserve recurrent stability.
        dtype_orig = x.dtype
        recall_gate = self.recall_gate
        q = q.float()
        k = k.float()
        v = v.float()
        q_rot = q_rot.float()
        k_rot = k_rot.float()
        beta = beta.float()
        decay = decay.float()
        recall_gate = recall_gate.float()

        # Forward pass (inclusive: 1..t).
        num_fwd, den_fwd = torch_chunk_sana_gdn(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=recall_gate,
            chunk_size=self.chunk_gdn_chunk_size,
            eps=self.eps,
            return_components=True,
        )

        # Backward pass (exclusive: t+1..T).
        def to_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(B, H_eff, self.dim, T_eff, S).permute(0, 1, 3, 2, 4)

        def from_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(B, H_eff, self.dim, N_eff)

        q_T = to_time_structure(q)
        k_T = to_time_structure(k)
        v_T = to_time_structure(v)
        q_rot_T = to_time_structure(q_rot)
        k_rot_T = to_time_structure(k_rot)

        q_bwd = torch.flip(q_T, dims=[2])
        q_rot_bwd = torch.flip(q_rot_T, dims=[2])

        k_bwd = flip_and_shift(k_T, dim=2, shift_val=0.0)
        v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
        k_rot_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        k_bwd_flat = from_time_structure(k_bwd)
        v_bwd_flat = from_time_structure(v_bwd)
        q_bwd_flat = from_time_structure(q_bwd)
        q_rot_bwd_flat = from_time_structure(q_rot_bwd)
        k_rot_bwd_flat = from_time_structure(k_rot_bwd)

        num_bwd_flipped, den_bwd_flipped = torch_chunk_sana_gdn(
            q_bwd_flat,
            k_bwd_flat,
            v_bwd_flat,
            q_rot_bwd_flat,
            k_rot_bwd_flat,
            beta_bwd,
            decay_bwd,
            recall_gate=recall_gate,
            chunk_size=self.chunk_gdn_chunk_size,
            eps=self.eps,
            return_components=True,
        )

        def flip_back(tensor: torch.Tensor) -> torch.Tensor:
            d_actual = tensor.shape[2]
            t_struct = tensor.view(B, H_eff, d_actual, T_eff, S)
            return torch.flip(t_struct, dims=[3]).reshape(B, H_eff, d_actual, N_eff)

        num_bwd = flip_back(num_bwd_flipped)
        den_bwd = flip_back(den_bwd_flipped)

        total_num = num_fwd + num_bwd
        total_den = den_fwd + den_bwd

        out = total_num / (total_den + self.eps)

        # Reshape and project output.
        if dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        out = out.permute(0, 3, 1, 2)
        N_out = out.shape[1]
        out = out.reshape(B, N_out, C)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(x.dtype))
            if token_valid_mask is not None:
                out = out * token_valid_mask.view(B, N_out, 1).to(out.dtype)
            return out
        return out


_frame_causal_mask_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _get_frame_causal_mask(T: int, S: int, device: torch.device) -> torch.Tensor:
    """Frame-wise block-causal mask: full attention within each frame,
    causal across frames.

    Returns a boolean tensor of shape ``(1, 1, T*S, T*S)`` where ``True`` indicates positions that may attend.
    """
    key = (T, S, device)
    if key not in _frame_causal_mask_cache:
        frame_idx = torch.arange(T, device=device).repeat_interleave(S)
        mask = frame_idx.unsqueeze(1) >= frame_idx.unsqueeze(0)
        _frame_causal_mask_cache[key] = mask.unsqueeze(0).unsqueeze(0)
    return _frame_causal_mask_cache[key]


def _forward_softmax_attn(
    self,
    x: torch.Tensor,
    HW: tuple[int, int, int],
    rotary_emb: torch.Tensor | None,
    frame_causal: bool,
    apply_output_gate: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Softmax attention (SDPA) reusing GDN parameters.

    Used by the hybrid GDN+Softmax architecture: every Nth block runs softmax attention instead of the gated-delta
    recurrence. Reuses the parent block's QKV/q_norm/k_norm/proj for parameter compatibility.
    """
    import torch.nn.functional as F

    B, N, C = x.shape
    T, H, W = HW
    S = H * W

    frame_valid_mask = kwargs.get("frame_valid_mask", None)
    token_valid_mask, _, _ = GDN._prepare_frame_valid_masks(
        frame_valid_mask,
        B=B,
        T=T,
        S=S,
        device=x.device,
        dtype=x.dtype,
    )
    if token_valid_mask is not None:
        x = x * token_valid_mask.view(B, N, 1)

    qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
    q, k, v = qkv.unbind(2)
    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1, 1)
        q, k, v = q * m, k * m, v * m

    q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
    k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

    if rotary_emb is not None:
        q_perm = q.permute(0, 2, 3, 1)
        k_perm = k.permute(0, 2, 3, 1)
        q_perm = _apply_rotary_emb(q_perm, rotary_emb)
        k_perm = _apply_rotary_emb(k_perm, rotary_emb)
        q = q_perm.permute(0, 3, 1, 2)
        k = k_perm.permute(0, 3, 1, 2)

    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1, 1)
        q, k, v = q * m, k * m, v * m

    q = q.transpose(1, 2)  # (B, H, N, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    dtype_orig = x.dtype
    if q.dtype == torch.float32:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    attn_mask = _get_frame_causal_mask(T, S, x.device) if frame_causal else None

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).reshape(B, N, C).to(dtype_orig)

    if apply_output_gate:
        # Re-apply the parent's output projection w/ silu gate; some GDN
        # variants split projection into proj_o + proj_gate; match those.
        if hasattr(self, "proj_gate"):
            out = out * F.silu(self.proj_gate(x))
        out = self.proj(out)
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def torch_chunk_cam_single_path_delta_rule(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    chunk_size: int | None = 21,
) -> torch.Tensor:
    """Parallel chunk-scan version of the single-path delta-rule recurrence.

    Restructured as a linear recurrence in D x D state space so that Phases 1 (transition-matrix construction) and 3
    (output projection) are fully parallel over T, while Phase 2 (the D x D state scan) is chunked.

    The recurrence:
        state[t] = state[t-1] * g[t] + delta_v[t] @ k_rot[t]^T
    where delta_v[t] = (v[t] - state[t-1]*g[t] @ k_rot[t]) * beta[t]

    is equivalent to:
        state[t] = state[t-1] @ W[t] + U[t]
    with:
        W[t] = g[t] * (I - beta[t] * k_rot[t] @ k_rot[t]^T) U[t] = beta[t] * v[t] @ k_rot[t]^T
    """
    B, H, D, N = q_rot.shape
    if beta.ndim not in (3, 4):
        raise ValueError(f"Expected beta.ndim in (3, 4), got {beta.ndim}.")
    T = beta.shape[2]
    if T <= 0:
        raise ValueError(f"Expected T > 0, got T={T}.")
    if N % T != 0:
        raise ValueError(f"Expected N divisible by T, got N={N}, T={T}.")
    S = N // T

    def to_frame_seq(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_rot = to_frame_seq(q_rot)
    k_rot = to_frame_seq(k_rot)
    v = to_frame_seq(v)

    if beta.ndim == 4:
        beta = beta.unsqueeze(3)
    else:
        beta = beta.view(B, H, T, 1, 1)
    decay = decay.view(B, H, T, 1, 1)

    # =========================================================================
    # Phase 1: PARALLEL PRE-PROCESSING  (fully parallel over T)
    # =========================================================================
    I = torch.eye(D, device=q_rot.device, dtype=q_rot.dtype).view(1, 1, 1, D, D)

    k_rot_beta = k_rot * beta
    W_kv = decay * (I - torch.matmul(k_rot_beta, k_rot.transpose(-1, -2)))
    U_kv = torch.matmul(v * beta, k_rot.transpose(-1, -2))

    # =========================================================================
    # Phase 2: CHUNKED SCAN over D x D state space
    # =========================================================================
    valid_chunk_index, _ = normalize_chunk_index(None, T, chunk_size)
    split_sizes = [valid_chunk_index[i + 1] - valid_chunk_index[i] for i in range(len(valid_chunk_index) - 1)]

    W_kv_c = W_kv.split(split_sizes, dim=2)
    U_kv_c = U_kv.split(split_sizes, dim=2)

    S_kv = torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
    out_S_kv: list[torch.Tensor] = []

    def _chunk_scan_kv(
        w_kv: torch.Tensor, u_kv: torch.Tensor, s_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_len = w_kv.shape[2]
        s_kv_list: list[torch.Tensor] = []
        for t in range(c_len):
            s_kv = torch.matmul(s_kv, w_kv[:, :, t]) + u_kv[:, :, t]
            s_kv_list.append(s_kv)
        return torch.stack(s_kv_list, dim=2), s_kv

    for i in range(len(split_sizes)):
        s_kv_all, S_kv = _chunk_scan_kv(W_kv_c[i], U_kv_c[i], S_kv)
        out_S_kv.append(s_kv_all)

    S_kv_all = torch.cat(out_S_kv, dim=2)

    # =========================================================================
    # Phase 3: PARALLEL OUTPUT PROJECTION  (no denominator)
    # =========================================================================
    out = torch.matmul(S_kv_all, q_rot)  # (B, H, T, D, S)

    return out.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


class _GDNUCPEBase(GDN):
    """Shared camera-branch logic for all GDN + UCPE variants.

    Adds a second attention branch whose positional encoding comes from UCPE per-ray camera transforms instead of the
    standard RoPE used by the main branch.

    **Camera-specific parameters** (4 Linear layers per block):
        ``q_proj_cam``, ``k_proj_cam``, ``v_proj_cam``, ``out_proj_cam``

    **Shared with main branch** (no duplication):
        QK norms, GDN gates (beta/gate/dt_bias/A_log/recall_gate), output gate, output projection.

    Requires ``cam_dim == in_dim`` and ``cam_heads == heads`` so that all shared parameters have matching dimensions.

    Subclasses only need to override ``_forward_cam_branch`` when the camera branch requires a different recurrence
    pattern (e.g. bidirectional or chunk-causal).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        cam_dim: int,
        cam_heads: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        **kwargs: object,
    ) -> None:
        super().__init__(in_dim, out_dim, **kwargs)

        self.patch_size = patch_size
        self.cam_dim = cam_dim
        self.cam_heads = cam_heads
        self.cam_head_dim = cam_dim // cam_heads

        if cam_dim != in_dim:
            raise ValueError(f"Parameter sharing requires cam_dim == in_dim, got cam_dim={cam_dim}, in_dim={in_dim}.")
        if cam_heads != self.heads:
            raise ValueError(
                f"Parameter sharing requires cam_heads == heads, got cam_heads={cam_heads}, heads={self.heads}."
            )
        if self.cam_head_dim % 4 != 0:
            raise ValueError(
                "UCPE camera branch requires cam_head_dim divisible by 4, "
                f"got {self.cam_head_dim} (cam_dim={cam_dim}, cam_heads={cam_heads})."
            )

        # ---- Camera-specific: QKV + output projections only ----
        self.q_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.k_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.v_proj_cam = nn.Linear(in_dim, cam_dim, bias=True)
        self.out_proj_cam = nn.Linear(cam_dim, out_dim, bias=True)

        # Keep branch-specific Q/K norms so camera statistics do not disturb the
        # main branch (and vice versa). Start from identical weights.
        self.q_norm_cam = deepcopy(self.q_norm)
        self.k_norm_cam = deepcopy(self.k_norm)

        nn.init.constant_(self.out_proj_cam.weight, 0)
        nn.init.constant_(self.out_proj_cam.bias, 0)

        # Short convolutions for camera branch (matching base GDN variant).
        if self.conv_kernel_size > 0:
            self.conv_k_cam = ShortConvolution(
                hidden_size=cam_dim,
                kernel_size=self.conv_kernel_size,
                activation=None,
            )
            if self.k_conv_only:
                self.conv_q_cam = None
                self.conv_v_cam = None
            else:
                self.conv_q_cam = ShortConvolution(
                    hidden_size=cam_dim,
                    kernel_size=self.conv_kernel_size,
                    activation=None,
                )
                self.conv_v_cam = ShortConvolution(
                    hidden_size=cam_dim,
                    kernel_size=self.conv_kernel_size,
                    activation=None,
                )
        else:
            self.conv_q_cam = None
            self.conv_k_cam = None
            self.conv_v_cam = None

    @staticmethod
    def _downscale_to_reference_rms(
        ref: torch.Tensor,
        transformed: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Downscale transformed tensor if its channel RMS exceeds reference.

        Args:
            ref: Reference tensor with target magnitude, shape (B, H, D, N).
            transformed: Tensor to stabilize, shape (B, H, D, N).
            eps: Numerical epsilon for RMS.

        Returns:
            Stabilized tensor with per-(B,H,N) channel RMS not larger than ref.
        """
        ref_rms = ref.square().mean(dim=2, keepdim=True).add(eps).sqrt()
        tr_rms = transformed.square().mean(dim=2, keepdim=True).add(eps).sqrt()
        scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0)
        return transformed * scale

    def _stabilize_cam_transforms(
        self,
        q_cam: torch.Tensor,
        k_cam: torch.Tensor,
        v_cam: torch.Tensor,
        q_cam_trans: torch.Tensor,
        k_cam_trans: torch.Tensor,
        v_cam_trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optional post-UCPE stabilization hook for experimental variants."""
        del q_cam, k_cam, v_cam
        return q_cam_trans, k_cam_trans, v_cam_trans

    # ------------------------------------------------------------------
    # Camera-branch building blocks
    # ------------------------------------------------------------------

    def _prepare_cam_qkv(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        *,
        token_valid_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple:
        """Project camera QKV, apply short conv + QK norm + kernel + scaling + UCPE.

        The processing order mirrors the base GDN branch:
          project -> mask -> short_conv -> QK_norm -> kernel -> scale -> permute -> UCPE

        Args:
            token_valid_mask: Pre-computed mask of shape ``(B, N)`` from the
                caller. Avoids redundant ``_prepare_frame_valid_masks`` calls.

        Returns:
            (q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, out_transform, inflation_sq)

        All tensors are shaped ``(B, cam_heads, cam_head_dim, N)``. ``out_transform`` is ``(P, rotary_emb_cam)``, the
        arguments :func:`_apply_ucpe_transform` needs for the inverse-output transform closure. ``inflation_sq`` is the
        energy inflation factor of shape ``(B, cam_heads, 1, N)``.
        """
        B, N, C = x.shape
        T, H, W = HW
        S = H * W

        # Pre-projection token masking (matching base branch).
        if token_valid_mask is not None:
            x = x * token_valid_mask.view(B, N, 1)

        # Fused camera QKV projection (1 GEMM instead of 3 kernel launches).
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = F.linear(x, qkv_w, qkv_b)
        q_cam, k_cam, v_cam = qkv_cam.chunk(3, dim=-1)

        # Post-projection token masking (before conv, matching base branch).
        if token_valid_mask is not None:
            token_mask = token_valid_mask.view(B, N, 1)
            q_cam = q_cam * token_mask
            k_cam = k_cam * token_mask
            v_cam = v_cam * token_mask

        # Short convolution along T (before norm / kernel activation).
        if self.conv_q_cam is not None:
            q_cam = self._apply_temporal_short_conv(q_cam, self.conv_q_cam, HW, **kwargs)
        if self.conv_k_cam is not None:
            k_cam = self._apply_temporal_short_conv(k_cam, self.conv_k_cam, HW, **kwargs)
        if self.conv_v_cam is not None:
            v_cam = self._apply_temporal_short_conv(v_cam, self.conv_v_cam, HW, **kwargs)

        # Camera-specific QK normalization.
        q_cam = self.q_norm_cam(q_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
        k_cam = self.k_norm_cam(k_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
        v_cam = v_cam.reshape(B, N, self.cam_heads, self.cam_head_dim)

        # ReLU kernel (shared).
        q_cam = self.kernel_func(q_cam)
        k_cam = self.kernel_func(k_cam)

        # FIXED: K scaling -- explicitly use ** for exponentiation!
        k_scale = (self.cam_head_dim**-0.5) * (S**-0.5)
        k_cam = k_cam * k_scale

        # Permute to (B, H, D, N) for GDN processing.
        q_cam = q_cam.permute(0, 2, 3, 1).contiguous()
        k_cam = k_cam.permute(0, 2, 3, 1).contiguous()
        v_cam = v_cam.permute(0, 2, 3, 1).contiguous()

        # Measure safe geometric norm before UCPE applies translations
        pre_ucpe_k_norm = torch.linalg.vector_norm(k_cam, dim=2, keepdim=True).clamp_min(1e-6)

        # UCPE per-ray transforms — reuse model-level cache when available
        # to avoid recomputing _process_camera_conditions_ucpe per block.
        ray_transforms = kwargs.get("ucpe_ray_transforms", None)
        if ray_transforms is None:
            ray_transforms = _prepare_ucpe_ray_transforms(
                head_dim=self.cam_head_dim,
                camera_conditions=camera_conditions,
                HW=HW,
                patch_size=self.patch_size,
                rotary_emb=rotary_emb,
            )
        P, P_T, P_inv, rotary_emb_cam = ray_transforms

        # UCPE expects (B, h, N, d); our tensors are (B, h, d, N). Avoid eager contiguous copies before the
        # transforms, and fuse the K/V transform (both use P_inv) into one call, then split back.
        q_cam_trans = (
            _apply_ucpe_transform(q_cam.transpose(-1, -2), P_T, rotary_emb_cam).transpose(-1, -2).contiguous()
        )
        kv_cam = torch.cat([k_cam, v_cam], dim=1)
        kv_cam_trans = (
            _apply_ucpe_transform(kv_cam.transpose(-1, -2), P_inv, rotary_emb_cam).transpose(-1, -2).contiguous()
        )
        k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

        q_cam_trans, k_cam_trans, v_cam_trans = self._stabilize_cam_transforms(
            q_cam=q_cam,
            k_cam=k_cam,
            v_cam=v_cam,
            q_cam_trans=q_cam_trans,
            k_cam_trans=k_cam_trans,
            v_cam_trans=v_cam_trans,
        )

        # Measure inflated geometric norm after UCPE
        post_ucpe_k_norm = torch.linalg.vector_norm(k_cam_trans, dim=2, keepdim=True).clamp_min(1e-6)

        # Calculate the squared inflation factor for beta discounting
        inflation_sq = (post_ucpe_k_norm / pre_ucpe_k_norm) ** 2

        return q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, (P, rotary_emb_cam), inflation_sq

    def _run_cam_gdn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """Run the shared GDN kernel on camera-branch tensors.

        Uses shared ``self.recall_gate``. Handles FP32 casting. Returns ``num / (den + eps)`` shaped ``(B, H, D, N)``.
        """
        recall_gate = self.recall_gate
        q = q.float()
        k = k.float()
        v = v.float()
        q_rot = q_rot.float()
        k_rot = k_rot.float()
        beta = beta.float()
        decay = decay.float()
        recall_gate = recall_gate.float()

        return torch_chunk_sana_gdn(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=recall_gate,
            chunk_size=self.chunk_gdn_chunk_size,
            eps=self.eps,
        )

    def _run_cam_gdn_components(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like ``_run_cam_gdn`` but returns ``(num, den)`` components."""
        recall_gate = self.recall_gate
        q = q.float()
        k = k.float()
        v = v.float()
        q_rot = q_rot.float()
        k_rot = k_rot.float()
        beta = beta.float()
        decay = decay.float()
        recall_gate = recall_gate.float()

        return torch_chunk_sana_gdn(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            recall_gate=recall_gate,
            chunk_size=self.chunk_gdn_chunk_size,
            eps=self.eps,
            return_components=True,
        )

    def _run_cam_single_path(
        self,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """Run the numerator-only camera delta-rule recurrence (parallel chunk scan)."""
        q_rot = q_rot.float()
        k_rot = k_rot.float()
        v = v.float()
        beta = beta.float()
        decay = decay.float()
        return torch_chunk_cam_single_path_delta_rule(
            q_rot, k_rot, v, beta, decay, chunk_size=self.chunk_gdn_chunk_size
        )

    # ------------------------------------------------------------------
    # Camera-branch forward (forward-only causal -- default)
    # ------------------------------------------------------------------

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward-only causal GDN camera branch with UCPE transforms.

        Subclasses override this for bidirectional / chunk-causal variants.

        Returns raw attention output ``(B, N, C)`` -- no output gate or projection applied (those are shared and
        applied in ``forward()``).
        """
        B, N, _ = x.shape
        T, H, W = HW
        S = H * W
        dtype_orig = x.dtype

        # Compute masks once; pass token_valid_mask to _prepare_cam_qkv for
        # pre-conv masking and reuse here for post-UCPE masking + gate masking.
        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None),
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )

        q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, out_transform, inflation_sq = self._prepare_cam_qkv(
            x,
            HW,
            camera_conditions,
            rotary_emb,
            token_valid_mask=token_valid_mask,
            **kwargs,
        )

        # Re-mask after UCPE transforms (which can reintroduce non-zero values).
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_cam = q_cam * token_mask_qkv
            k_cam = k_cam * token_mask_qkv
            v_cam_trans = v_cam_trans * token_mask_qkv
            q_cam_trans = q_cam_trans * token_mask_qkv
            k_cam_trans = k_cam_trans * token_mask_qkv

        # Shared GDN gates (use pre-computed when available).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        # Dynamic Beta Discounting: scale beta by UCPE inflation factor.
        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        out = self._run_cam_gdn(
            q_cam,
            k_cam,
            v_cam_trans,
            q_cam_trans,
            k_cam_trans,
            beta,
            decay,
        )

        if dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        # Inverse UCPE transform on output.
        out = (
            _apply_ucpe_transform(out.transpose(-1, -2), *out_transform, inverse_rope=True)
            .transpose(-1, -2)
            .contiguous()
        )
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Dual-branch forward: GDN main + UCPE camera.

        Flow:
            1. main_raw = GDN attention (no gate/proj)
            2. cam_raw = GDN+UCPE attention (no gate/proj)
            3. combined = main_raw + out_proj_cam(cam_raw) [zero at init]
            4. output = proj(output_gate(combined)) [shared, once]
        """
        # Pre-compute shared gates once for both branches.
        if HW is not None:
            precomputed_gates = self._compute_frame_gates(x, HW)
        else:
            precomputed_gates = None

        # Main branch -- raw attention without gate/proj.
        main_raw = super().forward(
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            precomputed_gates=precomputed_gates,
            **kwargs,
        )

        # Camera branch.
        cam_contrib: torch.Tensor | int = 0
        if camera_conditions is not None:
            if HW is None:
                raise ValueError("HW (T, H, W) must be provided for UCPE camera branch.")
            cam_raw = self._forward_cam_branch(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                chunk_size=chunk_size,
                precomputed_gates=precomputed_gates,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        # Combine, then shared gate + projection (applied once).
        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(x.dtype))


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class BidirectionalGDNUCPELiteLA(_GDNUCPEBase, BidirectionalGDN):
    """Bidirectional GDN with UCPE camera conditioning.

    Main branch: bidirectional GDN (inherited from ``BidirectionalGDN``). Camera branch: bidirectional GDN with UCPE
    transforms.
    """

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        dtype_orig = x.dtype

        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None),
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )

        q_cam, k_cam, v_cam_trans, q_cam_trans, k_cam_trans, out_transform, inflation_sq = self._prepare_cam_qkv(
            x,
            HW,
            camera_conditions,
            rotary_emb,
            token_valid_mask=token_valid_mask,
            **kwargs,
        )
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_cam = q_cam * token_mask_qkv
            k_cam = k_cam * token_mask_qkv
            v_cam_trans = v_cam_trans * token_mask_qkv
            q_cam_trans = q_cam_trans * token_mask_qkv
            k_cam_trans = k_cam_trans * token_mask_qkv

        # Shared GDN gates (use pre-computed when available).
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        # Dynamic Beta Discounting: scale beta by UCPE inflation factor.
        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        # -- Forward pass (inclusive 1..t) --
        num_fwd, den_fwd = self._run_cam_gdn_components(
            q_cam,
            k_cam,
            v_cam_trans,
            q_cam_trans,
            k_cam_trans,
            beta,
            decay,
        )

        # -- Backward pass (exclusive t+1..T) --
        def to_time(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4)

        def from_time(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 1, 3, 2, 4).reshape(B, H_heads, D_head, N)

        q_T = to_time(q_cam)
        k_T = to_time(k_cam)
        v_T = to_time(v_cam_trans)
        q_rot_T = to_time(q_cam_trans)
        k_rot_T = to_time(k_cam_trans)

        q_bwd = torch.flip(q_T, dims=[2])
        q_rot_bwd = torch.flip(q_rot_T, dims=[2])
        k_bwd = flip_and_shift(k_T, dim=2, shift_val=0.0)
        v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
        k_rot_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        num_bwd_f, den_bwd_f = self._run_cam_gdn_components(
            from_time(q_bwd),
            from_time(k_bwd),
            from_time(v_bwd),
            from_time(q_rot_bwd),
            from_time(k_rot_bwd),
            beta_bwd,
            decay_bwd,
        )

        def flip_back(tensor: torch.Tensor) -> torch.Tensor:
            d = tensor.shape[2]
            return torch.flip(
                tensor.view(B, H_heads, d, T, S),
                dims=[3],
            ).reshape(B, H_heads, d, N)

        num_bwd = flip_back(num_bwd_f)
        den_bwd = flip_back(den_bwd_f)
        out = (num_fwd + num_bwd) / (den_fwd + den_bwd + self.eps)

        if dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        out = (
            _apply_ucpe_transform(out.transpose(-1, -2), *out_transform, inverse_rope=True)
            .transpose(-1, -2)
            .contiguous()
        )
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out


class BidirectionalGDNUCPELiteLAPostUCPERenorm(BidirectionalGDNUCPELiteLA):
    """Bidirectional GDNUCPE with post-UCPE RMS downscaling.

    The raw UCPE transforms are still measured for debug logging, but the transformed camera tensors are downscaled
    back to their pre-UCPE RMS envelope before they enter the recurrence.
    """

    def _stabilize_cam_transforms(
        self,
        q_cam: torch.Tensor,
        k_cam: torch.Tensor,
        v_cam: torch.Tensor,
        q_cam_trans: torch.Tensor,
        k_cam_trans: torch.Tensor,
        v_cam_trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_cam_trans = self._downscale_to_reference_rms(q_cam, q_cam_trans)
        k_cam_trans = self._downscale_to_reference_rms(k_cam, k_cam_trans)
        v_cam_trans = self._downscale_to_reference_rms(v_cam, v_cam_trans)
        return q_cam_trans, k_cam_trans, v_cam_trans


class BidirectionalGDNUCPESinglePathLiteLA(BidirectionalGDNUCPELiteLAPostUCPERenorm):
    """Bidirectional UCPE camera branch with numerator-only delta-rule updates.

    This is an experimental ablation that keeps the main branch unchanged, applies UCPE plus post-UCPE RMS downscaling
    on the camera tensors, and replaces the camera branch's ``num / den`` recurrence with a single-path delta rule over
    the transformed camera stream only.
    """

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        T, H, W = HW
        S = H * W
        dtype_orig = x.dtype

        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None),
            B=B,
            T=T,
            S=S,
            device=x.device,
            dtype=x.dtype,
        )

        q_cam, _, v_cam_trans, q_cam_trans, k_cam_trans, out_transform, inflation_sq = self._prepare_cam_qkv(
            x,
            HW,
            camera_conditions,
            rotary_emb,
            token_valid_mask=token_valid_mask,
            **kwargs,
        )
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(B, 1, 1, N)
            q_cam = q_cam * token_mask_qkv
            v_cam_trans = v_cam_trans * token_mask_qkv
            q_cam_trans = q_cam_trans * token_mask_qkv
            k_cam_trans = k_cam_trans * token_mask_qkv

        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        H_heads = self.cam_heads
        D_head = self.cam_head_dim
        out_fwd = self._run_cam_single_path(
            q_cam_trans,
            k_cam_trans,
            v_cam_trans,
            beta,
            decay,
        )

        def to_time(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4)

        def from_time(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 1, 3, 2, 4).reshape(B, H_heads, D_head, N)

        q_rot_T = to_time(q_cam_trans)
        k_rot_T = to_time(k_cam_trans)
        v_T = to_time(v_cam_trans)

        q_rot_bwd = torch.flip(q_rot_T, dims=[2])
        k_rot_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
        v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        out_bwd_f = self._run_cam_single_path(
            from_time(q_rot_bwd),
            from_time(k_rot_bwd),
            from_time(v_bwd),
            beta_bwd,
            decay_bwd,
        )

        out_bwd = torch.flip(
            out_bwd_f.view(B, H_heads, D_head, T, S),
            dims=[3],
        ).reshape(B, H_heads, D_head, N)
        out = out_fwd + out_bwd

        if dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        out = (
            _apply_ucpe_transform(out.transpose(-1, -2), *out_transform, inverse_rope=True)
            .transpose(-1, -2)
            .contiguous()
        )
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out


def _prepare_cam_qkv_softmax(
    self,
    x: torch.Tensor,
    HW: tuple,
    camera_conditions: torch.Tensor,
    rotary_emb: torch.Tensor | None,
    *,
    token_valid_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple:
    """Camera branch Q/K/V for softmax attention.

    Mirrors ``_GDNUCPEBase._prepare_cam_qkv`` but skips the ReLU kernel and GDN key scaling — standard softmax SDPA
    provides its own 1/sqrt(d_k). Returns ``(q, k, v, out_transform)``, where the tensors are shaped ``(B, cam_heads,
    cam_head_dim, N)`` and ``out_transform`` is ``(P, rotary_emb_cam)``.
    """
    B, N, C = x.shape

    if token_valid_mask is not None:
        x = x * token_valid_mask.view(B, N, 1)

    qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
    qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
    qkv_cam = F.linear(x, qkv_w, qkv_b)
    q_cam, k_cam, v_cam = qkv_cam.chunk(3, dim=-1)

    if token_valid_mask is not None:
        m = token_valid_mask.view(B, N, 1)
        q_cam, k_cam, v_cam = q_cam * m, k_cam * m, v_cam * m

    if self.conv_q_cam is not None:
        q_cam = self._apply_temporal_short_conv(q_cam, self.conv_q_cam, HW, **kwargs)
    if self.conv_k_cam is not None:
        k_cam = self._apply_temporal_short_conv(k_cam, self.conv_k_cam, HW, **kwargs)
    if self.conv_v_cam is not None:
        v_cam = self._apply_temporal_short_conv(v_cam, self.conv_v_cam, HW, **kwargs)

    q_cam = self.q_norm_cam(q_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
    k_cam = self.k_norm_cam(k_cam).reshape(B, N, self.cam_heads, self.cam_head_dim)
    v_cam = v_cam.reshape(B, N, self.cam_heads, self.cam_head_dim)

    q_cam = q_cam.permute(0, 2, 3, 1).contiguous()
    k_cam = k_cam.permute(0, 2, 3, 1).contiguous()
    v_cam = v_cam.permute(0, 2, 3, 1).contiguous()

    ray_transforms = kwargs.get("ucpe_ray_transforms", None)
    if ray_transforms is None:
        ray_transforms = _prepare_ucpe_ray_transforms(
            head_dim=self.cam_head_dim,
            camera_conditions=camera_conditions,
            HW=HW,
            patch_size=self.patch_size,
            rotary_emb=rotary_emb,
        )
    P, P_T, P_inv, rotary_emb_cam = ray_transforms

    q_cam_trans = _apply_ucpe_transform(q_cam.transpose(-1, -2), P_T, rotary_emb_cam).transpose(-1, -2).contiguous()
    kv_cam = torch.cat([k_cam, v_cam], dim=1)
    kv_cam_trans = (
        _apply_ucpe_transform(kv_cam.transpose(-1, -2), P_inv, rotary_emb_cam).transpose(-1, -2).contiguous()
    )
    k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

    q_cam_trans, k_cam_trans, v_cam_trans = self._stabilize_cam_transforms(
        q_cam=q_cam,
        k_cam=k_cam,
        v_cam=v_cam,
        q_cam_trans=q_cam_trans,
        k_cam_trans=k_cam_trans,
        v_cam_trans=v_cam_trans,
    )
    return q_cam_trans, k_cam_trans, v_cam_trans, (P, rotary_emb_cam)


def _forward_cam_branch_softmax(
    self,
    x: torch.Tensor,
    HW: tuple,
    camera_conditions: torch.Tensor,
    rotary_emb: torch.Tensor | None,
    frame_causal: bool,
    **kwargs,
) -> torch.Tensor:
    """Bidirectional softmax camera branch (with UCPE transforms).

    Uses ``F.scaled_dot_product_attention`` with optional invalid-key masking.
    """
    B, N, _ = x.shape
    T, H, W = HW
    S = H * W

    token_valid_mask, _, _ = self._prepare_frame_valid_masks(
        kwargs.get("frame_valid_mask", None),
        B=B,
        T=T,
        S=S,
        device=x.device,
        dtype=x.dtype,
    )

    q_cam_trans, k_cam_trans, v_cam_trans, out_transform = _prepare_cam_qkv_softmax(
        self,
        x,
        HW,
        camera_conditions,
        rotary_emb,
        token_valid_mask=token_valid_mask,
        **kwargs,
    )

    if token_valid_mask is not None:
        m = token_valid_mask.view(B, 1, 1, N)
        q_cam_trans, v_cam_trans = q_cam_trans * m, v_cam_trans * m

    q_sdpa = q_cam_trans.transpose(-1, -2)
    k_sdpa = k_cam_trans.transpose(-1, -2)
    v_sdpa = v_cam_trans.transpose(-1, -2)

    dtype_orig = x.dtype
    q_sdpa, k_sdpa, v_sdpa = q_sdpa.float(), k_sdpa.float(), v_sdpa.float()
    # SDPA / FlashAttention only supports bf16/fp16; fp32 falls back to math backend.
    if q_sdpa.dtype == torch.float32:
        q_sdpa, k_sdpa, v_sdpa = q_sdpa.bfloat16(), k_sdpa.bfloat16(), v_sdpa.bfloat16()

    invalid_kv_logit_bias = None
    if token_valid_mask is not None and not bool(token_valid_mask.all()):
        invalid_kv_logit_bias = torch.where(
            token_valid_mask.bool().view(B, 1, 1, -1),
            torch.zeros((), dtype=q_sdpa.dtype, device=q_sdpa.device),
            torch.full((), -1e9, dtype=q_sdpa.dtype, device=q_sdpa.device),
        )

    # FlashAttention-2 only supports head_dim in {32, 64, 128, 256}.
    D = q_sdpa.shape[-1]
    _need_pad = D not in (32, 64, 128, 256) and D < 256
    if _need_pad:
        _pad_to = 128 if D <= 128 else 256
        _pad_size = _pad_to - D
        q_sdpa = F.pad(q_sdpa, (0, _pad_size))
        k_sdpa = F.pad(k_sdpa, (0, _pad_size))
        v_sdpa = F.pad(v_sdpa, (0, _pad_size))
    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=invalid_kv_logit_bias)
    if _need_pad:
        out = out[..., :D]

    out = out.transpose(-1, -2)
    if out.dtype != dtype_orig:
        out = out.to(dtype_orig)
    if token_valid_mask is not None:
        out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)
    out = (
        _apply_ucpe_transform(out.transpose(-1, -2), *out_transform, inverse_rope=True).transpose(-1, -2).contiguous()
    )
    out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
    if token_valid_mask is not None:
        out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
    return out


class _SoftmaxUCPESinglePathLiteLA(
    BidirectionalGDNUCPESinglePathLiteLA,
):
    """Softmax attention with UCPE camera conditioning (single-path).

    Replaces GDN recurrence with ``F.scaled_dot_product_attention``. Automatically selects the correct masking mode
    based on ``chunk_size``:

    - ``chunk_size is None`` or ``chunk_size >= T``: full bidirectional (no mask)
    - ``chunk_size < T``: chunk-causal (full within chunks, causal across)

    All parameters match the GDN variants for checkpoint compatibility. GDN-specific parameters are present but unused
    in forward.
    """

    def __init__(self, *args, conv_kernel_size: int = 0, **kwargs):
        super().__init__(*args, conv_kernel_size=0, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        main_raw = _forward_softmax_attn(
            self,
            x,
            HW,
            rotary_emb,
            frame_causal=False,
            apply_output_gate=False,
            chunk_size=chunk_size,
            **kwargs,
        )

        cam_contrib: torch.Tensor | int = 0
        if camera_conditions is not None:
            if HW is None:
                raise ValueError("HW must be provided for UCPE camera branch.")
            cam_raw = _forward_cam_branch_softmax(
                self,
                x,
                HW,
                camera_conditions,
                rotary_emb,
                frame_causal=False,
                chunk_size=chunk_size,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(x.dtype))


# Name used by the `camctrl_type` config string and the block-name mappings below.
BidirectionalSoftmaxUCPESinglePathLiteLA = _SoftmaxUCPESinglePathLiteLA


# The released `config.json` names the fused-Triton variants (`attn_type="BidirectionalGDNTriton"`,
# `camctrl_type="BidirectionalGDNUCPESinglePathLiteLABothTriton"`). The Triton kernels now live outside
# `diffusers`, so those names resolve to the equivalent pure-PyTorch implementations.
ATTENTION_BLOCKS.update(
    {
        "GDN": GDN,
        "BidirectionalGDN": BidirectionalGDN,
        "BidirectionalGDNTriton": BidirectionalGDN,
        "BidirectionalGDNUCPESinglePathLiteLA": BidirectionalGDNUCPESinglePathLiteLA,
        "BidirectionalGDNUCPESinglePathLiteLATriton": BidirectionalGDNUCPESinglePathLiteLA,
        "BidirectionalGDNUCPESinglePathLiteLABothTriton": BidirectionalGDNUCPESinglePathLiteLA,
    }
)


# ============================================================================
# DiT base + SANA-WM camera-controlled transformer + public wrapper
# ============================================================================


class SanaVideoMSCamCtrlBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        t_kernel_size=3,
        camctrl_type=None,
        patch_size=(1, 2, 2),
        cam_attn_compress=2,
        chunk_size=10,
        chunk_split_strategy="uniform",
        use_chunk_plucker_post_attn=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.chunk_split_strategy = chunk_split_strategy

        if use_chunk_plucker_post_attn:
            self.plucker_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.plucker_proj.weight)
            nn.init.zeros_(self.plucker_proj.bias)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Camera-branch attention. The legacy ``*Triton`` config strings resolve to the same pure-PyTorch class.
        if camctrl_type in (
            "BidirectionalGDNUCPESinglePathLiteLABothTriton",
            "BidirectionalGDNUCPESinglePathLiteLATriton",
            "BidirectionalGDNUCPESinglePathLiteLA",
        ):
            self_num_heads = hidden_size // linear_head_dim
            cam_cls = _resolve_attention_block(camctrl_type, role="camctrl_type")
            self.attn = cam_cls(
                hidden_size,
                hidden_size,
                heads=self_num_heads,
                cam_dim=hidden_size // cam_attn_compress,
                cam_heads=max(1, self_num_heads // cam_attn_compress),
                eps=1e-8,
                qk_norm=qk_norm,
                patch_size=patch_size,
                **block_kwargs,
            )
        elif camctrl_type == "BidirectionalSoftmaxUCPESinglePathLiteLA":
            self_num_heads = hidden_size // linear_head_dim
            self.attn = BidirectionalSoftmaxUCPESinglePathLiteLA(
                hidden_size,
                hidden_size,
                heads=self_num_heads,
                cam_dim=hidden_size // cam_attn_compress,
                cam_heads=max(1, self_num_heads // cam_attn_compress),
                eps=1e-8,
                qk_norm=qk_norm,
                patch_size=patch_size,
                **block_kwargs,
            )
        else:
            # Main attention (no camera branch).
            attn_cls = _resolve_attention_block(attn_type, role="attn_type")
            self.attn = attn_cls(
                hidden_size,
                hidden_size,
                heads=hidden_size // linear_head_dim,
                eps=1e-8,
                qk_norm=qk_norm,
            )

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # MLP
        if ffn_type == "GLUMBConvTemp":
            self.mlp = GLUMBConvTemp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                act=mlp_acts,
                t_kernel_size=t_kernel_size,
            )
        elif ffn_type == "mlp":

            def approx_gelu():
                return nn.GELU(approximate="tanh")

            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            self.mlp = None

        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    @staticmethod
    def _build_frame_token_mask(
        frame_valid_mask: Optional[torch.Tensor],
        *,
        B: int,
        T: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Convert frame-valid mask to token mask shaped ``(B, N, 1)``."""
        if frame_valid_mask is None:
            return None

        m = frame_valid_mask
        if m.ndim == 5:
            m = m[:, 0, :, 0, 0]
        elif m.ndim == 3 and m.shape[1] == 1:
            m = m[:, 0, :]
        elif m.ndim != 2:
            raise ValueError(
                "frame_valid_mask must be shaped (B, 1, T, 1, 1), (B, 1, T), or (B, T); "
                f"got shape={list(frame_valid_mask.shape)}"
            )

        if m.shape[0] != B or m.shape[1] != T:
            raise ValueError(f"frame_valid_mask shape mismatch: expected (B={B}, T={T}), got {list(m.shape)}")
        if T <= 0 or N % T != 0:
            raise ValueError(f"Invalid token/frame layout: N={N}, T={T}")

        S = N // T
        return m.to(device=device, dtype=dtype).view(B, T, 1).expand(B, T, S).reshape(B, N, 1)

    def forward(self, x, y, t, mask=None, THW=None, rotary_emb=None, block_mask=None, chunk_index=None, **kwargs):
        B, N, C = x.shape
        num_frames = t.shape[2]
        frame_valid_mask = kwargs.get("frame_valid_mask", None)
        frame_token_mask = self._build_frame_token_mask(
            frame_valid_mask,
            B=B,
            T=num_frames,
            N=N,
            device=x.device,
            dtype=x.dtype,
        )
        if frame_token_mask is not None:
            x = x * frame_token_mask

        t = t.reshape(B, num_frames, 6, -1)  # B,F,6,D
        # scale_shift_table: 6, hidden_size -> 1,1,6,hidden_size
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :, :] + t
        ).chunk(6, dim=-2)  # each chunk: B,F,1,D
        self_attn_kwargs = {
            "HW": THW,
            "rotary_emb": rotary_emb,
            "block_mask": block_mask,
            "camera_conditions": kwargs.get("camera_conditions", None),
            "ucpe_ray_transforms": kwargs.get("ucpe_ray_transforms", None),
            "camera_embedding": kwargs.get("camera_embedding", None),
            "frame_valid_mask": frame_valid_mask,
        }
        if chunk_index is not None:
            self_attn_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        if kwargs.get("chunk_index_global", None) is not None:
            self_attn_kwargs["chunk_index_global"] = kwargs.get("chunk_index_global")
        chunk_split_strategy = kwargs.get("chunk_split_strategy", self.chunk_split_strategy)
        if chunk_split_strategy is not None:
            self_attn_kwargs["chunk_split_strategy"] = chunk_split_strategy

        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        if chunk_size is not None:
            self_attn_kwargs["chunk_size"] = chunk_size

        x_norm1 = self.norm1(x).reshape(B, num_frames, -1, C)
        x_msa_in = (x_norm1 * (1 + scale_msa) + shift_msa).reshape(B, N, C)
        if frame_token_mask is not None:
            x_msa_in = x_msa_in * frame_token_mask
        attn_out = self.attn(x_msa_in, **self_attn_kwargs).reshape(B, num_frames, -1, C)
        attn_out = (gate_msa * attn_out).reshape(B, N, C)
        if frame_token_mask is not None:
            attn_out = attn_out * frame_token_mask
        x = x + attn_out
        if frame_token_mask is not None:
            x = x * frame_token_mask

        plucker_emb = kwargs.get("plucker_emb", None)
        if plucker_emb is not None and hasattr(self, "plucker_proj"):
            x = x + self.plucker_proj(plucker_emb)

        x = x + self.cross_attn(x, y, mask=mask)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        mlp_kwargs = {
            "HW": THW,
            "frame_valid_mask": frame_valid_mask,
        }
        if chunk_index is not None:
            mlp_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        if kwargs.get("chunk_index_global", None) is not None:
            mlp_kwargs["chunk_index_global"] = kwargs.get("chunk_index_global")
        if chunk_split_strategy is not None:
            mlp_kwargs["chunk_split_strategy"] = chunk_split_strategy

        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        if chunk_size is not None:
            mlp_kwargs["chunk_size"] = chunk_size

        x_norm2 = self.norm2(x).reshape(B, num_frames, -1, C)
        x_mlp_in = (x_norm2 * (1 + scale_mlp) + shift_mlp).reshape(B, N, C)
        if frame_token_mask is not None:
            x_mlp_in = x_mlp_in * frame_token_mask
        mlp_out = self.mlp(x_mlp_in, **mlp_kwargs).reshape(B, num_frames, -1, C)
        mlp_out = (gate_mlp * mlp_out).reshape(B, N, C)
        if frame_token_mask is not None:
            mlp_out = mlp_out * frame_token_mask
        x = x + mlp_out
        if frame_token_mask is not None:
            x = x * frame_token_mask

        return x


_GDN_TO_SOFTMAX_CAMCTRL: dict[str, str] = {
    "BidirectionalGDNUCPESinglePathLiteLABothTriton": "BidirectionalSoftmaxUCPESinglePathLiteLA",
}


def _inject_softmax_layers(
    attn_type_list: list,
    camctrl_type_list: list,
    softmax_every_n: int,
) -> tuple:
    """Replace every ``softmax_every_n``-th block's camctrl variant with its softmax counterpart.

    Pattern: for ``softmax_every_n=4``, blocks 3, 7, 11, ... (0-indexed at n-1) use softmax attention; the remaining
    blocks keep GDN. Blocks whose camctrl_type has no softmax mapping are left as-is.
    """
    attn_out = list(attn_type_list)
    camctrl_out = list(camctrl_type_list)
    for i in range(len(attn_out)):
        if (i + 1) % softmax_every_n != 0:
            continue
        if camctrl_out[i] in _GDN_TO_SOFTMAX_CAMCTRL:
            camctrl_out[i] = _GDN_TO_SOFTMAX_CAMCTRL[camctrl_out[i]]
    return attn_out, camctrl_out


class SanaWMTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    SANA-WM 1600M bidirectional camera-controlled DiT.

    A single-class DiT (depth=20, hidden_size=2240, patch_size=(1,1,1), num_heads=20 — i.e. the public
    ``Efficient-Large-Model/SANA-WM_bidirectional`` release). ``save_pretrained`` / ``from_pretrained`` work out of the
    box via :class:`~diffusers.configuration_utils.ConfigMixin`.

    Args:
        in_channels (`int`, defaults to 128): VAE latent channels (LTX-2).
        attn_type (`str`): Main-branch attention, e.g. ``"BidirectionalGDN"``. The released config uses the legacy
            ``"BidirectionalGDNTriton"`` name, which maps onto the same pure-PyTorch class.
        camctrl_type (`str`): Camera-branch attention, e.g. ``"BidirectionalGDNUCPESinglePathLiteLA"``. The released
            config uses the legacy ``"BidirectionalGDNUCPESinglePathLiteLABothTriton"`` name, which maps onto the same
            pure-PyTorch class.
        softmax_every_n (`int`, defaults to 4): Inject a softmax block every N blocks.
        linear_head_dim (`int`, defaults to 112): GDN head dimension.
        ffn_type (`str`, defaults to ``"GLUMBConvTemp"``): FFN.
        t_kernel_size (`int`, defaults to 3): Temporal conv kernel.
        conv_kernel_size (`int`, defaults to 4): Spatial conv kernel inside attention.
        k_conv_only (`bool`, defaults to True): Apply conv only on K.
        pos_embed_type (`str`, defaults to ``"wan_rope"``): Position embedding.
        qk_norm (`bool`, defaults to True): RMSNorm on Q/K.
        cross_norm (`bool`, defaults to True): RMSNorm on cross-attention K.
        y_norm (`bool`, defaults to True): Apply ``attention_y_norm`` to text embeddings.
        y_norm_scale_factor (`float`, defaults to 0.01): Scale factor for ``attention_y_norm``.
        init_cam_from_base (`bool`, defaults to True): Unused; the camera branch is loaded from the checkpoint.
            Kept so released `config.json` files load.
        chunk_split_strategy (`str`, defaults to ``"first_chunk_plus_one"``).
        use_chunk_plucker_post_attn (`bool`, defaults to True).
        chunk_plucker_channels (`int`, defaults to 48): ``6 dims * temporal_stride 8``.
        chunk_plucker_post_attn_blocks (`int`, defaults to 20): All blocks.
        fp32_attention (`bool`, defaults to True): Unused; attention always runs in fp32. Kept so released
            `config.json` files load.
        image_size (`int`, defaults to 720): Nominal image size.
        caption_channels (`int`, defaults to 2304): Gemma-2 hidden size.
        model_max_length (`int`, defaults to 300): Max prompt tokens.

    The state-dict is identical to the public sana checkpoint apart from the intentionally-removed ``pos_embed``
    buffer.
    """

    _supports_gradient_checkpointing = False
    _no_split_modules = ["SanaVideoMSCamCtrlBlock"]
    _repeated_blocks = ["SanaVideoMSCamCtrlBlock"]
    _skip_layerwise_casting_patterns = ["x_embedder", "plucker_embedder", "norm"]
    # NOTE: `_keep_in_fp32_modules` is intentionally unset. SANA-WM's blocks apply the
    # timestep modulation inline, so holding `t_embedder` / `t_block` /
    # `scale_shift_table` in fp32 would upcast the hidden states and feed fp32 activations
    # to bf16 weights. Supporting it needs explicit casts in the block forward first.

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 20,
        hidden_size: int = 2240,
        num_attention_heads: int = 20,
        patch_size: tuple[int, int, int] = (1, 1, 1),
        attn_type: str = "BidirectionalGDNTriton",
        camctrl_type: str = "BidirectionalGDNUCPESinglePathLiteLABothTriton",
        softmax_every_n: int = 4,
        linear_head_dim: int = 112,
        ffn_type: str = "GLUMBConvTemp",
        t_kernel_size: int = 3,
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        pos_embed_type: str = "wan_rope",
        qk_norm: bool = True,
        cross_norm: bool = True,
        y_norm: bool = True,
        y_norm_scale_factor: float = 0.01,
        cam_attn_compress: int = 1,
        init_cam_from_base: bool = True,
        chunk_split_strategy: str = "first_chunk_plus_one",
        use_chunk_plucker_post_attn: bool = True,
        chunk_plucker_channels: int = 48,
        chunk_plucker_post_attn_blocks: int = 20,
        fp32_attention: bool = True,
        image_size: int = 720,
        caption_channels: int = 2304,
        model_max_length: int = 300,
        mlp_ratio: float = 3.0,
        mlp_acts: tuple = ("silu", "silu", None),
        use_pe: bool = True,
        learn_sigma: bool = False,
        pred_sigma: bool = False,
        mixed_precision: str = "bf16",
    ) -> None:
        super().__init__()

        # The defaults describe the public SANA-WM_bidirectional release; they are
        # configurable so a small variant can be built (e.g. for tests).
        depth = num_layers
        num_heads = num_attention_heads
        patch_size = tuple(patch_size)

        # Remaining SanaMSVideoCamCtrl.__init__ defaults not exposed by the config signature.
        mlp_acts = list(mlp_acts)
        pe_interpolation = 1.0
        norm_eps = 1e-5
        patch_embed_kernel = None
        cfg_embed = False
        timestep_norm_scale_factor = 1.0
        rope_fhw_dim = None
        pack_latents = False
        camctrl_layers_num = None
        chunk_size = 10
        use_chunk_plucker_input = False

        # --- Base DiT config attributes (from Sana.__init__) ---
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.linear_head_dim = linear_head_dim
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.use_pe = use_pe
        self.pos_embed_type = pos_embed_type
        self.y_norm = y_norm
        # NOTE: ``self.config`` is provided (read-only) by ConfigMixin via @register_to_config.
        self.timestep_norm_scale_factor = timestep_norm_scale_factor

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cfg_embedder = None
        if cfg_embed:
            self.cfg_embedder = TimestepEmbedder(hidden_size)

        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)

        # --- Video camera-controlled DiT modules (from SanaMSVideoCamCtrl.__init__) ---
        self.chunk_size = chunk_size
        self.chunk_split_strategy = chunk_split_strategy
        self.patch_size = patch_size
        self.h = self.w = 0

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.pos_embed_ms = None
        self.pack_latents = pack_latents
        self.attn_type = attn_type

        self.camctrl_type = camctrl_type
        assert self.camctrl_type in [
            "BidirectionalGDNUCPESinglePathLiteLABothTriton",
            "BidirectionalSoftmaxUCPESinglePathLiteLA",
        ], f"Not supported camera control type: {self.camctrl_type}"

        self.camctrl_layers_num = camctrl_layers_num if camctrl_layers_num is not None else depth
        self.cam_attn_compress = cam_attn_compress

        kernel_size = patch_embed_kernel or patch_size
        x_embedder_in_channels = in_channels
        if self.pack_latents:
            x_embedder_in_channels = x_embedder_in_channels * 2 * 2
            self.out_channels = in_channels * 2 * 2

        self.x_embedder = PatchEmbedMS3D(
            patch_size, x_embedder_in_channels, hidden_size, kernel_size=kernel_size, bias=True
        )

        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        self.use_chunk_plucker_input = use_chunk_plucker_input
        self.use_chunk_plucker_post_attn = use_chunk_plucker_post_attn
        if self.use_chunk_plucker_input or self.use_chunk_plucker_post_attn:
            self.plucker_embedder = PatchEmbedMS3D(
                patch_size, chunk_plucker_channels, hidden_size, kernel_size=kernel_size, bias=True
            )
            nn.init.zeros_(self.plucker_embedder.proj.weight)
            nn.init.zeros_(self.plucker_embedder.proj.bias)

        # UCPE-style camera branch uses a 3-channel absmap (up_map + lat_map).
        self.raymap_embedder = PatchEmbedMS3D(patch_size, 3, hidden_size, kernel_size=kernel_size, bias=True)

        if attn_type in ["flash", "FlexLinearAttention", "flex"]:
            attention_head_dim = hidden_size // num_heads
        else:
            attention_head_dim = linear_head_dim

        if use_pe:
            if pos_embed_type != "wan_rope":
                raise ValueError(f'`pos_embed_type` must be "wan_rope", got {pos_embed_type!r}.')
            self.rope = WanRotaryPosEmbed(
                attention_head_dim=attention_head_dim, patch_size=patch_size, max_seq_len=1024, fhw_dim=rope_fhw_dim
            )
        self.softmax_every_n = softmax_every_n
        attn_type_list = [attn_type] * depth
        camctrl_type_list = [camctrl_type if i < self.camctrl_layers_num else None for i in range(depth)]
        if attn_type in ["flex", "FlexLinearAttention"]:
            attn_type_list[0] = "flash"
            attn_type_list[1] = "flash"

        if softmax_every_n > 0:
            attn_type_list, camctrl_type_list = _inject_softmax_layers(
                attn_type_list,
                camctrl_type_list,
                softmax_every_n,
            )
            logger.info(
                f"Hybrid attention (softmax_every_n={softmax_every_n}):\n"
                f"  attn_type_list = {attn_type_list}\n"
                f"  camctrl_type_list = {camctrl_type_list}"
            )

        self.blocks = nn.ModuleList(
            [
                SanaVideoMSCamCtrlBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    attn_type=attn_type_list[i],
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_norm=cross_norm,
                    t_kernel_size=t_kernel_size,
                    camctrl_type=camctrl_type_list[i],
                    patch_size=patch_size,
                    cam_attn_compress=self.cam_attn_compress,
                    chunk_size=chunk_size,
                    chunk_split_strategy=chunk_split_strategy,
                    conv_kernel_size=conv_kernel_size,
                    k_conv_only=k_conv_only,
                    use_chunk_plucker_post_attn=(
                        use_chunk_plucker_post_attn
                        and (chunk_plucker_post_attn_blocks < 0 or i < chunk_plucker_post_attn_blocks)
                    ),
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        if ffn_type == "GLUMBConvTemp":
            logger.info(f"{ffn_type} Temporal kernal: {t_kernel_size}")

        self.in_channels = self.out_channels = in_channels

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, frame):
        latents = latents.view(batch_size, num_channels_latents, frame, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 4, 6, 2, 3, 5)
        latents = latents.reshape(batch_size, num_channels_latents * 4, frame, height // 2, width // 2)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, frame):
        batch_size, channels, frame, H, W = latents.shape

        assert height % 2 == 0 and width % 2 == 0
        # latent height and width to be divisible by 2.
        latents = latents.view(batch_size, channels // 4, 2, 2, frame, height // 2, width // 2)
        latents = latents.permute(0, 1, 4, 5, 2, 6, 3)
        latents = latents.reshape(batch_size, channels // (2 * 2), frame, height, width)

        return latents

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        """Run the SANA-WM DiT.

        Args:
            hidden_states: ``(B, C, T, H, W)`` latents.
            timestep: ``(B, 1, T)`` per-frame diffusion timesteps (LTX style).
            encoder_hidden_states: ``(B, 1, L, D_caption)`` text embeddings.
            encoder_attention_mask: ``(B, L)`` text attention mask (diffusers convention).
            mask: Alias for ``encoder_attention_mask`` matching the sana DiT's
                kwarg name. If both are passed, ``mask`` takes precedence.
            return_dict: If ``True`` (default), returns a :class:`Transformer2DModelOutput`;
                otherwise returns a one-tuple ``(sample,)``.
            **kwargs: SANA-WM-specific conditioning — at minimum
                ``data_info``, ``camera_conditions``, ``chunk_plucker``.

        Returns:
            :class:`Transformer2DModelOutput` with ``sample`` of shape ``(B, C, T, H, W)``.
        """
        # The sana DiT names its text mask kwarg ``mask``.
        # Accept both ``mask=`` (sana convention) and ``encoder_attention_mask=``
        # (diffusers convention); the former wins if both are provided.
        if mask is None:
            mask = encoder_attention_mask
        x = hidden_states
        y = encoder_hidden_states

        bs = x.shape[0]
        x = x.to(self.dtype)
        if self.timestep_norm_scale_factor != 1.0:
            timestep = (timestep.float() / self.timestep_norm_scale_factor).to(torch.float32)
        else:
            timestep = timestep.long().to(torch.float32)
        y = y.to(self.dtype)
        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )

        data_info = kwargs.get("data_info", {})
        if data_info.get("image_vae_embeds", None) is not None:
            x = torch.cat([x, data_info["image_vae_embeds"].to(self.dtype)], dim=1)
        cam_embeds = kwargs.get("camera_conditions", None)
        if self.pack_latents:
            x = self._pack_latents(x, bs, self.in_channels, self.h, self.w, self.f)
            if cam_embeds is not None:
                cam_embeds = cam_embeds.to(self.dtype)

            self.h = self.h // 2
            self.w = self.w // 2

        if self.x_embedder.patch_size != self.x_embedder.kernel_size and self.x_embedder.kernel_size == (1, 2, 2):
            x = F.pad(x, (0, 1, 0, 1, 0, 0))
            if cam_embeds is not None:
                cam_embeds = F.pad(cam_embeds, (0, 1, 0, 1, 0, 0))

        x = self.x_embedder(x)
        if cam_embeds is not None:
            # Both surviving camctrl variants are UCPE-style: build raymats + 3-channel
            # absmap (up_map + lat_map) from the raw (B,F,20) camera conditions.
            raw_cam_conditions = cam_embeds
            cam_pos_embeds = kwargs.get("cam_pos_embeds", None)
            if cam_pos_embeds is not None and "absmap" in cam_pos_embeds:
                cam_embeds = cam_pos_embeds["absmap"]
                if "P" in cam_pos_embeds:
                    kwargs["raymats"] = cam_pos_embeds["P"]
            else:
                raymats, cam_embeds = _process_camera_conditions_ucpe(
                    raw_cam_conditions, bs, (self.f, self.h, self.w), self.patch_size
                )
                cam_embeds = cam_embeds.permute(0, 4, 1, 2, 3).to(self.dtype)
                kwargs["raymats"] = raymats
            if not (self.use_chunk_plucker_input or self.use_chunk_plucker_post_attn):
                cam_embeds = self.raymap_embedder(cam_embeds)
                x = x + cam_embeds
                kwargs["camera_embedding"] = cam_embeds
                kwargs["camera_conditions"] = raw_cam_conditions

        if self.use_chunk_plucker_input and "chunk_plucker" in kwargs:
            plucker_input = kwargs["chunk_plucker"].to(self.dtype)
            plucker_emb = self.plucker_embedder(plucker_input)
            x = x + plucker_emb

        if self.use_chunk_plucker_post_attn and "chunk_plucker" in kwargs:
            plucker_input = kwargs["chunk_plucker"].to(self.dtype)
            kwargs["plucker_emb"] = self.plucker_embedder(plucker_input)

        image_pos_embed = kwargs.get("pos_embeds", None)
        if self.use_pe and image_pos_embed is None:
            image_pos_embed = self.rope((self.f, self.h, self.w))
        elif image_pos_embed is not None:
            image_pos_embed = image_pos_embed.to(x.device)
            while image_pos_embed.ndim > 4:
                image_pos_embed = image_pos_embed.squeeze(1)

        t = self.t_embedder(timestep.flatten())  # (N, D)
        t0 = self.t_block(t)
        t = t.unflatten(dim=0, sizes=timestep.shape)
        t0 = t0.unflatten(dim=0, sizes=timestep.shape)

        y = self.y_embedder(y)  # (N, D)
        if self.y_norm:
            y = self.attention_y_norm(y)

        if mask is None:
            raise ValueError(
                "`mask` is required: SANA-WM's cross-attention needs the text padding mask to build its attention "
                "bias. Pass the prompt attention mask returned by the pipeline's `encode_prompt`."
            )
        mask = mask.to(torch.int16)
        mask = mask.repeat(y.shape[0] // mask.shape[0], 1) if mask.shape[0] != y.shape[0] else mask
        mask = mask.squeeze(1).squeeze(1)
        y_lens = mask

        block_mask = None

        if kwargs.get("camera_conditions") is not None:
            # Pre-compute the UCPE ray matrices once and share them across blocks
            # (both surviving camctrl variants are UCPE-style).
            if self.attn_type in ["flash", "FlexLinearAttention", "flex"]:
                head_dim = self.hidden_size // self.num_heads
            else:
                head_dim = self.linear_head_dim

            cam_pos_embeds = kwargs.get("cam_pos_embeds", None)
            if cam_pos_embeds is not None:
                for k, v in cam_pos_embeds.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(x.device)
                        if k == "absmap":
                            while v.ndim > 5:
                                v = v.squeeze(1)
                        else:
                            while v.ndim > 4:
                                v = v.squeeze(1)
                        cam_pos_embeds[k] = v

            kwargs["ucpe_ray_transforms"] = _prepare_ucpe_ray_transforms(
                head_dim=head_dim,
                camera_conditions=kwargs["camera_conditions"],
                HW=(self.f, self.h, self.w),
                patch_size=self.patch_size,
                rotary_emb=image_pos_embed,
                raymats=kwargs.get("raymats"),
                cam_pos_embeds=cam_pos_embeds,
            )

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                y,
                t0,
                y_lens,
                (self.f, self.h, self.w),
                image_pos_embed,
                block_mask=block_mask if i > 1 else None,
                **kwargs,
            )  # (N, T, D)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if self.pack_latents:
            x = self._unpack_latents(x, self.h * 2, self.w * 2, self.f)

        return Transformer2DModelOutput(sample=x) if return_dict else (x,)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C) imgs: (N, H, W, C)
        """
        c = self.out_channels
        p_f, p_h, p_w = self.x_embedder.patch_size
        h, w = self.h, self.w
        assert self.f * self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.f, h, w, p_f, p_h, p_w, c))
        x = torch.einsum("nfhwopqc->ncfohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, self.f * p_f, h * p_h, w * p_w))

        return imgs
