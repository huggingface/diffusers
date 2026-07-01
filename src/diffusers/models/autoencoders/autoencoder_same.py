# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
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

"""
SAME (Semantically-Aligned Music Encoder) autoencoder for Stable Audio 3.

Architecture (paper §2.1):
  encode: patch → TRB stack → linear projection → soft-norm bottleneck decode: soft-norm inverse → linear projection →
  reverse-TRB stack → unpatch

Total downsampling ratio = patch_size × ∏(strides). Default production ratio: 256 × 16 = 4096 × → 256-dim latents at
~10.76 Hz for 44.1 kHz stereo.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AutoencoderSAMEOutput(BaseOutput):
    """Output of :meth:`AutoencoderSAME.encode`."""

    latents: torch.Tensor


@dataclass
class AutoencoderSAMEDecoderOutput(BaseOutput):
    """Output of :meth:`AutoencoderSAME.decode`."""

    sample: torch.Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────


def _wn_conv1d(in_ch: int, out_ch: int, kernel: int = 1, **kw) -> nn.Conv1d:
    return weight_norm(nn.Conv1d(in_ch, out_ch, kernel, **kw))


def _pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1) -> torch.Tensor:
    """Right-pad *x* along *dim* with zeros so its size is divisible by *multiple*."""
    size = x.shape[dim]
    pad = (multiple - size % multiple) % multiple
    if pad == 0:
        return x
    shape = list(x.shape)
    shape[dim] = pad
    return torch.cat([x, x.new_zeros(shape)], dim=dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (2, -1))
    x1, x2 = x.unbind(-2)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Partial RoPE — rotate the first *rot_dim* channels, leave the rest."""
    rot_dim = freqs.shape[-1]
    out_dtype = t.dtype
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot.float()
    freqs = freqs[-t_rot.shape[-2] :].float()
    t_rot = (t_rot * freqs.cos()) + (_rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_rot.to(out_dtype), t_pass), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Normalization layers
# ──────────────────────────────────────────────────────────────────────────────


class _DynamicTanh(nn.Module):
    """Dynamic-Tanh (DyT) normalisation used in production SAME configs."""

    def __init__(self, dim: int, init_alpha: float = 4.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


# ──────────────────────────────────────────────────────────────────────────────
# Rotary embeddings
# ──────────────────────────────────────────────────────────────────────────────


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)  # (T, rot_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Feed-forward (SwiGLU)
# ──────────────────────────────────────────────────────────────────────────────


class _FeedForward(nn.Module):
    """
    GLU FFN with zero-initialised output projection.

    The gate activation is SiLU (``sinusoidal=False``, "SwiGLU") or ``sin(pi * gate)`` (``sinusoidal=True``). SAME-L uses
    sinusoidal activations in the trailing decoder transformer layers.
    """

    def __init__(self, dim: int, mult: int = 3, sinusoidal: bool = False):
        super().__init__()
        inner = int(dim * mult)
        self.sinusoidal = sinusoidal
        self.proj_in = nn.Linear(dim, inner * 2, bias=True)
        self.proj_out = nn.Linear(inner, dim, bias=True)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj_in(x).chunk(2, dim=-1)
        gate = torch.sin(math.pi * gate) if self.sinusoidal else F.silu(gate)
        return self.proj_out(x * gate)


# ──────────────────────────────────────────────────────────────────────────────
# Self-attention (standard or differential, with QK-DyT norm and partial RoPE)
# ──────────────────────────────────────────────────────────────────────────────


class _Attention(nn.Module):
    """
    Self-attention used inside TRB transformer blocks.

    When *use_differential* is True (default for SAME-L), the block runs two independent attention maps — Attn(Q1,K1,V)
    − Attn(Q2,K2,V) — so that attention patterns common to both heads cancel out, improving focus.
    """

    def __init__(self, dim: int, dim_heads: int = 128, use_differential: bool = True, qk_norm_eps: float = 1e-3):
        super().__init__()
        self.dim_heads = dim_heads
        self.num_heads = dim // dim_heads
        self.use_differential = use_differential

        n_proj = 5 if use_differential else 3
        self.to_qkv = nn.Linear(dim, dim * n_proj, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)

        self.q_norm = _DynamicTanh(dim_heads)
        self.k_norm = _DynamicTanh(dim_heads)
        self.rope = _RotaryEmbedding(dim_heads // 2)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, self.dim_heads)).transpose(1, 2)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        freqs = self.rope(N, x.device)

        if self.use_differential:
            q1, q2, k1, k2, v = self.to_qkv(x).chunk(5, dim=-1)
            q1, q2, k1, k2, v = map(self._heads, (q1, q2, k1, k2, v))
            q1, q2 = self.q_norm(q1), self.q_norm(q2)
            k1, k2 = self.k_norm(k1), self.k_norm(k2)
            q1, q2 = _apply_rotary(q1, freqs), _apply_rotary(q2, freqs)
            k1, k2 = _apply_rotary(k1, freqs), _apply_rotary(k2, freqs)
            out = F.scaled_dot_product_attention(q1, k1, v, attn_mask=attn_mask) - F.scaled_dot_product_attention(
                q2, k2, v, attn_mask=attn_mask
            )
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(self._heads, (q, k, v))
            q, k = self.q_norm(q), self.k_norm(k)
            q, k = _apply_rotary(q, freqs), _apply_rotary(k, freqs)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        return self.to_out(out.transpose(1, 2).flatten(-2))


# ──────────────────────────────────────────────────────────────────────────────
# Transformer block (pre-norm, zero-init branches)
# ──────────────────────────────────────────────────────────────────────────────


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 128,
        use_differential: bool = True,
        ff_mult: int = 3,
        sinusoidal: bool = False,
    ):
        super().__init__()
        self.norm_attn = _DynamicTanh(dim)
        self.attn = _Attention(dim, dim_heads=dim_heads, use_differential=use_differential)
        self.norm_ff = _DynamicTanh(dim)
        self.ff = _FeedForward(dim, mult=ff_mult, sinusoidal=sinusoidal)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm_ff(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Resampling Block (TRB)
# ──────────────────────────────────────────────────────────────────────────────


def _band_mask(seq_len: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Additive sliding-window attention mask of shape ``(seq_len, seq_len)``.

    Position ``i`` attends to position ``j`` iff ``-window <= (j - i) <= window`` (``0`` inside the band, ``-inf``
    outside).
    """
    idx = torch.arange(seq_len, device=device)
    delta = idx[None, :] - idx[:, None]
    keep = (delta >= -window) & (delta <= window)
    return torch.zeros((seq_len, seq_len), dtype=dtype, device=device).masked_fill(~keep, float("-inf"))


class SAMETransformerResamplingBlock(nn.Module):
    """
    Core building block of SAME.

    **Encoder mode** (stride S):
      Groups S consecutive input frames into one segment, appends a single learnable output embedding, then runs D
      transformer layers over the full flattened segment sequence and keeps only the output embedding → downsample by S.

    **Decoder mode** (stride S):
      Groups 1 input frame with S learnable output embeddings, runs D transformer layers over the full flattened
      sequence, then keeps the S output embeddings → upsample by S.

    Attention uses an overlapping *sliding-window* band mask over the flattened segment sequence: each token attends to
    ``sliding_window * (stride + 1)`` neighbours on each side. RoPE is computed over the full sequence length. This
    matches the reference implementation exactly (a single non-overlapping chunk would only match for one segment).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Down-/up-sampling factor.
        mode: ``"encoder"`` or ``"decoder"``.
        transformer_depth: Number of :class:`_TransformerBlock` layers.
        dim_heads: Attention head dimension.
        use_differential: Whether to use differential attention.
        chunk_size: Kept for config/back-compat; no longer used by the band-mask attention.
        ff_mult: Feed-forward expansion factor.
        sliding_window: Sliding-window half-width in latents (band half-width is ``sliding_window * (stride + 1)``).
        sinusoidal_blocks: Number of trailing transformer layers that use ``sin`` FFN gating instead of SiLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        mode: str = "encoder",
        transformer_depth: int = 3,
        dim_heads: int = 128,
        use_differential: bool = True,
        chunk_size: int = 128,
        ff_mult: int = 3,
        sliding_window: int = 1,
        sinusoidal_blocks: int = 0,
    ):
        super().__init__()
        if mode not in ("encoder", "decoder"):
            raise ValueError(f"mode must be 'encoder' or 'decoder', got {mode}")

        self.stride = stride
        self.chunk_size = chunk_size
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sliding_window = sliding_window

        # Transformer operates at out_channels (enc) or in_channels (dec)
        tdim = out_channels if mode == "encoder" else in_channels
        self.mapping = (
            _wn_conv1d(in_channels, out_channels, 1, padding=0) if in_channels != out_channels else nn.Identity()
        )
        self.transformers = nn.ModuleList(
            [
                _TransformerBlock(
                    tdim,
                    dim_heads=min(dim_heads, tdim),
                    use_differential=use_differential,
                    ff_mult=ff_mult,
                    # The trailing `sinusoidal_blocks` layers use sin gating (matches the reference).
                    sinusoidal=(transformer_depth - i) < sinusoidal_blocks,
                )
                for i in range(transformer_depth)
            ]
        )

        # Learnable "output" embeddings per segment. With variable stride a single token is shared across all
        # stride positions (broadcast), so both encoder and decoder store a single token:
        #   encoder → 1 new token of size out_channels
        #   decoder → 1 new token of size in_channels (broadcast to `stride` positions at runtime)
        if mode == "encoder":
            self.new_tokens = nn.Parameter(1e-5 * torch.randn(1, 1, out_channels))
        else:
            self.new_tokens = nn.Parameter(1e-5 * torch.randn(1, 1, in_channels))

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        S = self.stride

        # 1. Pad T to a multiple of the stride so every segment holds exactly S input frames.
        x = _pad_to_multiple(x, S, dim=-1)
        T_pad = x.shape[-1]

        # 2. Channel mapping (in_channels → out_channels) before transformer.
        x = self.mapping(x)  # (B, C_out, T_pad)

        # 3. Convert to sequence and segment into groups of S.
        x = x.transpose(1, 2)  # (B, T_pad, C_out)
        N = T_pad // S  # number of output latents
        x = x.reshape(B * N, S, self.out_channels)  # (B*N, S, C_out)

        # 4. Append one learnable output token per segment → (B*N, S+1, C_out).
        new = self.new_tokens.expand(B * N, 1, -1)
        x = torch.cat([x, new], dim=1)

        # 5. Flatten to the full segment sequence and run the band-mask transformer.
        sub = S + 1
        x = x.reshape(B, N * sub, self.out_channels)  # (B, N*(S+1), C_out)
        mask = _band_mask(x.shape[1], self.sliding_window * sub, x.device, x.dtype)
        for layer in self.transformers:
            x = layer(x, attn_mask=mask)

        # 6. Extract the last (output) token from each segment.
        x = x.reshape(B * N, sub, self.out_channels)[:, -1:, :]  # (B*N, 1, C_out)
        x = x.reshape(B, N, self.out_channels).transpose(1, 2)  # (B, C_out, N)

        # 7. Crop away padding-derived latents.
        n_latents = -(-T // S)  # ceil(T / S)
        return x[..., :n_latents]

    # ------------------------------------------------------------------
    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        S = self.stride

        # 1. Each input latent seeds one segment: (B*T, 1, C_in).
        x = x.transpose(1, 2).reshape(B * T, 1, self.in_channels)

        # 2. Append S learnable output tokens (broadcast from a single token) → (B*T, 1+S, C_in).
        new = self.new_tokens.expand(B * T, S, -1)
        x = torch.cat([x, new], dim=1)

        # 3. Flatten to the full segment sequence and run the band-mask transformer.
        sub = 1 + S
        x = x.reshape(B, T * sub, self.in_channels)
        mask = _band_mask(x.shape[1], self.sliding_window * sub, x.device, x.dtype)
        for layer in self.transformers:
            x = layer(x, attn_mask=mask)

        # 4. Extract the last S (output) tokens from each segment.
        x = x.reshape(B * T, sub, self.in_channels)[:, -S:, :]  # (B*T, S, C_in)
        x = x.reshape(B, T * S, self.in_channels).transpose(1, 2)  # (B, C_in, T*S)

        # 5. Channel mapping (in_channels → out_channels) after transformer.
        return self.mapping(x)  # (B, C_out, T*S)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x) if self.mode == "encoder" else self._decode(x)


# ──────────────────────────────────────────────────────────────────────────────
# Patch embedding / unpatching
# ──────────────────────────────────────────────────────────────────────────────


class _PatchEmbed(nn.Module):
    """
    Groups consecutive audio samples into non-overlapping patches, trading the time dimension for extra channels (256×
    or similar downsampling with zero learnable parameters).

    encode: (B, C, T) → (B, C·P, T//P) P = patch_size decode: (B, C·P, T//P) → (B, C, T)
    """

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        P = self.patch_size
        x = _pad_to_multiple(x, P, dim=-1)
        B, C, T = x.shape
        return x.reshape(B, C, T // P, P).permute(0, 1, 3, 2).reshape(B, C * P, T // P)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        P = self.patch_size
        B, CP, T = x.shape
        C = CP // P
        return x.reshape(B, C, P, T).permute(0, 1, 3, 2).reshape(B, C, T * P)


# ──────────────────────────────────────────────────────────────────────────────
# Soft-norm bottleneck
# ──────────────────────────────────────────────────────────────────────────────


class _SoftNormBottleneck(nn.Module):
    """
    Learnable affine normalisation of latents with running-std tracking.

    encode: z = (x · scale + bias) / running_std decode: x = z · running_std (inference-time inverse)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))
        self.register_buffer("running_std", torch.ones(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.bias
        if self.training:
            self.running_std.data = (self.running_std.data * 0.999 + x.std().detach() * 0.001).clamp(min=1e-4)
        return x / self.running_std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.running_std


# ──────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder stacks
# ──────────────────────────────────────────────────────────────────────────────


class SAMEEncoder(nn.Module):
    """
    Stack of :class:`SAMETransformerResamplingBlock` (encoder mode) followed by a linear projection to *latent_dim*.

    Input shape after patch embedding: ``(B, audio_channels × patch_size, T // patch_size)`` Output shape: ``(B,
    latent_dim, T // patch_size // ∏strides)``
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        c_mults: List[int],
        strides: List[int],
        transformer_depths: List[int],
        latent_dim: int,
        dim_heads: int = 128,
        use_differential: bool = True,
        chunk_size: int = 128,
        ff_mult: int = 3,
        sliding_window: int = 1,
        sinusoidal_blocks: List[int] = (0,),
    ):
        super().__init__()
        ch = [in_channels] + [c * channels for c in c_mults]
        self.blocks = nn.ModuleList(
            [
                SAMETransformerResamplingBlock(
                    in_channels=ch[i],
                    out_channels=ch[i + 1],
                    stride=strides[i],
                    mode="encoder",
                    transformer_depth=transformer_depths[i],
                    dim_heads=dim_heads,
                    use_differential=use_differential,
                    chunk_size=chunk_size,
                    ff_mult=ff_mult,
                    sliding_window=sliding_window,
                    sinusoidal_blocks=sinusoidal_blocks[i],
                )
                for i in range(len(strides))
            ]
        )
        self.proj = nn.Linear(ch[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        # (B, C_final, T_latent) → (B, latent_dim, T_latent)
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class SAMEDecoder(nn.Module):
    """
    Linear projection from *latent_dim* followed by a stack of :class:`SAMETransformerResamplingBlock` (decoder mode).

    Input shape: ``(B, latent_dim, T_latent)`` Output shape: ``(B, audio_channels × patch_size, T // patch_size)``
    """

    def __init__(
        self,
        out_channels: int,
        channels: int,
        c_mults: List[int],
        strides: List[int],
        transformer_depths: List[int],
        latent_dim: int,
        dim_heads: int = 128,
        use_differential: bool = True,
        chunk_size: int = 128,
        ff_mult: int = 3,
        sliding_window: int = 1,
        sinusoidal_blocks: List[int] = (0,),
    ):
        super().__init__()
        ch = [out_channels] + [c * channels for c in c_mults]
        self.proj = nn.Linear(latent_dim, ch[-1])
        self.blocks = nn.ModuleList(
            [
                SAMETransformerResamplingBlock(
                    in_channels=ch[i + 1],
                    out_channels=ch[i],
                    stride=strides[i],
                    mode="decoder",
                    transformer_depth=transformer_depths[i],
                    dim_heads=dim_heads,
                    use_differential=use_differential,
                    chunk_size=chunk_size,
                    ff_mult=ff_mult,
                    sliding_window=sliding_window,
                    sinusoidal_blocks=sinusoidal_blocks[i],
                )
                for i in range(len(strides) - 1, -1, -1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, latent_dim, T_latent) → (B, C_final, T_latent)
        x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Top-level AutoencoderSAME
# ──────────────────────────────────────────────────────────────────────────────


class AutoencoderSAME(ModelMixin, ConfigMixin):
    r"""
    Semantically-Aligned Music Encoder (SAME) autoencoder from *Stable Audio 3* (`arXiv 2605.17991
    <https://arxiv.org/abs/2605.17991>`_).

    The model consists of:

    * **Patch embedding** — reshapes stereo audio into non-overlapping patches, trading time for channels (``patch_size
      ×`` downsample, no learned params).
    * **Encoder TRB stack** — :class:`SAMETransformerResamplingBlock` blocks that further downsample by a factor of
      ``∏(encoder_strides)``.
    * **Soft-norm bottleneck** — learnable affine normalisation with running std.
    * **Decoder TRB stack** — mirrors the encoder in reverse.
    * **Unpatch** — reshapes channels back into the time dimension.

    Total downsampling ratio: ``patch_size × ∏(encoder_strides)``.

    The default hyperparameters match the **SAME-S** checkpoint (``stabilityai/SAME-S``). To load **SAME-L**
    (``stabilityai/SAME-L``, used by SA3 Medium) pass ``encoder_channels=256, encoder_transformer_depths=(12,)``.

    .. code-block:: python

        # SAME-S (108 M params, used by SA3 small models) model = AutoencoderSAME() # default values

        # SAME-L (852 M params, used by SA3 Medium) model = AutoencoderSAME(encoder_channels=256,
        encoder_transformer_depths=(12,))

    Parameters:
        audio_channels: Number of audio channels (2 for stereo).
        patch_size: Non-overlapping patch size applied before the TRB encoder
            (and reversed after the TRB decoder). Contributes ``patch_size ×`` to the total downsampling ratio.
            Production value: 256.
        encoder_channels: Base channel count for the TRB. 128 for SAME-S,
            256 for SAME-L.
        encoder_c_mults: Channel multiplier for each TRB level (one entry per
            TRB). Both SAME-S and SAME-L use ``(6,)`` — a single TRB whose hidden dimension is ``encoder_channels ×
            6``.
        encoder_strides: Down-/up-sampling stride for each TRB level. Both
            SAME-S and SAME-L use ``(16,)`` — one TRB with stride 16.
        encoder_transformer_depths: Transformer layers per TRB level.
            6 for SAME-S, 12 for SAME-L.
        latent_dim: Dimensionality of the latent space. 256 for both variants.
        use_differential_attention: If ``True``, use differential attention
            inside each TRB transformer block (default on for SAME-S/L).
        dim_heads: Attention head dimension. 64 for production SAME-S/L.
        encoder_chunk_size: Kept for config/back-compat; no longer governs
            attention (the sliding-window band mask spans the full sequence).
        ff_mult: SwiGLU feed-forward expansion factor.
        sliding_window: Sliding-window half-width (in latents) for the band-mask
            attention. Production SAME-S/L use 1.
        encoder_sinusoidal_blocks: Per-TRB count of trailing transformer layers
            that use ``sin`` FFN gating in the encoder (SAME-L: ``(0,)``).
        decoder_sinusoidal_blocks: Per-TRB count of trailing transformer layers
            that use ``sin`` FFN gating in the decoder (SAME-L: ``(8,)``).
        sampling_rate: Audio sample rate in Hz (e.g. 44100).
    """

    @register_to_config
    def __init__(
        self,
        audio_channels: int = 2,
        patch_size: int = 256,
        encoder_channels: int = 128,
        encoder_c_mults: List[int] = (6,),
        encoder_strides: List[int] = (16,),
        encoder_transformer_depths: List[int] = (6,),
        latent_dim: int = 256,
        use_differential_attention: bool = True,
        dim_heads: int = 64,
        encoder_chunk_size: int = 32,
        ff_mult: int = 3,
        sliding_window: int = 1,
        encoder_sinusoidal_blocks: List[int] = (0,),
        decoder_sinusoidal_blocks: List[int] = (0,),
        sampling_rate: int = 44100,
    ):
        super().__init__()

        # Derived constants — set as attributes so they're accessible without
        # touching the (possibly absent) sub-modules.
        self.sampling_rate = sampling_rate
        self.latent_dim = latent_dim
        self.downsampling_ratio = patch_size * math.prod(encoder_strides)

        patched_in = audio_channels * patch_size

        self.patch_embed = _PatchEmbed(patch_size)
        self.encoder = SAMEEncoder(
            in_channels=patched_in,
            channels=encoder_channels,
            c_mults=list(encoder_c_mults),
            strides=list(encoder_strides),
            transformer_depths=list(encoder_transformer_depths),
            latent_dim=latent_dim,
            dim_heads=dim_heads,
            use_differential=use_differential_attention,
            chunk_size=encoder_chunk_size,
            ff_mult=ff_mult,
            sliding_window=sliding_window,
            sinusoidal_blocks=list(encoder_sinusoidal_blocks),
        )
        self.bottleneck = _SoftNormBottleneck(latent_dim)
        self.decoder = SAMEDecoder(
            out_channels=patched_in,
            channels=encoder_channels,
            c_mults=list(encoder_c_mults),
            strides=list(encoder_strides),
            transformer_depths=list(encoder_transformer_depths),
            latent_dim=latent_dim,
            dim_heads=dim_heads,
            use_differential=use_differential_attention,
            chunk_size=encoder_chunk_size,
            ff_mult=ff_mult,
            sliding_window=sliding_window,
            sinusoidal_blocks=list(decoder_sinusoidal_blocks),
        )

    # ------------------------------------------------------------------
    @apply_forward_hook
    def encode(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[AutoencoderSAMEOutput, torch.Tensor]:
        """
        Encode stereo audio to latents.

        Args:
            sample: ``(B, audio_channels, T)`` waveform tensor.
            return_dict: If ``True`` return an :class:`AutoencoderSAMEOutput`.

        Returns:
            Latent tensor of shape ``(B, latent_dim, T // downsampling_ratio)`` (wrapped in
            :class:`AutoencoderSAMEOutput` when *return_dict* is ``True``).
        """
        x = self.patch_embed.encode(sample)
        x = self.encoder(x)
        x = self.bottleneck.encode(x)
        if not return_dict:
            return (x,)
        return AutoencoderSAMEOutput(latents=x)

    # ------------------------------------------------------------------
    @apply_forward_hook
    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[AutoencoderSAMEDecoderOutput, torch.Tensor]:
        """
        Decode latents back to stereo audio.

        Args:
            latents: ``(B, latent_dim, T_latent)`` latent tensor.
            return_dict: If ``True`` return an :class:`AutoencoderSAMEDecoderOutput`.

        Returns:
            Waveform tensor of shape ``(B, audio_channels, T_latent × downsampling_ratio)`` (wrapped in
            :class:`AutoencoderSAMEDecoderOutput` when *return_dict* is ``True``).
        """
        x = self.bottleneck.decode(latents)
        x = self.decoder(x)
        x = self.patch_embed.decode(x)
        if not return_dict:
            return (x,)
        return AutoencoderSAMEDecoderOutput(sample=x)

    # ------------------------------------------------------------------
    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[AutoencoderSAMEDecoderOutput, torch.Tensor]:
        """
        Encode and immediately decode *sample* (reconstruction).

        Args:
            sample: ``(B, audio_channels, T)`` waveform tensor.
            return_dict: If ``True`` return an :class:`AutoencoderSAMEDecoderOutput`.

        Returns:
            Reconstructed waveform (same shape as *sample*, possibly longer due to padding), wrapped in
            :class:`AutoencoderSAMEDecoderOutput` when *return_dict* is ``True``.
        """
        latents = self.encode(sample).latents
        return self.decode(latents, return_dict=return_dict)

    # ------------------------------------------------------------------
    def encode_audio(
        self,
        audio: torch.Tensor,
        chunked: bool = False,
        chunk_size: int = 128,
        overlap: int = 32,
    ) -> torch.Tensor:
        """
        Convenience wrapper that optionally encodes in overlapping latent chunks to limit peak memory usage for long
        audio sequences.

        Args:
            audio: Preprocessed ``(B, audio_channels, T)`` waveform.
            chunked: Whether to split long audio into overlapping chunks.
            chunk_size: Chunk size in latent frames (only used when *chunked*).
            overlap: Overlap in latent frames between adjacent chunks.

        Returns:
            Latent tensor ``(B, latent_dim, T_latent)``.
        """
        if not chunked or audio.shape[-1] < chunk_size * self.downsampling_ratio:
            return self.encode(audio).latents

        spl = self.downsampling_ratio
        hop_samples = (chunk_size - overlap) * spl
        total = audio.shape[-1]

        starts = list(range(0, total - chunk_size * spl + 1, hop_samples))
        if starts[-1] != total - chunk_size * spl:
            starts.append(total - chunk_size * spl)

        chunks = [self.encode(audio[..., s : s + chunk_size * spl]).latents for s in starts]

        total_lat = total // spl
        half_ov = overlap // 2
        out = audio.new_zeros(*chunks[0].shape[:-1], total_lat)
        for i, (s, ch) in enumerate(zip(starts, chunks)):
            is_first, is_last = i == 0, i == len(starts) - 1
            lat_start = (total_lat - chunk_size) if is_last else s // spl
            lo = 0 if is_first else half_ov
            hi = chunk_size if is_last else chunk_size - half_ov
            out[..., lat_start + lo : lat_start + hi] = ch[..., lo:hi]
        return out

    # ------------------------------------------------------------------
    def decode_audio(
        self,
        latents: torch.Tensor,
        chunked: bool = False,
        chunk_size: int = 128,
        overlap: int = 32,
    ) -> torch.Tensor:
        """
        Convenience wrapper that optionally decodes in overlapping latent chunks.

        Args:
            latents: ``(B, latent_dim, T_latent)`` latent tensor.
            chunked: Whether to split into overlapping chunks.
            chunk_size: Chunk size in latent frames.
            overlap: Overlap in latent frames between adjacent chunks.

        Returns:
            Waveform tensor ``(B, audio_channels, T)``.
        """
        if not chunked or latents.shape[-1] < chunk_size:
            return self.decode(latents).sample

        spl = self.downsampling_ratio
        hop = chunk_size - overlap
        total_lat = latents.shape[-1]

        starts = list(range(0, total_lat - chunk_size + 1, hop))
        if starts[-1] != total_lat - chunk_size:
            starts.append(total_lat - chunk_size)

        chunks = [self.decode(latents[..., s : s + chunk_size]).sample for s in starts]

        total_samples = total_lat * spl
        cs_samples = chunk_size * spl
        half_ov_s = (overlap // 2) * spl
        out = latents.new_zeros(*chunks[0].shape[:-1], total_samples)
        for i, (s, ch) in enumerate(zip(starts, chunks)):
            is_first, is_last = i == 0, i == len(starts) - 1
            samp_start = (total_samples - cs_samples) if is_last else s * spl
            lo = 0 if is_first else half_ov_s
            hi = cs_samples if is_last else cs_samples - half_ov_s
            out[..., samp_start + lo : samp_start + hi] = ch[..., lo:hi]
        return out
