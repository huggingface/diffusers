# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_natten_available, logging
from ...utils.accelerate_utils import apply_forward_hook
from ...utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .autoencoder_kl_ltx2 import LTX2VideoEncoder3d
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth on H/W only: `(B, C, F, H, W)` -> `(B, C * patch_size**2, F, H // p, W // p)`.

    The channel packing order is `(channel, width_offset, height_offset)`, matching the reference implementation's `b c
    (f p) (h q) (w r) -> b (c p r q) f h w` with `p = 1`.
    """
    batch_size, num_channels, num_frames, height, width = x.shape
    x = x.reshape(
        batch_size, num_channels, num_frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(
        batch_size, num_channels * patch_size * patch_size, num_frames, height // patch_size, width // patch_size
    )


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Depth-to-space on H/W only, the exact inverse of [`_patchify`]."""
    batch_size, num_channels, num_frames, height, width = x.shape
    num_channels = num_channels // (patch_size * patch_size)
    x = x.reshape(batch_size, num_channels, patch_size, patch_size, num_frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(batch_size, num_channels, num_frames, height * patch_size, width * patch_size)


def _neighborhood_block_mask(
    num_frames: int, height: int, width: int, kernel_size: tuple[int, int, int], device: torch.device
):
    """Build a FlexAttention `BlockMask` for 3D neighborhood attention.

    Each query attends to a `kernel_size` window that is centered where possible and *shifted inward* at the grid
    boundaries so it always holds exactly `kernel_size` positions. That inward shift (rather than truncating the
    window) is what NATTEN's `na3d` does, which is why the flex path can stand in for it.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    kernel_t, kernel_h, kernel_w = kernel_size
    kernel_t, kernel_h, kernel_w = min(kernel_t, num_frames), min(kernel_h, height), min(kernel_w, width)
    hw = height * width

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_t, q_rem = q_idx // hw, q_idx % hw
        q_h, q_w = q_rem // width, q_rem % width
        k_t, k_rem = kv_idx // hw, kv_idx % hw
        k_h, k_w = k_rem // width, k_rem % width

        start_t = torch.clamp(q_t - kernel_t // 2, 0, num_frames - kernel_t)
        start_h = torch.clamp(q_h - kernel_h // 2, 0, height - kernel_h)
        start_w = torch.clamp(q_w - kernel_w // 2, 0, width - kernel_w)
        window_t = (k_t >= start_t) & (k_t < start_t + kernel_t)
        window_h = (k_h >= start_h) & (k_h < start_h + kernel_h)
        window_w = (k_w >= start_w) & (k_w < start_w + kernel_w)
        return window_t & window_h & window_w

    seq_len = num_frames * hw
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


class LTX2VideoVaeRotaryPosEmbed3D(nn.Module):
    """Absolute 3D rotary embedding for the diffusion decoder's neighborhood attention.

    `head_dim` is split into (T, H, W) chunks, each rotated by its own axis position. Positions are the tensor's own
    0-based indices: attention here is always a local window with no causal masking, so the score between a query and a
    key depends only on their relative offset and a shared origin shift is a no-op. Rotation is computed in fp32 and
    cast back to the input dtype.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be a multiple of 8, got {head_dim}.")
        # Split `head_dim` across the (T, H, W) chunks the way the reference decoder does: a quarter to T, the
        # rest halved between H and W, with both halves kept even so each holds whole rotation pairs.
        dim_t = (head_dim // 4) // 2 * 2
        dim_hw = (head_dim - dim_t) // 2
        if dim_hw % 2 != 0:
            dim_t -= 2
            dim_hw = (head_dim - dim_t) // 2
        self.rope_dim_split = (dim_t, dim_hw, dim_hw)
        self.base = base

    def _inv_freqs(self, dim: int, device: torch.device) -> torch.Tensor:
        # The reference builds these in float64 and casts once. float64 is unsupported on some backends, so ask
        # for the device's widest available dtype instead: the difference is at most 1.5e-08 in the frequencies
        # and 1e-06 in the resulting angles, four orders of magnitude under bf16 resolution.
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, device)
        exponents = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
        return (1.0 / self.base**exponents).to(torch.float32)

    def _rotate_axis(self, x: torch.Tensor, positions: torch.Tensor, inv_freqs: torch.Tensor, axis: int):
        out_dtype = x.dtype
        pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        # Broadcast the angle over (B, T, H, W, heads, dim // 2), varying only along `axis`.
        shape = [1, 1, 1, 1, 1, inv_freqs.shape[0]]
        shape[axis] = positions.shape[0]
        angles = (positions[:, None] * inv_freqs[None, :]).reshape(shape)
        cos, sin = angles.cos(), angles.sin()
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.reshape(x.shape).to(out_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`hidden_states`: `(B, T, H, W, heads, head_dim)`."""
        dim_t, dim_h, _ = self.rope_dim_split
        num_frames, height, width = hidden_states.shape[1:4]
        device = hidden_states.device
        inv_t, inv_h, inv_w = (self._inv_freqs(dim, device) for dim in self.rope_dim_split)

        positions_t = torch.arange(num_frames, dtype=torch.float32, device=device)
        positions_h = torch.arange(height, dtype=torch.float32, device=device)
        positions_w = torch.arange(width, dtype=torch.float32, device=device)
        rotated_t = self._rotate_axis(hidden_states[..., :dim_t], positions_t, inv_t, axis=1)
        rotated_h = self._rotate_axis(hidden_states[..., dim_t : dim_t + dim_h], positions_h, inv_h, axis=2)
        rotated_w = self._rotate_axis(hidden_states[..., dim_t + dim_h :], positions_w, inv_w, axis=3)
        return torch.cat([rotated_t, rotated_h, rotated_w], dim=-1)


class LTX2VideoVaeNeighborhoodAttnProcessor:
    """Portable neighborhood-attention processor built on FlexAttention via `dispatch_attention_fn`.

    Runs anywhere the flex attention path runs. For bit-exact parity with the reference decoder, use
    [`LTX2VideoVaeNeighborhoodNattenProcessor`] instead.

    Requires the `flex` attention backend: the neighborhood window is expressed as a
    [`~torch.nn.attention.flex_attention.BlockMask`], which only the flex backend consumes. Switching the backend —
    e.g. via `model.set_attention_backend("flash")` — raises a `ValueError` rather than handing the mask to a backend
    that cannot read it.
    """

    _attention_backend = "flex"
    _parallel_config = None

    _SUPPORTED_BACKENDS = ("flex", "_native_flex")

    def __call__(self, attn: "LTX2VideoVaeNeighborhoodAttention", hidden_states: torch.Tensor) -> torch.Tensor:
        if self._attention_backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"LTX2VideoVaeNeighborhoodAttnProcessor requires the 'flex' attention backend (got "
                f"{self._attention_backend!r}). It builds a flex_attention.BlockMask for the neighborhood "
                f"window, which no other backend in `dispatch_attention_fn` accepts. To use NATTEN's kernels "
                f"instead, set LTX2VideoVaeNeighborhoodNattenProcessor via `set_attn_processor`."
            )

        batch_size, num_frames, height, width, _ = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)

        query = query.reshape(batch_size, num_frames * height * width, attn.heads, attn.head_dim)
        key = key.reshape(batch_size, num_frames * height * width, attn.heads, attn.head_dim)
        value = value.reshape(batch_size, num_frames * height * width, attn.heads, attn.head_dim)
        block_mask = _neighborhood_block_mask(num_frames, height, width, attn.kernel_size, hidden_states.device)
        # `scale=1.0`: the query is already scaled in `project_qkv`, as in the reference.
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=block_mask,
            scale=1.0,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, attn.heads * attn.head_dim)
        return attn.to_out[0](hidden_states)


class LTX2VideoVaeNeighborhoodNattenProcessor:
    """Neighborhood-attention processor using NATTEN's `na3d`, which is what the reference decoder calls.

    Requires the optional `natten` package and a supported GPU; `backend=None` lets NATTEN pick the fastest kernel for
    the device. No CPU path — use [`LTX2VideoVaeNeighborhoodAttnProcessor`] elsewhere.
    """

    def __init__(self, backend: str | None = None):
        if not is_natten_available():
            raise ImportError(
                "LTX2VideoVaeNeighborhoodNattenProcessor requires the `natten` package. Install it, or use the "
                "default `LTX2VideoVaeNeighborhoodAttnProcessor` (FlexAttention) instead."
            )
        self.backend = backend

    def __call__(self, attn: "LTX2VideoVaeNeighborhoodAttention", hidden_states: torch.Tensor) -> torch.Tensor:
        from natten.functional import na3d

        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        # NATTEN's CUTLASS kernels silently produce wrong output for non-contiguous inputs.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        # `scale=1.0`: the query is already scaled in `project_qkv`, as in the reference.
        hidden_states = na3d(query, key, value, kernel_size=attn.kernel_size, scale=1.0, backend=self.backend)
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](hidden_states)


class LTX2VideoVaeNeighborhoodAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = LTX2VideoVaeNeighborhoodAttnProcessor
    _available_processors = [LTX2VideoVaeNeighborhoodAttnProcessor, LTX2VideoVaeNeighborhoodNattenProcessor]
    # Both processors read `to_q`/`to_k`/`to_v` directly and have no fused path, so QKV fusion would build an
    # unused `to_qkv` and silently no-op. The flex/NATTEN swap goes through `set_attn_processor` instead.
    _supports_qkv_fusion = False

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim {dim} must be divisible by head_dim {head_dim}.")
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.rope = LTX2VideoVaeRotaryPosEmbed3D(head_dim, base=rope_base)
        self.set_processor(self._default_processor_cls())

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V as `(B, T, H, W, heads, head_dim)`, RMS-normed, query pre-scaled, then rotated.

        The query carries the `1 / sqrt(head_dim)` factor here so both processors can ask their attention backend for
        `scale=1.0` — this is the order the reference uses (norm, scale, then rotate).
        """
        batch_size, num_frames, height, width, _ = hidden_states.shape
        shape = (batch_size, num_frames, height, width, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape)
        key = self.to_k(hidden_states).view(shape)
        value = self.to_v(hidden_states).view(shape)

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query * self.scale
        return self.rope(query), self.rope(key), value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Channels-last in and out: `(B, T, H, W, C)`."""
        num_frames, height, width = hidden_states.shape[1:4]
        kernel_t, kernel_h, kernel_w = self.kernel_size
        if num_frames < kernel_t or height < kernel_h or width < kernel_w:
            raise ValueError(
                f"Neighborhood attention requires each spatial dim to be at least its kernel size; got "
                f"(T, H, W) = ({num_frames}, {height}, {width}) with kernel_size {self.kernel_size}."
            )
        return self.processor(self, hidden_states)


# Tokens per tile in `LTX2VideoVaeSwiGLU`, matching the reference decoder's own default. `w_gate(x)` and
# `w_up(x)` are both hidden-width and their product makes a third, so evaluating a whole video at once
# holds three hidden-width tensors live at the same time — at 121 frames and 512x768 that is
# 3 x 5.67 GiB, which by itself dominates decode memory. Fixing a token *count* rather than a number of
# tiles keeps that bound independent of resolution.
_SWIGLU_TILE_SIZE = 16384


class LTX2VideoVaeSwiGLU(nn.Module):
    """Gated MLP: `w_down(silu(w_gate(x)) * w_up(x))`, evaluated in tiles of `_SWIGLU_TILE_SIZE` tokens.

    Tiling is exact rather than an approximation: the MLP is pointwise across tokens, so splitting it changes only how
    many hidden-width elements exist at once, never a result. Verified bit-identical to the untiled evaluation on a
    full-size decode.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, *token_dims, channels = hidden_states.shape
        num_tokens = math.prod(token_dims)
        if num_tokens <= _SWIGLU_TILE_SIZE:
            return self.w_down(F.silu(self.w_gate(hidden_states)) * self.w_up(hidden_states))

        flat = hidden_states.reshape(batch_size, num_tokens, channels)
        out = torch.empty_like(flat)
        for start in range(0, num_tokens, _SWIGLU_TILE_SIZE):
            tile = flat[:, start : start + _SWIGLU_TILE_SIZE]
            out[:, start : start + _SWIGLU_TILE_SIZE] = self.w_down(F.silu(self.w_gate(tile)) * self.w_up(tile))
        return out.reshape(hidden_states.shape)


def _swiglu_hidden_dim(dim: int, mlp_ratio: float) -> int:
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class LTX2VideoVaeNABlock(nn.Module):
    """Pre-norm neighborhood-attention block used by the deterministic upsampling stages."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LTX2VideoVaeAdaLNZero(nn.Module):
    """Shared AdaLN-Zero modulation: a timestep embedding to seven `(B, 1, 1, 1, C)` chunks.

    Seven chunks (scale/shift/gate for attention and MLP, plus a context gate) is the reference's shape. Only the four
    scale/shift chunks are consumed: the decoder's residuals are ungated, and the static gates the checkpoint used to
    carry are folded into the following linear weights at conversion time.
    """

    num_chunks = 7

    def __init__(self, dim: int, t_emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.num_chunks * dim, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.proj(F.silu(t_emb)).chunk(self.num_chunks, dim=-1)
        return tuple(chunk[:, None, None, None, :] for chunk in chunks)


class LTX2VideoVaeDiffusionNABlock(nn.Module):
    """Neighborhood attention + SwiGLU, modulated by the shared AdaLN-Zero scale/shift.

    The decoder owns one `LTX2VideoVaeAdaLNZero`; each block adds its own `scale_shift_table` residual on top of it,
    injects the latent context through `context_proj`, and keeps its residuals ungated.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.context_channels = context_channels
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(LTX2VideoVaeAdaLNZero.num_chunks, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1)
            for i in range(LTX2VideoVaeAdaLNZero.num_chunks)
        ]

        hidden_states = hidden_states + self.context_proj(latent_context)
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states) * (1 + scale_msa) + shift_msa)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
        return hidden_states


class LTX2VideoVaePixelShuffleUpsampler(nn.Module):
    """Linear channel expansion followed by a channels-last pixel shuffle.

    When the temporal stride is 2 the shuffle produces a duplicate leading frame, which is dropped to keep the causal
    1:2 (composed 1:8) frame mapping.
    """

    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(self.stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(self.stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        stride_t, stride_h, stride_w = self.stride
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, height, width, self.out_channels, stride_t, stride_h, stride_w
        )
        # (b, t, p1, h, p2, w, p3, c) -> merge each stride into its own axis
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames * stride_t, height * stride_h, width * stride_w, self.out_channels
        )
        if stride_t == 2:
            hidden_states = hidden_states[:, 1:]
        return hidden_states


class LTX2VideoDiffusionDecoder3d(nn.Module):
    """The LTX-2.4 diffusion video decoder.

    Stages 1-4 deterministically upsample the latent into a context volume with neighborhood-attention blocks. Stage 5
    then denoises patchified pixels, conditioned on that context through AdaLN-Zero scale/shift. With
    `model_output_type="x0"` and a single step — how LTX-2.4 ships — stage 5 runs once and its prediction *is* the
    output; more steps add reverse Euler updates.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        upsample_channel_reductions: tuple[int, ...] = (2, 2, 1, 2),
        stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        t_emb_dim: int = 384,
        temporal_compression_ratio: int = 8,
        timestep_scale_multiplier: float = 1000.0,
        model_output_type: str = "x0",
        default_num_inference_steps: int = 1,
    ):
        super().__init__()
        if model_output_type not in ("x0", "v"):
            raise ValueError(f"model_output_type must be 'x0' or 'v', got {model_output_type!r}.")
        # Each upsample divides the channel count by its reduction factor, so the stage widths and the
        # reductions are two views of the same thing and an inconsistent pair would only fail deep inside
        # the first block.
        for stage_idx, reduction in enumerate(upsample_channel_reductions):
            expected = stage_channels[stage_idx] // reduction
            if stage_channels[stage_idx + 1] != expected:
                raise ValueError(
                    f"stage_channels[{stage_idx + 1}] must be stage_channels[{stage_idx}] // "
                    f"upsample_channel_reductions[{stage_idx}] = {expected}, got {stage_channels[stage_idx + 1]}."
                )

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.default_num_inference_steps = default_num_inference_steps
        self.temporal_compression_ratio = temporal_compression_ratio
        self.context_channels = stage_channels[-1]
        # NATTEN shifts its window inward at the grid border, so the last latent frame is replicated
        # through stages 1-4 and cropped off the context before stage 5, moving that border past the
        # frames that are kept.
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_idx, stride in enumerate(upsample_strides):
            channels = stage_channels[stage_idx]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        LTX2VideoVaeNABlock(
                            dim=channels,
                            kernel_size=stage_kernels[stage_idx],
                            head_dim=head_dim,
                        )
                        for _ in range(stage_depths[stage_idx])
                    ]
                )
            )
            self.upsamples.append(
                LTX2VideoVaePixelShuffleUpsampler(
                    in_channels=channels,
                    stride=stride,
                    out_channels_reduction_factor=upsample_channel_reductions[stage_idx],
                )
            )

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)

        stage5_channels = stage_channels[-1]
        noised_pixel_channels = out_channels * patch_size**2
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, stage5_channels, bias=True)
        self.shared_adaln = LTX2VideoVaeAdaLNZero(dim=stage5_channels, t_emb_dim=t_emb_dim)
        self.diff_blocks = nn.ModuleList(
            [
                LTX2VideoVaeDiffusionNABlock(
                    dim=stage5_channels,
                    kernel_size=stage5_kernel,
                    context_channels=self.context_channels,
                    head_dim=head_dim,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, noised_pixel_channels, bias=True)

    def forward_pre_diffusion(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Stages 1-4: latent `(B, C, T, H, W)` to context `(B, T_5, H_5, W_5, C_5)`, channels-last."""
        num_pad = self.trailing_pad_latent_frames
        if num_pad > 0:
            trailing = hidden_states[:, :, -1:].expand(-1, -1, num_pad, -1, -1)
            hidden_states = torch.cat([hidden_states, trailing], dim=2)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in(hidden_states)
        for blocks, upsample in zip(self.det_stages, self.upsamples):
            for block in blocks:
                hidden_states = block(hidden_states)
            hidden_states = upsample(hidden_states)

        if num_pad > 0:
            hidden_states = hidden_states[:, : -num_pad * self.temporal_compression_ratio]
        return hidden_states

    def forward_diffusion_step(
        self, latent_context: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """One stage-5 step. Returns the model's prediction in pixel space, `(B, C, F, H, W)`."""
        t_emb = self.t_embedder(
            self.timestep_scale_multiplier * timestep,
            resolution=None,
            aspect_ratio=None,
            batch_size=timestep.shape[0],
            hidden_dtype=latent_context.dtype,
        )
        modulation = self.shared_adaln(t_emb)

        hidden_states = _patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in_x_t(hidden_states)
        for block in self.diff_blocks:
            hidden_states = block(hidden_states, latent_context, modulation)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3).contiguous()
        return _unpatchify(hidden_states, self.patch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        batch_size, _, num_frames, height, width = hidden_states.shape
        height_ratio = math.prod(upsample.stride[1] for upsample in self.upsamples) * self.patch_size
        width_ratio = math.prod(upsample.stride[2] for upsample in self.upsamples) * self.patch_size
        # The temporal upsamples drop their duplicate leading frame, so T latent frames decode to
        # (T - 1) * ratio + 1 pixel frames — the causal 1-then-`ratio` mapping of the LTX-2 latent space.
        pixel_shape = (
            batch_size,
            self.out_channels,
            (num_frames - 1) * self.temporal_compression_ratio + 1,
            height * height_ratio,
            width * width_ratio,
        )

        latent_context = self.forward_pre_diffusion(hidden_states)
        x_t = randn_tensor(pixel_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)
        timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps, device=hidden_states.device, dtype=torch.float32
        )

        if num_inference_steps == 1 and self.model_output_type == "x0":
            return self.forward_diffusion_step(latent_context, x_t, timesteps[:1].expand(batch_size))

        for step_idx in range(num_inference_steps):
            t_now = timesteps[step_idx].expand(batch_size)
            t_next = timesteps[step_idx + 1] if step_idx + 1 < num_inference_steps else torch.zeros_like(t_now)
            model_out = self.forward_diffusion_step(latent_context, x_t, t_now).float()
            x_t_fp32 = x_t.float()
            if self.model_output_type == "x0":
                sigma = t_now.view(-1, *([1] * (x_t.ndim - 1)))
                model_out = (x_t_fp32 - model_out) / sigma
            dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1)))
            x_t = (x_t_fp32 - dt * model_out).to(x_t.dtype)
        return x_t


class AutoencoderKLLTX2VideoDiffusionDecoder(ModelMixin, AutoencoderMixin, AttentionMixin, ConfigMixin):
    r"""
    The LTX-2.4 video VAE: the LTX-2 causal convolutional encoder paired with the diffusion decoder used from LTX-2.4
    onwards.

    The encoder is unchanged from [`AutoencoderKLLTX2Video`] — same weights, same latent space — so latents are
    interchangeable between the two and either decoder can consume them.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    # Both block types close over a residual add that combines outputs from different children, so a device split
    # inside either one would separate tensors that have to meet again in the same forward.
    _no_split_modules = ["LTX2VideoVaeNABlock", "LTX2VideoVaeDiffusionNABlock"]
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 128,
        block_out_channels: tuple[int, ...] = (256, 512, 1024, 2048),
        down_block_types: tuple[str, ...] = (
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ),
        layers_per_block: tuple[int, ...] = (4, 6, 6, 2, 2),
        spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, True),
        downsample_type: tuple[str, ...] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        scaling_factor: float = 1.0,
        encoder_causal: bool = True,
        encoder_spatial_padding_mode: str = "zeros",
        decoder_head_dim: int = 64,
        decoder_stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        decoder_stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        decoder_stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        decoder_upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        decoder_upsample_channel_reductions: tuple[int, ...] = (2, 2, 1, 2),
        decoder_stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        decoder_t_emb_dim: int = 384,
        decoder_timestep_scale_multiplier: float = 1000.0,
        decoder_model_output_type: str = "x0",
        decoder_num_inference_steps: int = 1,
        spatial_compression_ratio: int = 32,
        temporal_compression_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.encoder = LTX2VideoEncoder3d(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            downsample_type=downsample_type,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=encoder_causal,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )
        self.decoder = LTX2VideoDiffusionDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            head_dim=decoder_head_dim,
            stage_channels=decoder_stage_channels,
            stage_depths=decoder_stage_depths,
            stage_kernels=decoder_stage_kernels,
            upsample_strides=decoder_upsample_strides,
            upsample_channel_reductions=decoder_upsample_channel_reductions,
            stage5_kernel=decoder_stage5_kernel,
            t_emb_dim=decoder_t_emb_dim,
            temporal_compression_ratio=temporal_compression_ratio,
            timestep_scale_multiplier=decoder_timestep_scale_multiplier,
            model_output_type=decoder_model_output_type,
            default_num_inference_steps=decoder_num_inference_steps,
        )

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio
        # Batch slicing is supported (`enable_slicing`); tiling is not — see `enable_tiling`.
        self.use_slicing = False

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, causal: bool | None = None, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        """Encode a batch of videos into latents."""
        if self.use_slicing and x.shape[0] > 1:
            h = torch.cat([self.encoder(x_slice, causal=causal) for x_slice in x.split(1)])
        else:
            h = self.encoder(x, causal=causal)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | torch.Tensor:
        """Decode a batch of latents.

        Unlike a convolutional decoder this one denoises, so it draws noise: pass `generator` for reproducibility. `z`
        is expected to be denormalized already (the pipeline applies `latents_mean` / `latents_std`), matching
        [`AutoencoderKLLTX2Video`].
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat(
                [
                    self.decoder(z_slice, generator=generator, num_inference_steps=num_inference_steps)
                    for z_slice in z.split(1)
                ]
            )
        else:
            decoded = self.decoder(z, generator=generator, num_inference_steps=num_inference_steps)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        num_inference_steps: int | None = None,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps the decoder takes. If `None`, falls back to the model default.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make the
                decoder's noise — and the posterior sampling, if enabled — deterministic. The decoder denoises, so
                without one every call returns a different result.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If `return_dict` is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        decoded = self.decode(z, generator=generator, num_inference_steps=num_inference_steps).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def enable_tiling(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Tiled decoding is not supported for the LTX-2.4 diffusion decoder: its neighborhood attention "
            "rejects tiles smaller than its kernel, so tile sizes cannot be chosen freely. Decode untiled."
        )
