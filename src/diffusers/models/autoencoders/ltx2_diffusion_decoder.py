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
from ...utils import is_kernels_available, logging
from ...utils.accelerate_utils import apply_forward_hook
from ...utils.constants import DIFFUSERS_DISABLE_REMOTE_CODE
from ...utils.torch_utils import maybe_adjust_dtype_for_device
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


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

    def __call__(
        self, attn: "LTX2VideoVaeNeighborhoodAttention", hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
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
        if block_mask is None:
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

    NATTEN is fetched from the Hub (`shi-labs/natten`, a trusted kernel publisher) through the `kernels` package rather
    than imported from a local install, so it needs `kernels`, a supported GPU, and `DIFFUSERS_DISABLE_REMOTE_CODE`
    unset; `backend=None` lets NATTEN pick the fastest kernel for the device. No CPU path — use
    [`LTX2VideoVaeNeighborhoodAttnProcessor`] elsewhere.

    `na3d` encodes the neighborhood window in the kernel itself, so this processor takes no `block_mask` and the
    decoder never builds one for it.
    """

    def __init__(self, backend: str | None = None):
        if DIFFUSERS_DISABLE_REMOTE_CODE:
            raise ValueError(
                "LTX2VideoVaeNeighborhoodNattenProcessor downloads the `shi-labs/natten` kernel from the Hub, which "
                "is disabled globally by the `DIFFUSERS_DISABLE_REMOTE_CODE` environment variable. Unset it, or use "
                "the default `LTX2VideoVaeNeighborhoodAttnProcessor` (FlexAttention) instead."
            )
        if not is_kernels_available():
            raise ImportError(
                "LTX2VideoVaeNeighborhoodNattenProcessor fetches NATTEN from the Hub with the `kernels` package. "
                "Install it with `pip install kernels`, or use the default "
                "`LTX2VideoVaeNeighborhoodAttnProcessor` (FlexAttention) instead."
            )
        from kernels import get_kernel

        self._na3d = get_kernel("shi-labs/natten", version=1).na3d
        self.backend = backend

    def __call__(
        self, attn: "LTX2VideoVaeNeighborhoodAttention", hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        # NATTEN's CUTLASS kernels silently produce wrong output for non-contiguous inputs.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        # `scale=1.0`: the query is already scaled in `project_qkv`, as in the reference.
        hidden_states = self._na3d(query, key, value, kernel_size=attn.kernel_size, scale=1.0, backend=self.backend)
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

    def build_block_mask(self, hidden_states: torch.Tensor):
        """The flex `BlockMask` for this grid, or `None` if the processor does not read one.

        The mask depends only on the token grid and the kernel, both of which are fixed within a decoder stage, so a
        stage builds it once and hands it to every block. NATTEN gets `None`: it encodes the window in its kernel, and
        at production grids the mask is not merely wasteful but larger than device memory (a 69x64x96 stage needs 167
        GiB), so it must never be built for a path that would not read it.
        """
        if not isinstance(self.processor, LTX2VideoVaeNeighborhoodAttnProcessor):
            return None
        num_frames, height, width = hidden_states.shape[1:4]
        return _neighborhood_block_mask(num_frames, height, width, self.kernel_size, hidden_states.device)

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        """Channels-last in and out: `(B, T, H, W, C)`."""
        num_frames, height, width = hidden_states.shape[1:4]
        kernel_t, kernel_h, kernel_w = self.kernel_size
        if num_frames < kernel_t or height < kernel_h or width < kernel_w:
            raise ValueError(
                f"Neighborhood attention requires each spatial dim to be at least its kernel size; got "
                f"(T, H, W) = ({num_frames}, {height}, {width}) with kernel_size {self.kernel_size}."
            )
        return self.processor(self, hidden_states, block_mask)


# Tokens per tile in `LTX2VideoVaeSwiGLU`, matching the reference decoder's own default. `w_gate(x)` and
# `w_up(x)` are both hidden-width and their product makes a third, so evaluating a whole video at once
# holds three hidden-width tensors live at the same time — at 121 frames and 512x768 that is
# 3 x 5.67 GiB, which by itself dominates decode memory. Fixing a token *count* rather than a number of
# tiles keeps that bound independent of resolution.
_SWIGLU_TILE_SIZE = 16384


class LTX2VideoVaeSwiGLU(nn.Module):
    """Gated MLP: `w_down(silu(w_gate(x)) * w_up(x))`, evaluated in tiles of `_SWIGLU_TILE_SIZE` tokens.

    Tiling is not an approximation: the MLP is pointwise across tokens, so splitting it changes only how many
    hidden-width elements exist at once, never what is computed. Outputs can still differ from the untiled evaluation
    by a few ulps, since a matmul over a tile may reduce in a different order than the full-tensor call.
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

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), block_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LTX2VideoVaeAdaLNZero(nn.Module):
    """Shared AdaLN-Zero modulation: a timestep embedding to seven `(B, 1, 1, 1, C)` chunks.

    Seven chunks (scale/shift/gate for attention and MLP, plus a context gate) is the reference's shape. Only the four
    scale/shift chunks are consumed: the decoder's residuals are ungated, and the static gates the checkpoint used to
    carry are folded into the following linear weights at conversion time.
    """

    def __init__(self, dim: int, t_emb_dim: int, num_chunks: int = 7):
        super().__init__()
        self.num_chunks = num_chunks
        self.proj = nn.Linear(t_emb_dim, num_chunks * dim, bias=True)

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
        num_mod_params: int = 7,
    ):
        super().__init__()
        self.context_channels = context_channels
        self.num_mod_params = num_mod_params
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(num_mod_params, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        block_mask=None,
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(self.num_mod_params)
        ]

        hidden_states = hidden_states + self.context_proj(latent_context)
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states) * (1 + scale_msa) + shift_msa, block_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
        return hidden_states


class LTX2VideoVaePixelShuffleUpsampler(nn.Module):
    """Linear channel expansion followed by a channels-last pixel shuffle.

    When the temporal stride is 2 the shuffle produces a duplicate leading frame, which is dropped to keep the causal
    1:2 (composed 1:8) frame mapping. `drop_leading_frame=False` keeps it: a tiled decode passes that for temporal
    tiles that do not contain t=0, where the first input frame is an interior frame whose two output frames are both
    real content.
    """

    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(self.stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(self.stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
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
        if stride_t == 2 and drop_leading_frame:
            hidden_states = hidden_states[:, 1:]
        return hidden_states


class LTX2VideoDiffusionDecoder3d(nn.Module):
    """The LTX-2.5 diffusion video decoder.

    Stages 1-4 deterministically upsample the latent into a context volume with neighborhood-attention blocks; that
    volume conditions stage 5, which is an ordinary diffusion transformer over patchified pixels.
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
                    num_mod_params=self.shared_adaln.num_chunks,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, noised_pixel_channels, bias=True)

    def encode_context_stages_1_to_3(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """All deterministic context stages but the last: latent `(B, C, T, H, W)` to a channels-last feature volume.

        The trailing ghost frames added for NATTEN's border shift stay in the output; [`encode_context_stage_4`] crops
        them. The split at this point exists for tiled decoding: these stages are cheap enough to run on the full
        volume, while stage 4 and the diffusion stage — where the grid and the channel-hidden products get large — run
        per tile.
        """
        num_pad = self.trailing_pad_latent_frames
        if num_pad > 0:
            trailing = hidden_states[:, :, -1:].expand(-1, -1, num_pad, -1, -1)
            hidden_states = torch.cat([hidden_states, trailing], dim=2)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in(hidden_states)
        for blocks, upsample in zip(self.det_stages[:-1], self.upsamples[:-1]):
            # The grid and kernel are fixed within a stage, so every block shares one mask.
            block_mask = blocks[0].attn.build_block_mask(hidden_states)
            for block in blocks:
                hidden_states = block(hidden_states, block_mask)
            hidden_states = upsample(hidden_states)
        return hidden_states

    def encode_context_stage_4(
        self, hidden_states: torch.Tensor, drop_leading_frame: bool = True, crop_trailing_ghost: bool = True
    ) -> torch.Tensor:
        """Last deterministic stage: [`encode_context_stages_1_to_3`] output to context `(B, T_5, H_5, W_5, C_5)`.

        The defaults describe the untiled decode. A tiled decode overrides them per temporal tile: only the tile
        containing t=0 drops the upsample's duplicate leading frame, and only the tile containing the video end carries
        the trailing ghost frames to crop.
        """
        blocks = self.det_stages[-1]
        block_mask = blocks[0].attn.build_block_mask(hidden_states)
        for block in blocks:
            hidden_states = block(hidden_states, block_mask)
        hidden_states = self.upsamples[-1](hidden_states, drop_leading_frame=drop_leading_frame)

        num_pad = self.trailing_pad_latent_frames
        if crop_trailing_ghost and num_pad > 0:
            hidden_states = hidden_states[:, : -num_pad * self.temporal_compression_ratio]
        return hidden_states

    def forward(
        self, hidden_states: torch.Tensor, latent_context: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        r"""
        One stage-5 denoising step.

        Args:
            hidden_states (`torch.Tensor`):
                Noised pixels of shape `(B, C, F, H, W)`.
            latent_context (`torch.Tensor`):
                The conditioning volume from [`encode_context_stage_4`], of shape `(B, F, H // patch_size, W //
                patch_size, C_5)`. It is projected into the residual stream of every block rather than cross-attended,
                and shares the token grid with `hidden_states`.
            timestep (`torch.Tensor`):
                Noise level in `[0, 1]`, of shape `(B,)`.

        Returns:
            `torch.Tensor`: the model's prediction in pixel space, `(B, C, F, H, W)`.
        """
        t_emb = self.t_embedder(
            self.timestep_scale_multiplier * timestep,
            resolution=None,
            aspect_ratio=None,
            batch_size=timestep.shape[0],
            hidden_dtype=latent_context.dtype,
        )
        modulation = self.shared_adaln(t_emb)

        hidden_states = _patchify(hidden_states, self.patch_size).permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in_x_t(hidden_states)
        block_mask = self.diff_blocks[0].attn.build_block_mask(hidden_states)
        for block in self.diff_blocks:
            hidden_states = block(hidden_states, latent_context, modulation, block_mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3).contiguous()
        return _unpatchify(hidden_states, self.patch_size)


class LTX2VideoDiffusionDecoderModel(ModelMixin, AttentionMixin, ConfigMixin):
    r"""
    The LTX-2 diffusion video decoder, introduced in LTX-2.5.

    This is a decoder, not an autoencoder: it has no encoder and cannot produce latents. Encoding stays with
    [`AutoencoderKLLTX2Video`], whose latent space this consumes unchanged, so latents are interchangeable between the
    convolutional decoder and this one.

    It is also a diffusion model rather than a deterministic decoder — it denoises pixels conditioned on a context
    volume built from the latents — which is why it is driven by [`LTX2VideoDiffusionDecodePipeline`] rather than being
    passed as a pipeline's `vae`. [`forward`] is a single denoising step, like any other denoiser in the library: the
    loop over steps, the scheduler that drives it, and the tiling wrapped around it all live in that pipeline. What
    stays here is the model itself, plus the tile *sizes* — [`enable_tiling`] configures the pipeline's tiling the way
    `vae.enable_tiling()` does everywhere else.

    The latent statistics are carried here as buffers so the decode pipeline can denormalize without loading a second
    autoencoder just for two vectors.

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
        out_channels: int = 3,
        latent_channels: int = 128,
        patch_size: int = 4,
        scaling_factor: float = 1.0,
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

        # When decoding a large enough video, the memory-dominant stages (the last deterministic stage and the
        # stage-5 diffusion blocks) can run on overlapping tiles that are blended back together. The earlier
        # stages always see the full latent, so tiling changes the output only near tile borders. The decode
        # pipeline reads the settings below; nothing here acts on them.
        self.use_tiling = False

        # The tile size and the distance between the starts of two consecutive tiles, in pixels/frames of the
        # decoded video; their difference is the blended overlap. Defaults match the reference implementation.
        self.tile_sample_min_height = 768
        self.tile_sample_min_width = 768
        self.tile_sample_min_num_frames = 80
        self.tile_sample_stride_height = 704
        self.tile_sample_stride_width = 704
        self.tile_sample_stride_num_frames = 56

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_num_frames: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
        tile_sample_stride_num_frames: int | None = None,
    ) -> None:
        r"""
        Enable tiled decoding. The deterministic upsampling stages before the last one always process the full latent
        (they run at low resolution and are cheap); the last stage and the stage-5 diffusion blocks — which dominate
        decode memory — run on overlapping tiles whose seams are blended linearly.

        These are settings, not behaviour: [`LTX2VideoDiffusionDecodePipeline`] reads them when it decodes.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The height of one decoded tile, in pixels.
            tile_sample_min_width (`int`, *optional*):
                The width of one decoded tile, in pixels.
            tile_sample_min_num_frames (`int`, *optional*):
                The number of frames of one decoded tile.
            tile_sample_stride_height (`int`, *optional*):
                The distance in pixels between the tops of two consecutive vertical tiles; the difference to
                `tile_sample_min_height` is the blended overlap.
            tile_sample_stride_width (`int`, *optional*):
                The distance in pixels between the left edges of two consecutive horizontal tiles.
            tile_sample_stride_num_frames (`int`, *optional*):
                The distance in frames between the starts of two consecutive temporal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        r"""Disable tiled decoding, returning to decoding the whole video in one pass."""
        self.use_tiling = False

    # `@apply_forward_hook` on both context stages: accelerate's offload hooks fire on `forward`, and the
    # decode pipeline calls these before it ever calls one, so without it a CPU-offloaded model stays on the
    # CPU here.
    @apply_forward_hook
    def encode_context_stages_1_to_3(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""All deterministic context stages but the last: latent `(B, C, T, H, W)` to a channels-last feature volume.

        The trailing ghost frames added for NATTEN's border shift stay in the output; [`encode_context_stage_4`] crops
        them. The split exists for tiled decoding: these stages are cheap enough to run on the full volume, while stage
        4 and the diffusion stage — where the grid and the channel-hidden products get large — run per tile.
        """
        return self.decoder.encode_context_stages_1_to_3(hidden_states)

    @apply_forward_hook
    def encode_context_stage_4(
        self, hidden_states: torch.Tensor, drop_leading_frame: bool = True, crop_trailing_ghost: bool = True
    ) -> torch.Tensor:
        r"""Last deterministic stage: [`encode_context_stages_1_to_3`] output to context `(B, T_5, H_5, W_5, C_5)`.

        The defaults describe an untiled decode. A tiled decode overrides them per temporal tile: only the tile
        containing t=0 drops the upsample's duplicate leading frame, and only the tile containing the video end carries
        the trailing ghost frames to crop.
        """
        return self.decoder.encode_context_stage_4(
            hidden_states, drop_leading_frame=drop_leading_frame, crop_trailing_ghost=crop_trailing_ghost
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        timestep: torch.Tensor,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        r"""
        One denoising step. The loop over steps, and the tiling around it, belong to
        [`LTX2VideoDiffusionDecodePipeline`].

        Args:
            hidden_states (`torch.Tensor`):
                Noised pixels of shape `(B, C, F, H, W)`.
            latent_context (`torch.Tensor`):
                The conditioning volume from [`encode_context_stage_4`], of shape `(B, F, H // patch_size, W //
                patch_size, C_5)`. It is projected into the residual stream of every block rather than cross-attended,
                and shares the token grid with `hidden_states`.
            timestep (`torch.Tensor`):
                Noise level in `[0, 1]`, of shape `(B,)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`: the model's prediction in pixel space,
            `(B, C, F, H, W)`. Whether that is the denoised sample or the velocity is set by the
            `decoder_model_output_type` config value.
        """
        sample = self.decoder(hidden_states, latent_context, timestep)

        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)
