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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..activations import get_activation
from ..attention import AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import RMSNorm


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


class SanaWMTemporalShortConvolution(nn.Module):
    """Depthwise short convolution over the temporal axis, run in both directions.

    SANA-WM's GDN attention applies a short depthwise conv to Q/K/V before the linear-attention kernel. The reference
    implementation used the *causal* `fla.modules.ShortConvolution` layer (with `activation=None`); this is a
    self-contained PyTorch implementation -- so the model needs no `fla-core` dependency and can be built on any device
    -- that runs the causal kernel forwards and backwards to obtain the non-causal filter the bidirectional model
    needs.

    A causal depthwise Conv1d with kernel ``[w_0, w_1, ..., w_{k-1}]`` computes at time *t*:

        ``y_fwd[t] = w_0 * x[t-k+1] + ... + w_{k-1} * x[t]``

    Running the same kernel on the time-flipped input and flipping back gives:

        ``y_bwd[t] = w_{k-1} * x[t] + ... + w_0 * x[t+k-1]``

    Both passes include the current timestep ``x[t]`` with the center weight ``w_{k-1}``. To avoid double-counting one
    copy of the center contribution is subtracted:

        ``y = y_fwd + y_bwd - w_{k-1} * x``

    The result is a symmetric temporal filter where every position in the window ``[t-k+1, t+k-1]`` is counted exactly
    once.

    Args:
        hidden_size (`int`): Number of channels (the conv is depthwise, one group per channel).
        kernel_size (`int`): Temporal kernel width.
        bias (`bool`, defaults to `False`): Whether to add a per-channel bias.
    """

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        # Same parameter layout as the reference implementation: (C, 1, K).
        self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def _causal_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Depthwise causal conv over `(batch, seq_len, hidden_size)` inputs."""
        seq_len = hidden_states.shape[1]
        # Left-pad by (K - 1) and drop the tail so output[t] only sees inputs <= t.
        hidden_states = F.conv1d(
            hidden_states.transpose(1, 2),
            self.weight.to(hidden_states.dtype),
            None if self.bias is None else self.bias.to(hidden_states.dtype),
            groups=self.hidden_size,
            padding=self.kernel_size - 1,
        )[..., :seq_len]
        return hidden_states.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Apply the bidirectional conv along the temporal axis, with the spatial axis merged into the batch.

        Args:
            hidden_states (`torch.Tensor`): Input of shape `(batch, num_frames * spatial_size, hidden_size)`.
            num_frames (`int`): Number of frames the sequence axis is split into.

        Returns:
            `torch.Tensor`: Tensor of the same shape and dtype as `hidden_states`.
        """
        batch_size, seq_len, channels = hidden_states.shape
        spatial_size = seq_len // num_frames
        dtype_in = hidden_states.dtype

        # (B, T * S, C) -> (B * S, T, C). The causal conv backward is not reliable on the non-contiguous
        # strided layout this permutation produces, hence the explicit `contiguous()`.
        hidden_states = (
            hidden_states.reshape(batch_size, num_frames, spatial_size, channels)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * spatial_size, num_frames, channels)
        )

        causal_forward = self._causal_conv(hidden_states)
        causal_backward = self._causal_conv(hidden_states.flip(1)).flip(1)

        # Subtract the shared center tap (last weight of the causal kernel). Weight shape: (channels, 1, kernel_size),
        # so the last element along dim=-1 is the weight applied to x[t].
        center_term = hidden_states * self.weight[:, 0, -1].unsqueeze(0).unsqueeze(0)

        hidden_states = causal_forward + causal_backward - center_term
        if hidden_states.dtype != dtype_in:
            hidden_states = hidden_states.to(dtype_in)

        return (
            hidden_states.reshape(batch_size, spatial_size, num_frames, channels)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, channels)
        )


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

    def forward(self, hidden_states: torch.Tensor, HW: Tuple[int, int, int]) -> torch.Tensor:
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


class SanaWMCrossAttnProcessor:
    """Cross-attention from image queries to the text condition."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MultiHeadCrossAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, channels = hidden_states.shape

        query = attn.q_linear(hidden_states)
        key, value = attn.kv_linear(encoder_hidden_states).view(batch_size, -1, 2, channels).unbind(2)

        query = attn.q_norm(query).view(batch_size, -1, attn.heads, attn.head_dim)
        key = attn.k_norm(key).view(batch_size, -1, attn.heads, attn.head_dim)
        value = value.view(batch_size, -1, attn.heads, attn.head_dim)

        # A boolean mask (rather than an additive float one) keeps the varlen backends usable.
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask.bool()[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, -1, channels).type_as(query)
        return attn.proj(hidden_states)


class MultiHeadCrossAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = SanaWMCrossAttnProcessor
    _available_processors = [SanaWMCrossAttnProcessor]

    def __init__(self, d_model, num_heads, qk_norm=False, processor=None, **block_kwargs):
        super().__init__()
        if not (d_model % num_heads == 0):
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.heads = num_heads
        self.head_dim = d_model // num_heads
        self.inner_dim = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.proj = nn.Linear(d_model, d_model)
        if qk_norm:
            self.q_norm = RMSNorm(d_model, eps=1e-6)
            self.k_norm = RMSNorm(d_model, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.set_processor(processor if processor is not None else self._default_processor_cls())

    def forward(self, x, cond, mask=None):
        return self.processor(self, x, cond, attention_mask=mask)


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
        # `get_parameter_dtype` is layerwise-casting aware: under layerwise casting the storage dtype
        # (e.g. FP8) differs from the compute dtype, and `next(self.parameters()).dtype` returns the former.
        return get_parameter_dtype(self)


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


class SanaWMRotaryPosEmbed(nn.Module):
    """Rotary position embedding for SANA-WM.

    Deliberately not shared with Wan's rotary embedding: the per-axis split is configurable through `fhw_dim`, and the
    frequencies stay complex in a single `freqs` buffer rather than being split into real cos/sin buffers.
    """

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
            if not (attention_head_dim == sum(fhw_dim)):
                raise ValueError(f"attention_head_dim {attention_head_dim} must match sum(fhw_dim) {sum(fhw_dim)}")
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
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
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
    if not (transforms.shape[-2:] == (4, 4)):
        raise ValueError(f"`transforms` must have shape (..., 4, 4), got {tuple(transforms.shape)}.")
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

    # Uniform chunk boundaries over the temporal axis. A small trailing remainder is absorbed
    # into the last chunk, since `causal_conv1d` crashes on length-1 sequences.
    boundaries = list(range(0, T, chunk_size)) or [0]
    if len(boundaries) > 1 and (T - boundaries[-1]) < chunk_size:
        boundaries.pop()
    if boundaries[-1] != T:
        boundaries.append(T)
    split_sizes = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

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

    final_num = out_num.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)
    final_den = out_den.permute(0, 1, 3, 2, 4).reshape(B, H, 1, N)

    if return_components:
        return final_num, final_den

    return final_num / (final_den + eps)


# ---------------------------------------------------------------------------
# Helpers for hot-path operations
# ---------------------------------------------------------------------------


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
    # Uniform chunk boundaries over the temporal axis. A small trailing remainder is absorbed
    # into the last chunk, since `causal_conv1d` crashes on length-1 sequences.
    boundaries = list(range(0, T, chunk_size)) or [0]
    if len(boundaries) > 1 and (T - boundaries[-1]) < chunk_size:
        boundaries.pop()
    if boundaries[-1] != T:
        boundaries.append(T)
    split_sizes = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

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


def _prepare_frame_valid_masks(
    frame_valid_mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_frames: int,
    spatial_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Convert a frame-valid mask to the token / beta / decay masks the attention branches use.

    Args:
        frame_valid_mask (`torch.Tensor`, *optional*):
            Per-frame validity mask shaped `(B, 1, T, 1, 1)`, `(B, 1, T)` or `(B, T)`. `None` disables all masking.
        batch_size (`int`): Batch size `B`.
        num_frames (`int`): Number of frames `T`.
        spatial_size (`int`): Tokens per frame `S`.
        device (`torch.device`): Device of the returned masks.
        dtype (`torch.dtype`): Dtype of the returned masks.

    Returns:
        `tuple`: `(token_valid_mask, beta_valid_mask, decay_valid_mask)` shaped `(B, T * S)`, `(B, 1, T, 1)` and `(B,
        1, T)`, or three `None` when `frame_valid_mask` is `None`.
    """
    if frame_valid_mask is None:
        return None, None, None

    mask = frame_valid_mask
    if mask.ndim == 5:
        # (B, 1, T, 1, 1)
        mask = mask[:, 0, :, 0, 0]
    elif mask.ndim == 3 and mask.shape[1] == 1:
        # (B, 1, T)
        mask = mask[:, 0, :]
    elif mask.ndim != 2:
        raise ValueError(
            "frame_valid_mask must be shaped (B, 1, T, 1, 1), (B, 1, T), or (B, T); "
            f"got shape={list(frame_valid_mask.shape)}"
        )

    if mask.shape[0] != batch_size or mask.shape[1] != num_frames:
        raise ValueError(
            f"frame_valid_mask shape mismatch: expected (B={batch_size}, T={num_frames}), got {list(mask.shape)}"
        )

    mask = mask.to(device=device, dtype=dtype)
    token_valid_mask = mask[:, :, None].expand(batch_size, num_frames, spatial_size).reshape(batch_size, -1)
    beta_valid_mask = mask.view(batch_size, 1, num_frames, 1)
    decay_valid_mask = mask.view(batch_size, 1, num_frames)
    return token_valid_mask, beta_valid_mask, decay_valid_mask


def _downscale_to_reference_rms(
    reference: torch.Tensor,
    transformed: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Downscale a UCPE-transformed tensor if its channel RMS exceeds the reference.

    Args:
        reference (`torch.Tensor`): Pre-UCPE tensor carrying the target magnitude, shaped `(B, H, D, N)`.
        transformed (`torch.Tensor`): Tensor to stabilize, same shape as `reference`.
        eps (`float`, defaults to 1e-6): Numerical epsilon of the RMS.

    Returns:
        `torch.Tensor`: Stabilized tensor whose per-`(B, H, N)` channel RMS is not larger than the reference's.
    """
    reference_rms = reference.square().mean(dim=2, keepdim=True).add(eps).sqrt()
    transformed_rms = transformed.square().mean(dim=2, keepdim=True).add(eps).sqrt()
    scale = (reference_rms / transformed_rms.clamp_min(eps)).clamp(max=1.0)
    return transformed * scale


class SanaWMBidirectionalGDNAttention(nn.Module):
    """Bidirectional gated-delta-net linear attention over the temporal axis.

    The delta rule runs twice -- forwards (frames ``1..t``) and backwards (frames ``t+1..T``) -- and the numerator /
    denominator streams of both passes are summed before the final normalization. RoPE is applied to the numerator
    stream only, so the denominator (``Z``) stream keeps unrotated queries/keys and mass is conserved.

    This module holds no parameters. The projections, short convolutions, norms and gates feeding it live on
    [`BidirectionalGDNUCPESinglePathLiteLA`], whose `forward` issues every layer call and passes the resulting tensors
    in.

    Args:
        eps (`float`, defaults to 1e-15): Denominator epsilon of the linear-attention normalization.
        chunk_size (`int`, defaults to 21): Temporal chunk length of the state scan.
    """

    def __init__(self, eps: float = 1e-15, chunk_size: int = 21) -> None:
        super().__init__()
        self.eps = eps
        self.chunk_size = chunk_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        recall_gate: torch.Tensor,
        num_frames: int,
        rotary_emb: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        beta_valid_mask: torch.Tensor | None = None,
        decay_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the bidirectional delta rule.

        Args:
            query (`torch.Tensor`): Queries of shape `(B, N, H, D)`, already normalized and passed through the kernel.
            key (`torch.Tensor`): Keys of shape `(B, N, H, D)`, same preprocessing as `query`.
            value (`torch.Tensor`): Values of shape `(B, N, H, D)`.
            beta (`torch.Tensor`): Per-frame delta-rule gate of shape `(B, H, T, S)`.
            decay (`torch.Tensor`): Per-frame decay gate of shape `(B, H, T)`.
            recall_gate (`torch.Tensor`): Recall gate buffer forwarded to the scan.
            num_frames (`int`): Number of frames `T` the sequence axis is split into.
            rotary_emb (`torch.Tensor`, *optional*): Rotary embeddings applied to the numerator stream.
            token_valid_mask (`torch.Tensor`, *optional*): Token mask of shape `(B, N)`.
            beta_valid_mask (`torch.Tensor`, *optional*): Frame mask of shape `(B, 1, T, 1)` applied to `beta`.
            decay_valid_mask (`torch.Tensor`, *optional*): Frame mask of shape `(B, 1, T)` applied to `decay`.

        Returns:
            `torch.Tensor`: Raw attention output of shape `(B, N, H * D)`; the shared output gate and projection are
            applied by the caller.
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        spatial_size = seq_len // num_frames
        dtype_orig = value.dtype

        key = key * ((head_dim**-0.5) * (spatial_size**-0.5))

        # Permute to (B, H, D, N) for processing.
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(batch_size, 1, 1, seq_len)
            query = query * token_mask_qkv
            key = key * token_mask_qkv
            value = value * token_mask_qkv

        # RoPE preparation (numerator only).
        if rotary_emb is not None:
            query_rot = _apply_rotary_emb(query, rotary_emb)
            key_rot = _apply_rotary_emb(key, rotary_emb)
        else:
            query_rot = query
            key_rot = key
        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(batch_size, 1, 1, seq_len)
            query_rot = query_rot * token_mask_qkv
            key_rot = key_rot * token_mask_qkv

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_mask = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_mask + (1.0 - decay_mask)

        # Force FP32 to preserve recurrent stability.
        query = query.float()
        key = key.float()
        value = value.float()
        query_rot = query_rot.float()
        key_rot = key_rot.float()
        beta = beta.float()
        decay = decay.float()
        recall_gate = recall_gate.float()

        # Forward pass (inclusive: 1..t).
        num_fwd, den_fwd = torch_chunk_sana_gdn(
            query,
            key,
            value,
            query_rot,
            key_rot,
            beta,
            decay,
            recall_gate=recall_gate,
            chunk_size=self.chunk_size,
            eps=self.eps,
            return_components=True,
        )

        # Backward pass (exclusive: t+1..T).
        def to_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, num_heads, head_dim, num_frames, spatial_size).permute(0, 1, 3, 2, 4)

        def from_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, seq_len)

        query_bwd = torch.flip(to_time_structure(query), dims=[2])
        query_rot_bwd = torch.flip(to_time_structure(query_rot), dims=[2])
        key_bwd = flip_and_shift(to_time_structure(key), dim=2, shift_val=0.0)
        value_bwd = flip_and_shift(to_time_structure(value), dim=2, shift_val=0.0)
        key_rot_bwd = flip_and_shift(to_time_structure(key_rot), dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        num_bwd_flipped, den_bwd_flipped = torch_chunk_sana_gdn(
            from_time_structure(query_bwd),
            from_time_structure(key_bwd),
            from_time_structure(value_bwd),
            from_time_structure(query_rot_bwd),
            from_time_structure(key_rot_bwd),
            beta_bwd,
            decay_bwd,
            recall_gate=recall_gate,
            chunk_size=self.chunk_size,
            eps=self.eps,
            return_components=True,
        )

        def flip_back(tensor: torch.Tensor) -> torch.Tensor:
            # The denominator stream carries a single channel, hence the runtime `dim` lookup.
            dim = tensor.shape[2]
            tensor = tensor.view(batch_size, num_heads, dim, num_frames, spatial_size)
            return torch.flip(tensor, dims=[3]).reshape(batch_size, num_heads, dim, seq_len)

        total_num = num_fwd + flip_back(num_bwd_flipped)
        total_den = den_fwd + flip_back(den_bwd_flipped)

        hidden_states = total_num / (total_den + self.eps)

        if dtype_orig != torch.float32:
            hidden_states = hidden_states.to(dtype_orig)

        hidden_states = hidden_states.permute(0, 3, 1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.view(batch_size, seq_len, 1).to(hidden_states.dtype)
        return hidden_states


class SanaWMBidirectionalGDNCamAttention(nn.Module):
    """Camera-control counterpart of [`SanaWMBidirectionalGDNAttention`].

    The recurrence is the same bidirectional delta rule, but the queries/keys/values are positionally encoded with the
    UCPE per-ray transforms instead of RoPE, and the rule is reduced to its numerator ("single path") stream. The
    transformed tensors are downscaled back to their pre-UCPE RMS envelope, and the energy the transform still adds is
    discounted from ``beta``.

    This module holds no parameters; [`BidirectionalGDNUCPESinglePathLiteLA.forward`] issues every layer call and
    passes the resulting tensors in.

    Args:
        chunk_size (`int`, defaults to 21): Temporal chunk length of the state scan.
    """

    def __init__(self, chunk_size: int = 21) -> None:
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ray_transforms: tuple,
        beta: torch.Tensor,
        decay: torch.Tensor,
        num_frames: int,
        token_valid_mask: torch.Tensor | None = None,
        beta_valid_mask: torch.Tensor | None = None,
        decay_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the bidirectional single-path delta rule in UCPE ray space.

        Args:
            query (`torch.Tensor`): Camera queries of shape `(B, N, H, D)`, already normalized and passed through the
                kernel.
            key (`torch.Tensor`): Camera keys of shape `(B, N, H, D)`, same preprocessing as `query`.
            value (`torch.Tensor`): Camera values of shape `(B, N, H, D)`.
            ray_transforms (`tuple`): `(P, P_T, P_inv, rotary_emb_cam)` UCPE transforms; `P_T` encodes the queries,
                `P_inv` the keys/values and `P` decodes the output.
            beta (`torch.Tensor`): Per-frame delta-rule gate of shape `(B, H, T, S)`.
            decay (`torch.Tensor`): Per-frame decay gate of shape `(B, H, T)`.
            num_frames (`int`): Number of frames `T` the sequence axis is split into.
            token_valid_mask (`torch.Tensor`, *optional*): Token mask of shape `(B, N)`.
            beta_valid_mask (`torch.Tensor`, *optional*): Frame mask of shape `(B, 1, T, 1)` applied to `beta`.
            decay_valid_mask (`torch.Tensor`, *optional*): Frame mask of shape `(B, 1, T)` applied to `decay`.

        Returns:
            `torch.Tensor`: Raw camera attention output of shape `(B, N, H * D)`, in world space; the caller projects
            it back into the residual stream.
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        spatial_size = seq_len // num_frames
        dtype_orig = value.dtype

        key = key * ((head_dim**-0.5) * (spatial_size**-0.5))

        # Permute to (B, H, D, N) for processing.
        query = query.permute(0, 2, 3, 1).contiguous()
        key = key.permute(0, 2, 3, 1).contiguous()
        value = value.permute(0, 2, 3, 1).contiguous()

        # Measure the safe geometric norm before UCPE applies translations.
        pre_ucpe_key_norm = torch.linalg.vector_norm(key, dim=2, keepdim=True).clamp_min(1e-6)

        # UCPE expects (B, h, N, d); our tensors are (B, h, d, N). Avoid eager contiguous copies before the
        # transforms, and fuse the K/V transform (both use P_inv) into one call, then split back.
        P, P_T, P_inv, rotary_emb_cam = ray_transforms
        query_ucpe = _apply_ucpe_transform(query.transpose(-1, -2), P_T, rotary_emb_cam).transpose(-1, -2).contiguous()
        key_value = torch.cat([key, value], dim=1)
        key_value_ucpe = (
            _apply_ucpe_transform(key_value.transpose(-1, -2), P_inv, rotary_emb_cam).transpose(-1, -2).contiguous()
        )
        key_ucpe, value_ucpe = torch.chunk(key_value_ucpe, chunks=2, dim=1)

        # Downscale the transformed tensors back to their pre-UCPE RMS envelope.
        query_ucpe = _downscale_to_reference_rms(query, query_ucpe)
        key_ucpe = _downscale_to_reference_rms(key, key_ucpe)
        value_ucpe = _downscale_to_reference_rms(value, value_ucpe)

        # Measure the inflated geometric norm after UCPE, for the beta discount below.
        post_ucpe_key_norm = torch.linalg.vector_norm(key_ucpe, dim=2, keepdim=True).clamp_min(1e-6)
        inflation_sq = (post_ucpe_key_norm / pre_ucpe_key_norm) ** 2

        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(batch_size, 1, 1, seq_len)
            value_ucpe = value_ucpe * token_mask_qkv
            query_ucpe = query_ucpe * token_mask_qkv
            key_ucpe = key_ucpe * token_mask_qkv

        # Dynamic beta discounting: scale beta by the UCPE inflation factor.
        frame_inflation_sq = inflation_sq.view(batch_size, num_heads, num_frames, spatial_size).mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_mask = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_mask + (1.0 - decay_mask)

        # Forward pass (inclusive: 1..t). Force FP32 to preserve recurrent stability.
        out_fwd = torch_chunk_cam_single_path_delta_rule(
            query_ucpe.float(),
            key_ucpe.float(),
            value_ucpe.float(),
            beta.float(),
            decay.float(),
            chunk_size=self.chunk_size,
        )

        # Backward pass (exclusive: t+1..T).
        def to_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, num_heads, head_dim, num_frames, spatial_size).permute(0, 1, 3, 2, 4)

        def from_time_structure(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, seq_len)

        query_bwd = torch.flip(to_time_structure(query_ucpe), dims=[2])
        key_bwd = flip_and_shift(to_time_structure(key_ucpe), dim=2, shift_val=0.0)
        value_bwd = flip_and_shift(to_time_structure(value_ucpe), dim=2, shift_val=0.0)
        beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
        decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

        out_bwd_flipped = torch_chunk_cam_single_path_delta_rule(
            from_time_structure(query_bwd).float(),
            from_time_structure(key_bwd).float(),
            from_time_structure(value_bwd).float(),
            beta_bwd.float(),
            decay_bwd.float(),
            chunk_size=self.chunk_size,
        )
        out_bwd = torch.flip(
            out_bwd_flipped.view(batch_size, num_heads, head_dim, num_frames, spatial_size),
            dims=[3],
        ).reshape(batch_size, num_heads, head_dim, seq_len)

        hidden_states = out_fwd + out_bwd

        if dtype_orig != torch.float32:
            hidden_states = hidden_states.to(dtype_orig)
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.view(batch_size, 1, 1, seq_len).to(hidden_states.dtype)

        # Decode back from ray space to world space.
        hidden_states = (
            _apply_ucpe_transform(hidden_states.transpose(-1, -2), P, rotary_emb_cam, inverse_rope=True)
            .transpose(-1, -2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_size, num_heads * head_dim, seq_len).permute(0, 2, 1)
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.view(batch_size, seq_len, 1).to(hidden_states.dtype)
        return hidden_states


class BidirectionalGDNUCPESinglePathLiteLA(nn.Module):
    """Bidirectional Gated-Delta-Net attention with a UCPE camera-control branch.

    This is the attention block used by every non-softmax layer of the released SANA-WM checkpoint. Two branches run
    over the same tokens and are summed before a single shared output gate + projection:

    - **Main branch** ([`SanaWMBidirectionalGDNAttention`]) -- bidirectional linear attention with a gated delta rule
      over the temporal axis. A ReLU kernel is applied to Q/K, RoPE is applied to the numerator stream only, and the
      denominator (Z) stream keeps unrotated Q/K so mass is conserved. The gates (``beta`` / ``decay``) are computed
      per frame and shared spatially, while the states are maintained per pixel.
    - **Camera branch** ([`SanaWMBidirectionalGDNCamAttention`]) -- the same bidirectional recurrence, but positionally
      encoded with UCPE per-ray transforms instead of RoPE and reduced to a numerator-only ("single path") delta rule.
      The transformed camera tensors are downscaled back to their pre-UCPE RMS envelope before entering the recurrence.

    Both branch modules are parameter-free: every layer call happens in this class's `forward`, which owns the
    projections, short convolutions, norms and gates the two branches consume.

    Camera-specific parameters: ``q_proj_cam``, ``k_proj_cam``, ``v_proj_cam``, ``out_proj_cam``, ``q_norm_cam``,
    ``k_norm_cam`` and ``conv_k_cam``. The GDN gates (``beta_proj`` / ``gate_proj`` / ``dt_bias`` / ``A_log`` /
    ``recall_gate``), the output gate and the output projection are shared by both branches.

    Args:
        in_dim (`int`): Input channels.
        out_dim (`int`): Output channels.
        cam_dim (`int`): Camera-branch width; must equal `in_dim` so the shared parameters line up.
        cam_heads (`int`): Camera-branch heads; must equal `heads` and divide `cam_dim` into multiples of 4.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`): Latent patch size, used to map camera
            intrinsics onto the token grid.
        heads (`int`, *optional*): Number of attention heads; derived from `out_dim // dim * heads_ratio` when `None`.
        heads_ratio (`float`, defaults to 1.0): Head-count multiplier used when `heads` is `None`.
        dim (`int`, defaults to 32): Head dimension used when `heads` is `None`.
        eps (`float`, defaults to 1e-15): Denominator epsilon of the linear-attention normalization.
        use_bias (`bool`, defaults to `False`): Whether the fused QKV projection has a bias.
        qk_norm (`bool`, defaults to `False`): Apply RMSNorm to Q/K.
        norm_eps (`float`, defaults to 1e-5): Epsilon of the Q/K RMSNorm.
        use_output_gate (`bool`, defaults to `True`): Apply the shared silu output gate.
        chunk_gdn_chunk_size (`int`, defaults to 21): Temporal chunk length of the state scan.
        conv_kernel_size (`int`, defaults to 4): Temporal short-convolution width; `0` disables the convolutions.
        k_conv_only (`bool`, defaults to `True`): Apply the short convolution to K only.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        cam_dim: int,
        cam_heads: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
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

        self.kernel_func = nn.ReLU(inplace=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.in_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.in_dim, eps=norm_eps)
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

        # Short convolutions (depthwise Conv1d along T).
        self.conv_kernel_size = conv_kernel_size
        if conv_kernel_size > 0:
            self.conv_k = SanaWMTemporalShortConvolution(
                hidden_size=out_dim,
                kernel_size=conv_kernel_size,
            )
            if k_conv_only:
                self.conv_q = None
                self.conv_v = None
            else:
                self.conv_q = SanaWMTemporalShortConvolution(
                    hidden_size=out_dim,
                    kernel_size=conv_kernel_size,
                )
                self.conv_v = SanaWMTemporalShortConvolution(
                    hidden_size=out_dim,
                    kernel_size=conv_kernel_size,
                )
        else:
            self.conv_q = None
            self.conv_k = None
            self.conv_v = None

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

        # Short convolutions for the camera branch (matching the main branch).
        if self.conv_kernel_size > 0:
            self.conv_k_cam = SanaWMTemporalShortConvolution(
                hidden_size=cam_dim,
                kernel_size=self.conv_kernel_size,
            )
            if self.k_conv_only:
                self.conv_q_cam = None
                self.conv_v_cam = None
            else:
                self.conv_q_cam = SanaWMTemporalShortConvolution(
                    hidden_size=cam_dim,
                    kernel_size=self.conv_kernel_size,
                )
                self.conv_v_cam = SanaWMTemporalShortConvolution(
                    hidden_size=cam_dim,
                    kernel_size=self.conv_kernel_size,
                )
        else:
            self.conv_q_cam = None
            self.conv_k_cam = None
            self.conv_v_cam = None

        # Branch compute modules. They own no parameters -- so the checkpoint layout is untouched -- and only carry
        # the recurrence configuration; `forward` runs every layer call and hands them the resulting tensors.
        self.attn = SanaWMBidirectionalGDNAttention(eps=eps, chunk_size=chunk_gdn_chunk_size)
        self.cam_attn = SanaWMBidirectionalGDNCamAttention(chunk_size=chunk_gdn_chunk_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        *,
        frame_valid_mask: torch.Tensor | None = None,
        ucpe_ray_transforms: tuple | None = None,
    ) -> torch.Tensor:
        """Dual-branch forward: bidirectional GDN main branch + UCPE camera branch.

        Flow:
            1. attn_output = GDN attention (no gate/proj)
            2. cam_output = GDN+UCPE attention (no gate/proj)
            3. combined = attn_output + out_proj_cam(cam_output) [zero at init]
            4. output = proj(output_gate(combined)) [shared, once]

        Args:
            x: Input tensor of shape ``(B, N, C)``.
            mask: Unused attention mask (kept for API compatibility).
            HW: Tuple of ``(T, H, W)`` describing the token layout.
            rotary_emb: Optional rotary embeddings for q/k.
            block_mask: Unused block mask (kept for API compatibility).
            camera_conditions: Raw ``(B, T, 20)`` camera conditions enabling the camera branch.
            chunk_size: Unused chunk length (kept for API compatibility).
            frame_valid_mask: Optional per-frame validity mask used to zero out padded frames, shaped
                ``(B, 1, T, 1, 1)``, ``(B, 1, T)`` or ``(B, T)``.
            ucpe_ray_transforms: Optional pre-computed UCPE transforms shared across blocks.

        Returns:
            Tensor of shape ``(B, N, C)`` after attention and projection.
        """
        del mask, block_mask, chunk_size

        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for GDN attention.")

        batch_size, seq_len, channels = x.shape
        num_frames, height, width = HW
        spatial_size = height * width

        token_valid_mask, beta_valid_mask, decay_valid_mask = _prepare_frame_valid_masks(
            frame_valid_mask,
            batch_size=batch_size,
            num_frames=num_frames,
            spatial_size=spatial_size,
            device=x.device,
            dtype=x.dtype,
        )

        # Per-frame beta / decay gates, computed once from the unmasked input and shared by both branches. `beta` is
        # broadcast over the spatial axis; `decay` is a per-frame scalar per head.
        beta = (
            self.beta_proj(x).sigmoid().reshape(batch_size, num_frames, spatial_size, self.heads).permute(0, 3, 1, 2)
        )
        frame_hidden_states = x.reshape(batch_size, num_frames, spatial_size, channels).mean(dim=2)
        gate = self.gate_proj(frame_hidden_states).float()
        dt = self.dt_bias.float().view(1, 1, -1)
        decay_rate = self.A_log.float().exp().view(1, 1, -1)
        decay = (-decay_rate * F.softplus(gate + dt)).exp().transpose(1, 2)

        # ---- Main branch: fused QKV -> short conv -> Q/K norm -> ReLU kernel -> bidirectional GDN ----
        hidden_states = x
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.view(batch_size, seq_len, 1)

        query, key, value = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.heads, self.dim).unbind(2)
        if token_valid_mask is not None:
            token_mask = token_valid_mask.view(batch_size, seq_len, 1, 1)
            query = query * token_mask
            key = key * token_mask
            value = value * token_mask

        # Short convolution along T (before norm / kernel activation).
        if self.conv_q is not None:
            query = self.conv_q(query.reshape(batch_size, seq_len, channels), num_frames).reshape(
                batch_size, seq_len, self.heads, self.dim
            )
        if self.conv_k is not None:
            key = self.conv_k(key.reshape(batch_size, seq_len, channels), num_frames).reshape(
                batch_size, seq_len, self.heads, self.dim
            )
        if self.conv_v is not None:
            value = self.conv_v(value.reshape(batch_size, seq_len, channels), num_frames).reshape(
                batch_size, seq_len, self.heads, self.dim
            )

        # Q/K norm runs on the flattened channels (B, N, C), then the tensors go back to (B, N, H, D).
        query = self.q_norm(query.reshape(batch_size, seq_len, channels)).reshape(
            batch_size, seq_len, self.heads, self.dim
        )
        key = self.k_norm(key.reshape(batch_size, seq_len, channels)).reshape(
            batch_size, seq_len, self.heads, self.dim
        )
        query = self.kernel_func(query)
        key = self.kernel_func(key)

        attn_output = self.attn(
            query,
            key,
            value,
            beta,
            decay,
            self.recall_gate,
            num_frames,
            rotary_emb=rotary_emb,
            token_valid_mask=token_valid_mask,
            beta_valid_mask=beta_valid_mask,
            decay_valid_mask=decay_valid_mask,
        )

        # ---- Camera branch: same pipeline on the camera projections, positionally encoded with UCPE ----
        if camera_conditions is not None:
            cam_hidden_states = x
            if token_valid_mask is not None:
                cam_hidden_states = cam_hidden_states * token_valid_mask.view(batch_size, seq_len, 1)

            query_cam = self.q_proj_cam(cam_hidden_states)
            key_cam = self.k_proj_cam(cam_hidden_states)
            value_cam = self.v_proj_cam(cam_hidden_states)
            if token_valid_mask is not None:
                token_mask = token_valid_mask.view(batch_size, seq_len, 1)
                query_cam = query_cam * token_mask
                key_cam = key_cam * token_mask
                value_cam = value_cam * token_mask

            if self.conv_q_cam is not None:
                query_cam = self.conv_q_cam(query_cam, num_frames)
            if self.conv_k_cam is not None:
                key_cam = self.conv_k_cam(key_cam, num_frames)
            if self.conv_v_cam is not None:
                value_cam = self.conv_v_cam(value_cam, num_frames)

            query_cam = self.q_norm_cam(query_cam).reshape(batch_size, seq_len, self.cam_heads, self.cam_head_dim)
            key_cam = self.k_norm_cam(key_cam).reshape(batch_size, seq_len, self.cam_heads, self.cam_head_dim)
            value_cam = value_cam.reshape(batch_size, seq_len, self.cam_heads, self.cam_head_dim)
            query_cam = self.kernel_func(query_cam)
            key_cam = self.kernel_func(key_cam)

            # Reuse the model-level cache when available, to avoid recomputing the ray transforms per block.
            ray_transforms = ucpe_ray_transforms
            if ray_transforms is None:
                ray_transforms = _prepare_ucpe_ray_transforms(
                    head_dim=self.cam_head_dim,
                    camera_conditions=camera_conditions,
                    HW=HW,
                    patch_size=self.patch_size,
                    rotary_emb=rotary_emb,
                )

            cam_output = self.cam_attn(
                query_cam,
                key_cam,
                value_cam,
                ray_transforms,
                beta,
                decay,
                num_frames,
                token_valid_mask=token_valid_mask,
                beta_valid_mask=beta_valid_mask,
                decay_valid_mask=decay_valid_mask,
            )
            attn_output = attn_output + self.out_proj_cam(cam_output)

        # Shared output gate + projection, applied once over both branches.
        if self.use_output_gate and self.output_gate is not None:
            attn_output = attn_output * F.silu(self.output_gate(x).to(torch.float32))
        return self.proj(attn_output.to(x.dtype))


class SanaWMSoftmaxAttention(nn.Module):
    """Softmax counterpart of [`SanaWMBidirectionalGDNAttention`].

    Replaces the bidirectional recurrence with a full (non-causal) `F.scaled_dot_product_attention` over the whole
    token sequence. RoPE is applied to the queries and keys, and no linear-attention kernel or key scaling is needed
    (softmax attention brings its own ``1 / sqrt(d_k)``).

    This module holds no parameters; [`_SoftmaxUCPESinglePathLiteLA.forward`] issues every layer call and passes the
    resulting tensors in.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run softmax attention over the token sequence.

        Args:
            query (`torch.Tensor`): Queries of shape `(B, N, H, D)`, already normalized.
            key (`torch.Tensor`): Keys of shape `(B, N, H, D)`, already normalized.
            value (`torch.Tensor`): Values of shape `(B, N, H, D)`.
            rotary_emb (`torch.Tensor`, *optional*): Rotary embeddings applied to `query` and `key`.
            token_valid_mask (`torch.Tensor`, *optional*): Token mask of shape `(B, N)`.

        Returns:
            `torch.Tensor`: Raw attention output of shape `(B, N, H * D)`; the shared output gate and projection are
            applied by the caller.
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        dtype_orig = value.dtype

        # `_apply_rotary_emb` works on (B, H, D, N) features.
        if rotary_emb is not None:
            query = _apply_rotary_emb(query.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)
            key = _apply_rotary_emb(key.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)

        if token_valid_mask is not None:
            token_mask = token_valid_mask.view(batch_size, seq_len, 1, 1)
            query = query * token_mask
            key = key * token_mask
            value = value * token_mask

        query = query.transpose(1, 2)  # (B, H, N, D)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # SDPA / FlashAttention only supports bf16/fp16; fp32 falls back to the math backend.
        if query.dtype == torch.float32:
            query, key, value = query.bfloat16(), key.bfloat16(), value.bfloat16()

        hidden_states = F.scaled_dot_product_attention(query, key, value)
        return hidden_states.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim).to(dtype_orig)


class SanaWMSoftmaxCamAttention(nn.Module):
    """Softmax counterpart of [`SanaWMBidirectionalGDNCamAttention`].

    Keeps the UCPE per-ray encode/decode of the camera branch but replaces the single-path delta rule with a full
    (non-causal) `F.scaled_dot_product_attention`. Padded frames are masked with an additive logit bias on the keys
    instead of being dropped from the recurrence, and no linear-attention kernel, key scaling or gate is involved.

    This module holds no parameters; [`_SoftmaxUCPESinglePathLiteLA.forward`] issues every layer call and passes the
    resulting tensors in.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ray_transforms: tuple,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run softmax attention in UCPE ray space.

        Args:
            query (`torch.Tensor`): Camera queries of shape `(B, N, H, D)`, already normalized.
            key (`torch.Tensor`): Camera keys of shape `(B, N, H, D)`, already normalized.
            value (`torch.Tensor`): Camera values of shape `(B, N, H, D)`.
            ray_transforms (`tuple`): `(P, P_T, P_inv, rotary_emb_cam)` UCPE transforms; `P_T` encodes the queries,
                `P_inv` the keys/values and `P` decodes the output.
            token_valid_mask (`torch.Tensor`, *optional*): Token mask of shape `(B, N)`.

        Returns:
            `torch.Tensor`: Raw camera attention output of shape `(B, N, H * D)`, in world space; the caller projects
            it back into the residual stream.
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        dtype_orig = value.dtype

        # Permute to (B, H, D, N), matching the GDN camera branch.
        query = query.permute(0, 2, 3, 1).contiguous()
        key = key.permute(0, 2, 3, 1).contiguous()
        value = value.permute(0, 2, 3, 1).contiguous()

        # UCPE expects (B, h, N, d); our tensors are (B, h, d, N). Avoid eager contiguous copies before the
        # transforms, and fuse the K/V transform (both use P_inv) into one call, then split back.
        P, P_T, P_inv, rotary_emb_cam = ray_transforms
        query_ucpe = _apply_ucpe_transform(query.transpose(-1, -2), P_T, rotary_emb_cam).transpose(-1, -2).contiguous()
        key_value = torch.cat([key, value], dim=1)
        key_value_ucpe = (
            _apply_ucpe_transform(key_value.transpose(-1, -2), P_inv, rotary_emb_cam).transpose(-1, -2).contiguous()
        )
        key_ucpe, value_ucpe = torch.chunk(key_value_ucpe, chunks=2, dim=1)

        # Downscale the transformed tensors back to their pre-UCPE RMS envelope.
        query_ucpe = _downscale_to_reference_rms(query, query_ucpe)
        key_ucpe = _downscale_to_reference_rms(key, key_ucpe)
        value_ucpe = _downscale_to_reference_rms(value, value_ucpe)

        if token_valid_mask is not None:
            token_mask_qkv = token_valid_mask.view(batch_size, 1, 1, seq_len)
            query_ucpe = query_ucpe * token_mask_qkv
            value_ucpe = value_ucpe * token_mask_qkv

        query_sdpa = query_ucpe.transpose(-1, -2)
        key_sdpa = key_ucpe.transpose(-1, -2)
        value_sdpa = value_ucpe.transpose(-1, -2)

        query_sdpa, key_sdpa, value_sdpa = query_sdpa.float(), key_sdpa.float(), value_sdpa.float()
        # SDPA / FlashAttention only supports bf16/fp16; fp32 falls back to math backend.
        if query_sdpa.dtype == torch.float32:
            query_sdpa, key_sdpa, value_sdpa = query_sdpa.bfloat16(), key_sdpa.bfloat16(), value_sdpa.bfloat16()

        # Invalid frames are masked out of the keys with an additive bias rather than being zeroed.
        invalid_kv_logit_bias = None
        if token_valid_mask is not None and not bool(token_valid_mask.all()):
            invalid_kv_logit_bias = torch.where(
                token_valid_mask.bool().view(batch_size, 1, 1, -1),
                torch.zeros((), dtype=query_sdpa.dtype, device=query_sdpa.device),
                torch.full((), -1e9, dtype=query_sdpa.dtype, device=query_sdpa.device),
            )

        # FlashAttention-2 only supports head_dim in {32, 64, 128, 256}.
        need_pad = head_dim not in (32, 64, 128, 256) and head_dim < 256
        if need_pad:
            pad_size = (128 if head_dim <= 128 else 256) - head_dim
            query_sdpa = F.pad(query_sdpa, (0, pad_size))
            key_sdpa = F.pad(key_sdpa, (0, pad_size))
            value_sdpa = F.pad(value_sdpa, (0, pad_size))
        hidden_states = F.scaled_dot_product_attention(
            query_sdpa, key_sdpa, value_sdpa, attn_mask=invalid_kv_logit_bias
        )
        if need_pad:
            hidden_states = hidden_states[..., :head_dim]

        hidden_states = hidden_states.transpose(-1, -2)
        if hidden_states.dtype != dtype_orig:
            hidden_states = hidden_states.to(dtype_orig)
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.view(batch_size, 1, 1, seq_len).to(hidden_states.dtype)

        # Decode back from ray space to world space.
        hidden_states = (
            _apply_ucpe_transform(hidden_states.transpose(-1, -2), P, rotary_emb_cam, inverse_rope=True)
            .transpose(-1, -2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_size, num_heads * head_dim, seq_len).permute(0, 2, 1)
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.view(batch_size, seq_len, 1).to(hidden_states.dtype)
        return hidden_states


class _SoftmaxUCPESinglePathLiteLA(nn.Module):
    """Softmax counterpart of [`BidirectionalGDNUCPESinglePathLiteLA`].

    The released checkpoint uses this block for every ``softmax_every_n``-th layer. It keeps the exact parameter layout
    of [`BidirectionalGDNUCPESinglePathLiteLA`] -- so both variants load from the same checkpoint -- but replaces the
    main-branch recurrence with [`SanaWMSoftmaxAttention`] and the camera-branch recurrence with
    [`SanaWMSoftmaxCamAttention`], both a full (non-causal) ``F.scaled_dot_product_attention``. Short convolutions are
    never built for this variant, and the GDN-only parameters (``beta_proj`` / ``gate_proj`` / ``dt_bias`` / ``A_log``
    / ``recall_gate``) exist only so the shared checkpoint loads: they are created in the same order, under the same
    names, and are unused by the forward.

    Both branch modules are parameter-free: every layer call happens in this class's `forward`, which owns the
    projections and norms the two branches consume.

    Args:
        in_dim (`int`): Input channels.
        out_dim (`int`): Output channels.
        cam_dim (`int`): Camera-branch width; must equal `in_dim` so the shared parameters line up.
        cam_heads (`int`): Camera-branch heads; must equal `heads` and divide `cam_dim` into multiples of 4.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`): Latent patch size, used to map camera
            intrinsics onto the token grid.
        heads (`int`, *optional*): Number of attention heads; derived from `out_dim // dim * heads_ratio` when `None`.
        heads_ratio (`float`, defaults to 1.0): Head-count multiplier used when `heads` is `None`.
        dim (`int`, defaults to 32): Head dimension used when `heads` is `None`.
        use_bias (`bool`, defaults to `False`): Whether the fused QKV projection has a bias.
        qk_norm (`bool`, defaults to `False`): Apply RMSNorm to Q/K.
        norm_eps (`float`, defaults to 1e-5): Epsilon of the Q/K RMSNorm.
        use_output_gate (`bool`, defaults to `True`): Apply the shared silu output gate.
        kwargs: The GDN-only options (`eps`, `chunk_gdn_chunk_size`, `conv_kernel_size`, `k_conv_only`) are accepted
            and ignored, so both attention variants can be built from the same block keywords.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        cam_dim: int,
        cam_heads: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        use_output_gate: bool = True,
        **kwargs: object,
    ) -> None:
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__()

        # Fused QKV projection and output projection (the `q_norm` / `k_norm`
        # attributes are set further down, depending on `qk_norm`).
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=use_bias)
        self.proj = nn.Linear(in_dim, in_dim)

        self.heads = heads
        self.dim = out_dim // heads

        if qk_norm:
            self.q_norm = RMSNorm(in_dim, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # GDN-only gates. Softmax attention never reads them, but they are part of the
        # shared checkpoint, so they are built here in the checkpoint's order.
        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)

        A = torch.zeros(heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min),
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

        # Branch compute modules. They own no parameters -- so the checkpoint layout is untouched -- and `forward`
        # runs every layer call and hands them the resulting tensors.
        self.attn = SanaWMSoftmaxAttention()
        self.cam_attn = SanaWMSoftmaxCamAttention()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        *,
        frame_valid_mask: torch.Tensor | None = None,
        ucpe_ray_transforms: tuple | None = None,
    ) -> torch.Tensor:
        """Dual-branch forward: softmax main branch + softmax UCPE camera branch.

        Args:
            x: Input tensor of shape ``(B, N, C)``.
            mask: Unused attention mask (kept for API compatibility).
            HW: Tuple of ``(T, H, W)`` describing the token layout.
            rotary_emb: Optional rotary embeddings for q/k.
            block_mask: Unused block mask (kept for API compatibility).
            camera_conditions: Raw ``(B, T, 20)`` camera conditions enabling the camera branch.
            chunk_size: Unused chunk length (kept for API compatibility).
            frame_valid_mask: Optional per-frame validity mask used to zero out padded frames, shaped
                ``(B, 1, T, 1, 1)``, ``(B, 1, T)`` or ``(B, T)``.
            ucpe_ray_transforms: Optional pre-computed UCPE transforms shared across blocks.

        Returns:
            Tensor of shape ``(B, N, C)`` after attention and projection.
        """
        del mask, block_mask, chunk_size

        if HW is None:
            raise ValueError("HW (T, H, W) must be provided for softmax attention.")

        batch_size, seq_len, channels = x.shape
        num_frames, height, width = HW
        spatial_size = height * width

        token_valid_mask, _, _ = _prepare_frame_valid_masks(
            frame_valid_mask,
            batch_size=batch_size,
            num_frames=num_frames,
            spatial_size=spatial_size,
            device=x.device,
            dtype=x.dtype,
        )

        # ---- Main branch: fused QKV -> Q/K norm -> softmax attention ----
        hidden_states = x
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.view(batch_size, seq_len, 1)

        query, key, value = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.heads, self.dim).unbind(2)
        if token_valid_mask is not None:
            token_mask = token_valid_mask.view(batch_size, seq_len, 1, 1)
            query = query * token_mask
            key = key * token_mask
            value = value * token_mask

        # Q/K norm runs on the flattened channels (B, N, C), then the tensors go back to (B, N, H, D).
        query = self.q_norm(query.reshape(batch_size, seq_len, channels)).reshape(
            batch_size, seq_len, self.heads, self.dim
        )
        key = self.k_norm(key.reshape(batch_size, seq_len, channels)).reshape(
            batch_size, seq_len, self.heads, self.dim
        )

        attn_output = self.attn(query, key, value, rotary_emb=rotary_emb, token_valid_mask=token_valid_mask)

        # ---- Camera branch: same pipeline on the camera projections, positionally encoded with UCPE ----
        if camera_conditions is not None:
            cam_hidden_states = x
            if token_valid_mask is not None:
                cam_hidden_states = cam_hidden_states * token_valid_mask.view(batch_size, seq_len, 1)

            query_cam = self.q_proj_cam(cam_hidden_states)
            key_cam = self.k_proj_cam(cam_hidden_states)
            value_cam = self.v_proj_cam(cam_hidden_states)
            if token_valid_mask is not None:
                token_mask = token_valid_mask.view(batch_size, seq_len, 1)
                query_cam = query_cam * token_mask
                key_cam = key_cam * token_mask
                value_cam = value_cam * token_mask

            query_cam = self.q_norm_cam(query_cam).reshape(batch_size, seq_len, self.cam_heads, self.cam_head_dim)
            key_cam = self.k_norm_cam(key_cam).reshape(batch_size, seq_len, self.cam_heads, self.cam_head_dim)
            value_cam = value_cam.reshape(batch_size, seq_len, self.cam_heads, self.cam_head_dim)

            # Reuse the model-level cache when available, to avoid recomputing the ray transforms per block.
            ray_transforms = ucpe_ray_transforms
            if ray_transforms is None:
                ray_transforms = _prepare_ucpe_ray_transforms(
                    head_dim=self.cam_head_dim,
                    camera_conditions=camera_conditions,
                    HW=HW,
                    patch_size=self.patch_size,
                    rotary_emb=rotary_emb,
                )

            cam_output = self.cam_attn(
                query_cam, key_cam, value_cam, ray_transforms, token_valid_mask=token_valid_mask
            )
            attn_output = attn_output + self.out_proj_cam(cam_output)

        # Shared output gate + projection, applied once over both branches.
        if self.use_output_gate and self.output_gate is not None:
            attn_output = attn_output * F.silu(self.output_gate(x).to(torch.float32))
        return self.proj(attn_output.to(x.dtype))


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
        attn_cls,
        mlp_ratio=4.0,
        qk_norm=False,
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        t_kernel_size=3,
        patch_size=(1, 2, 2),
        cam_attn_compress=2,
        chunk_size=10,
        use_chunk_plucker_post_attn=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size

        if use_chunk_plucker_post_attn:
            self.plucker_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Camera-conditioned (UCPE) attention: either the GDN or the softmax variant.
        self_num_heads = hidden_size // linear_head_dim
        self.attn = attn_cls(
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

    def forward(
        self,
        x,
        y,
        t,
        mask=None,
        THW=None,
        rotary_emb=None,
        block_mask=None,
        *,
        camera_conditions=None,
        ucpe_ray_transforms=None,
        plucker_emb=None,
        frame_valid_mask=None,
        chunk_size=None,
    ):
        """Run one adaLN-Zero block: self-attention -> cross-attention -> FFN.

        Args:
            x: ``(B, N, C)`` token sequence.
            y: ``(B, 1, L, C)`` text embeddings for cross-attention.
            t: ``(B, 1, T, 6 * C)`` adaLN modulation input.
            mask: Text padding mask for cross-attention.
            THW: ``(T, H, W)`` token layout.
            rotary_emb: Rotary embeddings for the self-attention branch.
            block_mask: Optional block mask forwarded to the attention.
            camera_conditions: Raw camera conditions enabling the camera branch.
            ucpe_ray_transforms: Pre-computed UCPE transforms shared across blocks.
            plucker_emb: Optional post-attention Plucker embedding.
            frame_valid_mask: Optional per-frame validity mask.
            chunk_size: Chunk length override; falls back to ``self.chunk_size``.
        """
        B, N, C = x.shape
        num_frames = t.shape[2]
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
        if chunk_size is None:
            chunk_size = self.chunk_size

        x_norm1 = self.norm1(x).reshape(B, num_frames, -1, C)
        x_msa_in = (x_norm1 * (1 + scale_msa) + shift_msa).reshape(B, N, C)
        if frame_token_mask is not None:
            x_msa_in = x_msa_in * frame_token_mask
        # Camera-conditioned (UCPE) attention: dual-branch (main + camera) forward.
        attn_out = self.attn(
            x_msa_in,
            HW=THW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            camera_conditions=camera_conditions,
            chunk_size=chunk_size,
            frame_valid_mask=frame_valid_mask,
            ucpe_ray_transforms=ucpe_ray_transforms,
        )
        attn_out = attn_out.reshape(B, num_frames, -1, C)
        attn_out = (gate_msa * attn_out).reshape(B, N, C)
        if frame_token_mask is not None:
            attn_out = attn_out * frame_token_mask
        x = x + attn_out
        if frame_token_mask is not None:
            x = x * frame_token_mask

        if plucker_emb is not None and hasattr(self, "plucker_proj"):
            x = x + self.plucker_proj(plucker_emb)

        x = x + self.cross_attn(x, y, mask=mask)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        x_norm2 = self.norm2(x).reshape(B, num_frames, -1, C)
        x_mlp_in = (x_norm2 * (1 + scale_mlp) + shift_mlp).reshape(B, N, C)
        if frame_token_mask is not None:
            x_mlp_in = x_mlp_in * frame_token_mask
        mlp_out = self.mlp(x_mlp_in, HW=THW).reshape(B, num_frames, -1, C)
        mlp_out = (gate_mlp * mlp_out).reshape(B, N, C)
        if frame_token_mask is not None:
            mlp_out = mlp_out * frame_token_mask
        x = x + mlp_out
        if frame_token_mask is not None:
            x = x * frame_token_mask

        return x


class SanaWMTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    SANA-WM 1600M bidirectional camera-controlled DiT.

    A single-class DiT (depth=20, hidden_size=2240, patch_size=(1,1,1), num_heads=20 — i.e. the public
    ``Efficient-Large-Model/SANA-WM_bidirectional`` release). ``save_pretrained`` / ``from_pretrained`` work out of the
    box via :class:`~diffusers.configuration_utils.ConfigMixin`.

    Every block runs a camera-conditioned (UCPE) attention: [`BidirectionalGDNUCPESinglePathLiteLA`], except every
    ``softmax_every_n``-th block which runs its softmax counterpart [`_SoftmaxUCPESinglePathLiteLA`].

    Args:
        in_channels (`int`, defaults to 128): VAE latent channels (LTX-2).
        softmax_every_n (`int`, defaults to 4): Use a softmax attention block every N blocks.
        linear_head_dim (`int`, defaults to 112): GDN head dimension.
        ffn_type (`str`, defaults to ``"GLUMBConvTemp"``): FFN.
        t_kernel_size (`int`, defaults to 3): Temporal conv kernel.
        conv_kernel_size (`int`, defaults to 4): Spatial conv kernel inside attention.
        k_conv_only (`bool`, defaults to True): Apply conv only on K.
        pos_embed_type (`str`, defaults to ``"wan_rope"``): Position embedding.
        qk_norm (`bool`, defaults to True): RMSNorm on Q/K.
        cross_norm (`bool`, defaults to True): RMSNorm on cross-attention K.
        y_norm (`bool`, defaults to True): Apply ``attention_y_norm`` to text embeddings.
        init_cam_from_base (`bool`, defaults to True): Unused; the camera branch is loaded from the checkpoint.
            Kept so released `config.json` files load.
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
        cam_attn_compress: int = 1,
        init_cam_from_base: bool = True,
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
            self.attention_y_norm = RMSNorm(hidden_size, eps=norm_eps)

        # --- Video camera-controlled DiT modules (from SanaMSVideoCamCtrl.__init__) ---
        self.chunk_size = chunk_size
        self.patch_size = patch_size

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.pos_embed_ms = None
        self.pack_latents = pack_latents
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

        if use_pe:
            if pos_embed_type != "wan_rope":
                raise ValueError(f'`pos_embed_type` must be "wan_rope", got {pos_embed_type!r}.')
            self.rope = SanaWMRotaryPosEmbed(
                attention_head_dim=linear_head_dim, patch_size=patch_size, max_seq_len=1024, fhw_dim=rope_fhw_dim
            )

        # Every ``softmax_every_n``-th block swaps the GDN recurrence for softmax attention; both variants share the
        # same parameter layout.
        self.softmax_every_n = softmax_every_n
        attn_cls_list = []
        for i in range(depth):
            if softmax_every_n > 0 and (i + 1) % softmax_every_n == 0:
                attn_cls = _SoftmaxUCPESinglePathLiteLA
            else:
                attn_cls = BidirectionalGDNUCPESinglePathLiteLA
            attn_cls_list.append(attn_cls)

        self.blocks = nn.ModuleList(
            [
                SanaVideoMSCamCtrlBlock(
                    hidden_size,
                    num_heads,
                    attn_cls=attn_cls_list[i],
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_norm=cross_norm,
                    t_kernel_size=t_kernel_size,
                    patch_size=patch_size,
                    cam_attn_compress=self.cam_attn_compress,
                    chunk_size=chunk_size,
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

        if not (height % 2 == 0 and width % 2 == 0):
            raise ValueError(f"Latent height and width must be divisible by 2, got {height}x{width}.")
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
        data_info: Optional[dict] = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_plucker: torch.Tensor | None = None,
        cam_pos_embeds: Optional[dict] = None,
        pos_embeds: torch.Tensor | None = None,
        raymats: torch.Tensor | None = None,
        frame_valid_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
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
            data_info: Extra conditioning; ``data_info["image_vae_embeds"]`` is
                concatenated to the latents along the channel axis when present.
            camera_conditions: ``(B, T, 20)`` raw camera conditions driving the
                camera-control (UCPE) branch.
            chunk_plucker: Plucker ray embeddings ``(B, C, T, H, W)``, consumed when
                the model is configured with ``use_chunk_plucker_input`` / ``use_chunk_plucker_post_attn``.
            cam_pos_embeds: Optional pre-computed camera positional embeddings
                (``"absmap"`` / ``"P"`` entries) reused instead of recomputing them.
            pos_embeds: Optional pre-computed rotary position embeddings; when ``None``
                they are built from the latent shape.
            raymats: Optional pre-computed UCPE ray matrices (used only when
                ``cam_pos_embeds`` does not carry ``"P"``).
            frame_valid_mask: Optional per-frame validity mask, shaped
                ``(B, 1, T, 1, 1)``, ``(B, 1, T)`` or ``(B, T)``.
            chunk_size: Chunk length override forwarded to the blocks; falls back
                to each block's configured ``chunk_size``.

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
        post_patch_num_frames, post_patch_height, post_patch_width = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )

        if data_info is None:
            data_info = {}
        if data_info.get("image_vae_embeds", None) is not None:
            x = torch.cat([x, data_info["image_vae_embeds"].to(self.dtype)], dim=1)
        cam_embeds = camera_conditions
        if self.pack_latents:
            x = self._pack_latents(x, bs, self.in_channels, post_patch_height, post_patch_width, post_patch_num_frames)
            if cam_embeds is not None:
                cam_embeds = cam_embeds.to(self.dtype)

            post_patch_height = post_patch_height // 2
            post_patch_width = post_patch_width // 2

        if self.x_embedder.patch_size != self.x_embedder.kernel_size and self.x_embedder.kernel_size == (1, 2, 2):
            x = F.pad(x, (0, 1, 0, 1, 0, 0))
            if cam_embeds is not None:
                cam_embeds = F.pad(cam_embeds, (0, 1, 0, 1, 0, 0))

        x = self.x_embedder(x)
        if cam_embeds is not None:
            # Both attention variants are UCPE-style: build raymats + 3-channel absmap
            # (up_map + lat_map) from the raw (B,F,20) camera conditions.
            raw_cam_conditions = cam_embeds
            if cam_pos_embeds is not None and "absmap" in cam_pos_embeds:
                cam_embeds = cam_pos_embeds["absmap"]
                if "P" in cam_pos_embeds:
                    raymats = cam_pos_embeds["P"]
            else:
                raymats, cam_embeds = _process_camera_conditions_ucpe(
                    raw_cam_conditions,
                    bs,
                    (post_patch_num_frames, post_patch_height, post_patch_width),
                    self.patch_size,
                )
                cam_embeds = cam_embeds.permute(0, 4, 1, 2, 3).to(self.dtype)
            if not (self.use_chunk_plucker_input or self.use_chunk_plucker_post_attn):
                cam_embeds = self.raymap_embedder(cam_embeds)
                x = x + cam_embeds
                camera_conditions = raw_cam_conditions

        post_attn_plucker_emb = None
        if self.use_chunk_plucker_input and chunk_plucker is not None:
            plucker_input = chunk_plucker.to(self.dtype)
            plucker_emb = self.plucker_embedder(plucker_input)
            x = x + plucker_emb

        if self.use_chunk_plucker_post_attn and chunk_plucker is not None:
            plucker_input = chunk_plucker.to(self.dtype)
            post_attn_plucker_emb = self.plucker_embedder(plucker_input)

        image_pos_embed = pos_embeds
        if self.use_pe and image_pos_embed is None:
            image_pos_embed = self.rope((post_patch_num_frames, post_patch_height, post_patch_width))
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

        ucpe_ray_transforms = None
        if camera_conditions is not None:
            # Pre-compute the UCPE ray matrices once and share them across blocks
            # (both attention variants are UCPE-style).
            head_dim = self.linear_head_dim

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

            ucpe_ray_transforms = _prepare_ucpe_ray_transforms(
                head_dim=head_dim,
                camera_conditions=camera_conditions,
                HW=(post_patch_num_frames, post_patch_height, post_patch_width),
                patch_size=self.patch_size,
                rotary_emb=image_pos_embed,
                raymats=raymats,
                cam_pos_embeds=cam_pos_embeds,
            )

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                y,
                t0,
                y_lens,
                (post_patch_num_frames, post_patch_height, post_patch_width),
                image_pos_embed,
                block_mask=block_mask if i > 1 else None,
                camera_conditions=camera_conditions,
                ucpe_ray_transforms=ucpe_ray_transforms,
                plucker_emb=post_attn_plucker_emb,
                frame_valid_mask=frame_valid_mask,
                chunk_size=chunk_size,
            )  # (N, T, D)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, post_patch_num_frames, post_patch_height, post_patch_width)  # (N, out_channels, H, W)
        if self.pack_latents:
            x = self._unpack_latents(x, post_patch_height * 2, post_patch_width * 2, post_patch_num_frames)

        return Transformer2DModelOutput(sample=x) if return_dict else (x,)

    def unpatchify(self, x, post_patch_num_frames, post_patch_height, post_patch_width):
        """
        x: (N, T, patch_size**2 * C) imgs: (N, H, W, C)
        """
        c = self.out_channels
        p_f, p_h, p_w = self.x_embedder.patch_size
        if post_patch_num_frames * post_patch_height * post_patch_width != x.shape[1]:
            raise ValueError(
                f"Expected {post_patch_num_frames * post_patch_height * post_patch_width} tokens for a "
                f"({post_patch_num_frames}, {post_patch_height}, {post_patch_width}) latent, but got {x.shape[1]}."
            )

        x = x.reshape(shape=(x.shape[0], post_patch_num_frames, post_patch_height, post_patch_width, p_f, p_h, p_w, c))
        x = torch.einsum("nfhwopqc->ncfohpwq", x)
        imgs = x.reshape(
            shape=(x.shape[0], c, post_patch_num_frames * p_f, post_patch_height * p_h, post_patch_width * p_w)
        )

        return imgs
