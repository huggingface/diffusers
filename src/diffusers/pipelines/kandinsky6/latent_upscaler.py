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

"""Kandinsky 6 SR latent-upscaler bank Diffusers component."""

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin


def merge_batch_time(x: Tensor) -> Tensor:
    """Merge the temporal axis into batch: ``(B, C, T, H, W) -> (B*T, C, H, W)``."""
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def split_batch_time(x: Tensor, b: int, t: int) -> Tensor:
    """Inverse of :func:`merge_batch_time`: ``(B*T, C, H, W) -> (B, C, T, H, W)``."""
    _, c, h, w = x.shape
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


UpsampleMode = str


# Convolution primitives shared by the latent upsampler architectures.


DIMS_2 = 2
DIMS_3 = 3

TemporalPadding = Literal["zeros", "replicate", "causal"]
UpsamplePaddingMode = Literal["reflect", "zeros"]


class TemporalReplicateConv3d(nn.Conv3d):
    """``Conv3d`` that repeats the edge frame along T and zero-pads H/W.

    K-VAE extends the temporal axis by repeating a boundary frame — the encoder
    and decoder both seed their causal padding with a replica of frame 0 — while
    padding H/W with zeros (``padding_mode: zeros`` in the shipped sidecar).
    ``nn.Conv3d`` cannot express that split, because ``padding_mode`` applies to
    every padded dim at once; here T is padded explicitly and H/W is left to the
    convolution.

    Without this, a zero-padded temporal axis makes the first and last latent
    frames see a hole where a neighbour should be — on K-VAE latents that hole
    lands on frame 0, which is already the odd one out (it encodes a single
    pixel frame, while every later latent aggregates four).

    Parameter names and shapes are those of ``nn.Conv3d``, so checkpoints
    trained before this padding change still load.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel as ``(kt, kh, kw)`` or a single int applied to all dims.
        padding: The "same" padding the caller would have passed to ``nn.Conv3d``.
            Its H/W entries go to the convolution; its T entry becomes the width
            of the replicate pad.
        groups: Convolution groups (``in_channels`` for a depthwise conv).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        padding: int | tuple[int, int, int] = 0,
        *,
        groups: int = 1,
    ) -> None:
        """Initialize the convolution with H/W padding only, keeping T for forward."""
        pad_t, pad_h, pad_w = (padding, padding, padding) if isinstance(padding, int) else padding
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=(0, pad_h, pad_w),
            groups=groups,
        )
        self.temporal_pad = pad_t

    def temporal_pad_lr(self) -> tuple[int, int]:
        """Split the temporal pad budget into ``(before, after)`` frame counts."""
        return (self.temporal_pad, self.temporal_pad)

    def forward(self, x: Tensor) -> Tensor:
        """Replicate-pad T, then convolve with zero padding on H and W."""
        before, after = self.temporal_pad_lr()
        if before or after:
            x = functional.pad(x, (0, 0, 0, 0, before, after), mode="replicate")
        return super().forward(x)


class TemporalCausalConv3d(TemporalReplicateConv3d):
    """``TemporalReplicateConv3d`` with K-VAE's causal temporal window.

    K-VAE's ``CausalConv3d`` spends the whole temporal pad budget in front:
    ``kT - 1`` replicas of frame 0 precede the clip and nothing follows it, so
    each 3-tap kernel reads ``(t-2, t-1, t)``. Moving the symmetric budget
    (``pad_t`` per side) to the front reproduces that window exactly, which is
    what lets decoder kernels load without recentring — and keeps every frame's
    output independent of its future, the property a chunked streaming
    inference would rely on.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        padding: int | tuple[int, int, int] = 0,
        *,
        groups: int = 1,
    ) -> None:
        """Validate that the pad budget covers exactly the causal window."""
        super().__init__(in_channels, out_channels, kernel_size, padding, groups=groups)
        kernel_t = self.kernel_size[0]
        if 2 * self.temporal_pad != kernel_t - 1:
            msg = (
                f"a causal window needs the full kT-1 = {kernel_t - 1} pad budget in front, "
                f"but padding supplies 2 * {self.temporal_pad}"
            )
            raise ValueError(msg)

    def temporal_pad_lr(self) -> tuple[int, int]:
        """Put the whole budget before the clip: taps end on the current frame."""
        return (2 * self.temporal_pad, 0)


def make_conv(dims: int, temporal_padding: TemporalPadding) -> type[nn.Module]:
    """Pick the convolution class for a stack of ``dims``-dimensional convs.

    Args:
        dims: 2 for ``Conv2d``, 3 for ``Conv3d``.
        temporal_padding: How ``dims == 3`` convolutions extend T. ``"zeros"``
            keeps the stock convolution; ``"replicate"`` repeats the edge frame;
            ``"causal"`` pads the past only (K-VAE semantics).
            Ignored for ``dims == 2``, which has no temporal axis.

    Returns:
        The convolution class to instantiate.
    """
    if dims == DIMS_2:
        return nn.Conv2d
    if temporal_padding == "replicate":
        return TemporalReplicateConv3d
    if temporal_padding == "causal":
        return TemporalCausalConv3d
    return nn.Conv3d


# Reusable building blocks for latent upsampler architectures.


class StochasticDepth(nn.Module):
    """Drop residual paths without requiring torchvision."""

    def __init__(self, p: float, mode: str = "row") -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"drop probability must be in [0, 1], got {p}")
        if mode not in ("batch", "row"):
            raise ValueError(f"mode must be 'batch' or 'row', got {mode!r}")
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) if self.mode == "row" else (1,) * x.ndim
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(1.0 - self.p)
        return x * noise / (1.0 - self.p)


class RMSNorm(nn.Module):
    """Channel-first Root Mean Square normalization with learnable gamma.

    Args:
        dim: Number of channels.
        dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
    """

    def __init__(self, dim: int, dims: int = 2) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Number of channels.
            dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
        """
        super().__init__()
        broadcastable_dims = (1, 1) if dims == 2 else (1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *broadcastable_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization along the channel dimension."""
        return functional.normalize(x, dim=1) * self.scale * self.gamma


class ModulatedRMSNorm(nn.Module):
    """RMSNorm followed by spatial FiLM modulation conditioned on a side tensor.

    Computes ``RMSNorm(x) * conv_y(zq) + conv_b(zq)`` where ``conv_y`` and
    ``conv_b`` are 1x1 convolutions of ``zq`` (the conditioning tensor, e.g.
    the LQ latent).  ``zq`` is aligned to ``x`` by nearest-neighbor
    interpolation along the spatial dims, so the same ``zq`` can be reused
    across feature pyramids of different resolutions.

    Both convs keep the stock PyTorch initialization, so modulation is active
    from the first step — the same regime as K-VAE's ``CachedSpatialNorm3D``,
    whose ``conv_y`` / ``conv_b`` are plain 1x1 convs with no custom init.

    Args:
        dim: Number of feature channels.
        zq_dim: Number of channels in the conditioning tensor.
        dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
    """

    def __init__(self, dim: int, zq_dim: int, dims: int = 2) -> None:
        """Initialize ModulatedRMSNorm.

        Args:
            dim: Number of feature channels.
            zq_dim: Number of channels in the conditioning tensor.
            dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
        """
        super().__init__()
        conv = nn.Conv2d if dims == 2 else nn.Conv3d
        self.norm = RMSNorm(dim, dims=dims)
        self.conv_y = conv(zq_dim, dim, kernel_size=1)
        self.conv_b = conv(zq_dim, dim, kernel_size=1)

    def forward(self, x: Tensor, zq: Tensor) -> Tensor:
        """Apply RMS norm + FiLM modulation by spatially-aligned ``zq``."""
        if zq.shape[2:] != x.shape[2:]:
            zq = functional.interpolate(zq, size=x.shape[2:], mode="nearest")
        return self.norm(x) * self.conv_y(zq) + self.conv_b(zq)


class LayerScale(nn.Module):
    """Learnable per-channel scaling applied before residual addition.

    Args:
        channels: Number of channels.
        init_value: Initial scale value (small for training stability).
        dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
    """

    def __init__(self, channels: int, init_value: float = 1e-6, dims: int = 2) -> None:
        """Initialize LayerScale.

        Args:
            channels: Number of channels.
            init_value: Initial scale value.
            dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
        """
        super().__init__()
        broadcastable_dims = (1, 1) if dims == 2 else (1, 1, 1)
        self.gamma = nn.Parameter(init_value * torch.ones(channels, *broadcastable_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Scale input by learnable per-channel gamma."""
        return x * self.gamma


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXtV2).

    Aggregates global spatial info per channel and normalizes across channels
    to encourage feature diversity and prevent feature collapse.

    Args:
        channels: Number of channels.
        dims: Convolution dimensionality — 2 for spatial dims ``(2,3)``, 3 for ``(2,3,4)``.
    """

    def __init__(self, channels: int, dims: int = 2) -> None:
        """Initialize GRN.

        Args:
            channels: Number of channels.
            dims: Convolution dimensionality — 2 for spatial dims ``(2,3)``, 3 for ``(2,3,4)``.
        """
        super().__init__()
        self.spatial_dims: tuple[int, ...] = (2, 3) if dims == 2 else (2, 3, 4)
        broadcastable = (1, 1) if dims == 2 else (1, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, *broadcastable))
        self.beta = nn.Parameter(torch.zeros(1, channels, *broadcastable))

    def forward(self, x: Tensor) -> Tensor:
        """Apply global response normalization."""
        gx = torch.norm(x, p=2, dim=self.spatial_dims, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional LayerScale, GRN, and StochasticDepth.

    Standard path: ``RMSNorm → SiLU → Conv3x3 → RMSNorm → SiLU → [GRN →] Conv3x3``.
    Depthwise path: ``DWConv_KxK → RMSNorm → PWConv1x1 → SiLU → [GRN →] PWConv1x1``.
    Skip: identity when ``in_channels == out_channels``, else ``Conv 1x1``.

    When ``zq_dim`` is set, every ``RMSNorm`` is replaced by a
    ``ModulatedRMSNorm`` conditioned on a side tensor ``zq``, and the inner
    layers are stored as named submodules so ``zq`` can be threaded through
    ``forward(x, zq)``.  When ``zq_dim is None``, the legacy ``nn.Sequential``
    layout is preserved bit-identically — the parameter names match older
    checkpoints, so loading them remains backward compatible.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        mid_channels: int | None = None,
        dims: int = 2,
        layer_scale_init: float | None = None,
        *,
        grn: bool = False,
        stochastic_depth_prob: float = 0.0,
        depthwise: bool = False,
        kernel_size: int = 3,
        zq_dim: int | None = None,
        temporal_padding: TemporalPadding = "zeros",
    ) -> None:
        """Initialize residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (defaults to ``in_channels``).
            mid_channels: Internal channel width (defaults to ``out_channels``).
            dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
            layer_scale_init: Initial LayerScale gamma value. ``None`` disables LayerScale.
            grn: Whether to insert Global Response Normalization on mid_channels.
            stochastic_depth_prob: Drop probability for stochastic depth (0 = off).
            depthwise: Use depthwise separable convolutions instead of standard convs.
            kernel_size: Kernel size for the depthwise convolution (only used when ``depthwise=True``).
            zq_dim: When set, replace every ``RMSNorm`` with ``ModulatedRMSNorm(zq_dim)``;
                the block then accepts a conditioning tensor in ``forward(x, zq)``.
            temporal_padding: How ``dims == 3`` convolutions extend T — ``"zeros"``
                (default) or ``"replicate"``, which repeats the edge frame as K-VAE does.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or out_channels
        self.depthwise = depthwise
        self.has_grn = grn
        self.zq_dim = zq_dim
        conv = make_conv(dims, temporal_padding)

        if zq_dim is None:
            self._build_legacy(in_channels, out_channels, mid_channels, dims, conv, grn=grn, kernel_size=kernel_size)
        else:
            self._build_modulated(
                in_channels,
                out_channels,
                mid_channels,
                dims,
                conv,
                grn=grn,
                kernel_size=kernel_size,
                zq_dim=zq_dim,
            )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels else conv(in_channels, out_channels, kernel_size=1)
        )
        self.layer_scale: nn.Module = (
            LayerScale(out_channels, init_value=layer_scale_init, dims=dims)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.stochastic_depth: nn.Module = (
            StochasticDepth(stochastic_depth_prob, mode="row") if stochastic_depth_prob > 0 else nn.Identity()
        )

    def _build_legacy(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        dims: int,
        conv: type[nn.Module],
        *,
        grn: bool,
        kernel_size: int,
    ) -> None:
        """Build the original ``nn.Sequential`` layout (parameter names unchanged)."""
        if self.depthwise:
            ks = kernel_size if dims == 2 else (min(kernel_size, 5), kernel_size, kernel_size)
            pad = kernel_size // 2 if dims == 2 else (min(kernel_size, 5) // 2, kernel_size // 2, kernel_size // 2)
            layers: list[nn.Module] = [
                conv(in_channels, in_channels, kernel_size=ks, padding=pad, groups=in_channels),
                RMSNorm(in_channels, dims=dims),
                conv(in_channels, mid_channels, kernel_size=1),
                nn.SiLU(),
            ]
            if grn:
                layers.append(GRN(mid_channels, dims=dims))
            layers.append(conv(mid_channels, out_channels, kernel_size=1))
        else:
            layers = [
                RMSNorm(in_channels, dims=dims),
                nn.SiLU(),
                conv(in_channels, mid_channels, kernel_size=3, padding=1),
                RMSNorm(mid_channels, dims=dims),
                nn.SiLU(),
            ]
            if grn:
                layers.append(GRN(mid_channels, dims=dims))
            layers.append(conv(mid_channels, out_channels, kernel_size=3, padding=1))
        self.block = nn.Sequential(*layers)

    def _build_modulated(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        dims: int,
        conv: type[nn.Module],
        *,
        grn: bool,
        kernel_size: int,
        zq_dim: int,
    ) -> None:
        """Build a path with FiLM-modulated RMSNorms accepting ``zq`` in forward."""
        if self.depthwise:
            ks = kernel_size if dims == 2 else (min(kernel_size, 5), kernel_size, kernel_size)
            pad = kernel_size // 2 if dims == 2 else (min(kernel_size, 5) // 2, kernel_size // 2, kernel_size // 2)
            self.dwconv = conv(in_channels, in_channels, kernel_size=ks, padding=pad, groups=in_channels)
            self.norm1 = ModulatedRMSNorm(in_channels, zq_dim, dims=dims)
            self.pwconv1 = conv(in_channels, mid_channels, kernel_size=1)
            self.grn: nn.Module | None = GRN(mid_channels, dims=dims) if grn else None
            self.pwconv2 = conv(mid_channels, out_channels, kernel_size=1)
        else:
            self.norm1 = ModulatedRMSNorm(in_channels, zq_dim, dims=dims)
            self.conv1 = conv(in_channels, mid_channels, kernel_size=3, padding=1)
            self.norm2 = ModulatedRMSNorm(mid_channels, zq_dim, dims=dims)
            self.grn = GRN(mid_channels, dims=dims) if grn else None
            self.conv2 = conv(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, zq: Tensor | None = None) -> Tensor:
        """Apply residual block with skip connection.

        Args:
            x: Feature tensor.
            zq: Optional conditioning tensor for FiLM-modulated norms.  Required
                when the block was constructed with ``zq_dim`` set; ignored otherwise.
        """
        out = self.block(x) if self.zq_dim is None else self.forward_modulated_path(x, zq)  # type: ignore
        out = self.layer_scale(out)
        return self.shortcut(x) + self.stochastic_depth(out)

    def forward_modulated_path(
        self,
        x: Tensor,
        zq: Tensor,
        norm1: ModulatedRMSNorm | None = None,
        norm2: ModulatedRMSNorm | None = None,
    ) -> Tensor:
        """Evaluate the residual path with optional scale-specific norms."""
        if self.zq_dim is None:
            msg = "forward_modulated_path requires a block with zq_dim"
            raise ValueError(msg)
        active_norm1 = norm1 if norm1 is not None else self.norm1
        if self.depthwise:
            h = self.dwconv(x)
            h = active_norm1(h, zq)
            h = self.pwconv1(h)
            h = functional.silu(h)
            if self.grn is not None:
                h = self.grn(h)
            return self.pwconv2(h)
        active_norm2 = norm2 if norm2 is not None else self.norm2
        h = active_norm1(x, zq)
        h = functional.silu(h)
        h = self.conv1(h)
        h = active_norm2(h, zq)
        h = functional.silu(h)
        if self.grn is not None:
            h = self.grn(h)
        return self.conv2(h)

    def forward_with_norms(
        self,
        x: Tensor,
        zq: Tensor,
        norm1: ModulatedRMSNorm,
        norm2: ModulatedRMSNorm,
    ) -> Tensor:
        """Apply the block with x2-specific modulation and shared transforms."""
        if self.depthwise:
            msg = "scale-specific normalization does not support depthwise residual blocks"
            raise ValueError(msg)
        out = self.forward_modulated_path(x, zq, norm1=norm1, norm2=norm2)
        out = self.layer_scale(out)
        return self.shortcut(x) + self.stochastic_depth(out)

    def zero_init_residual(self) -> None:
        """Initialize the residual path to zero so the block starts as identity."""
        if self.zq_dim is None:
            final_conv = self.block[-1]
        elif self.depthwise:
            final_conv = self.pwconv2
        else:
            final_conv = self.conv2
        nn.init.zeros_(final_conv.weight)  # type: ignore[arg-type]
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)  # type: ignore[arg-type]


# Spatial upsampling operations: pixel-shuffle, bilinear, and K-VAE PXS v2.


VALID_DIMS = {DIMS_2, DIMS_3}


def icnr_(
    weight: torch.Tensor,
    factor: int,
    init_fn: Callable[[torch.Tensor], torch.Tensor] = nn.init.kaiming_normal_,
) -> None:
    """Initialize sub-pixel-conv weight so initial output equals nearest 2x.

    Implements the ICNR scheme of Aitken et al. (2017): the ``factor**2``
    sub-channel groups consumed by ``PixelShuffle(factor)`` are initialized
    with identical kernels, so on the first forward pass the output is a
    pure nearest-neighbor upsample. This eliminates checkerboard artifacts
    at the start of training.

    Args:
        weight: Conv weight tensor of shape ``(C_out * factor**2, C_in, *kernel)``.
            Modified in-place.
        factor: Upscale factor used by the downstream ``PixelShuffle``.
        init_fn: Base initializer applied to the underlying ``C_out`` kernel
            before replication. Defaults to Kaiming normal.

    Raises:
        ValueError: If ``weight.shape[0]`` is not divisible by ``factor**2``.
    """
    out_c = weight.shape[0] // factor**2
    if out_c * factor**2 != weight.shape[0]:
        msg = f"weight.shape[0] ({weight.shape[0]}) must be divisible by factor**2 ({factor**2})"
        raise ValueError(msg)
    base = torch.empty(out_c, *weight.shape[1:], device=weight.device, dtype=weight.dtype)
    init_fn(base)
    weight.data.copy_(base.repeat_interleave(factor**2, dim=0))


def spatial_nearest_2x(x: torch.Tensor, dims: int, factor: int) -> torch.Tensor:
    """Apply nearest-neighbor upsampling to H, W only, preserving B (and T).

    For 5D ``(B, C, T, H, W)`` tensors, T is merged into batch, interpolated,
    and unmerged. For 4D tensors, interpolation runs directly.

    Args:
        x: Input tensor of shape ``(B, C, H, W)`` or ``(B, C, T, H, W)``.
        dims: Tensor dimensionality — 2 for 4D, 3 for 5D.
        factor: Spatial upscale factor.

    Returns:
        Upsampled tensor with H, W scaled by ``factor``.
    """
    if dims == DIMS_3:
        b, _c, t, _h, _w = x.shape
        x = merge_batch_time(x)
        x = functional.interpolate(x, scale_factor=factor, mode="nearest")
        return split_batch_time(x, b, t)
    return functional.interpolate(x, scale_factor=factor, mode="nearest")


class PixelShuffleND(nn.Module):
    """Conv + pixel-shuffle that upscales only H and W.

    Internally applies a convolution to expand channels by ``upscale_factor ** 2``,
    then rearranges channels into spatial dimensions.

    Supports both 4D ``(B, C, H, W)`` and 5D ``(B, C, T, H, W)`` tensors.
    The temporal dimension is always preserved.

    Args:
        channels: Number of input (and output) channels.
        dims: Tensor dimensionality — 2 for 4D input, 3 for 5D (video) input.
        factor: Spatial upscale factor applied to both H and W.
        temporal_mix: Whether the 3D convolution mixes across the temporal dimension.
            When ``False`` and ``dims == 3``, uses ``kernel_size=(1, 3, 3)`` so that
            each frame is convolved independently in H/W. Ignored when ``dims == 2``.
            Defaults to ``True`` (full 3x3x3 kernel).
        icnr: Apply ICNR initialization (Aitken et al., 2017) to the conv weight
            and zero its bias, so the module starts as a nearest-neighbor upsample.
            Removes checkerboard at init. Defaults to ``False``.
        temporal_padding: How the ``temporal_mix=True`` kernel extends T —
            ``"zeros"`` (default) or ``"replicate"``, matching K-VAE's edge repeat.
    """

    def __init__(
        self,
        channels: int,
        dims: int,
        factor: int = 2,
        *,
        temporal_mix: bool = True,
        icnr: bool = False,
        temporal_padding: TemporalPadding = "zeros",
    ) -> None:
        """Initialize PixelShuffleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.dims = dims
        self.factor = factor

        if dims == DIMS_2:
            self.conv = nn.Conv2d(channels, channels * factor**2, kernel_size=3, padding=1)
        elif temporal_mix:
            conv3d = make_conv(DIMS_3, temporal_padding)
            self.conv = conv3d(channels, channels * factor**2, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv3d(channels, channels * factor**2, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        if icnr:
            icnr_(self.conv.weight, factor)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand channels via convolution, then pixel-shuffle into spatial dimensions."""
        x = self.conv(x)
        if self.dims == DIMS_3:
            b, c_by_factor2, t, h, w = x.shape
            c = c_by_factor2 // self.factor**2
            return (
                x.reshape(b, c, self.factor, self.factor, t, h, w)
                .permute(0, 1, 4, 5, 2, 6, 3)
                .reshape(b, c, t, h * self.factor, w * self.factor)
            )
        return functional.pixel_shuffle(x, self.factor)


class BilinearUpsampleND(nn.Module):
    """Parameter-free bilinear spatial upsampling for 4D and 5D tensors.

    Upscales only H and W dimensions. For 5D ``(B, C, T, H, W)`` tensors,
    the temporal dimension is merged into the batch, interpolated in 2D,
    and unmerged.

    Args:
        dims: Tensor dimensionality — 2 for 4D input, 3 for 5D (video) input.
        factor: Spatial upscale factor applied to both H and W.
    """

    def __init__(self, dims: int, factor: int = 2) -> None:
        """Initialize BilinearUpsampleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.dims = dims
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bilinear interpolation to spatial dimensions."""
        if self.dims == DIMS_3:
            b, _c, t, _h, _w = x.shape
            x = merge_batch_time(x)
            x = functional.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False)
            return split_batch_time(x, b, t)
        return functional.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False)


class PXSv2UpsampleND(nn.Module):
    """K-VAE PXS v2 spatial upsample: ``nearest 2x + Conv_(1,3,3) residual + Conv_1x1``.

    Replicates ``CachedPXSUpsample.spatial_upsample_NEW`` followed by the
    post-``linear`` 1x1x1 conv from K-VAE-3D-2.0. The base path is a
    parameter-free nearest upsample; the residual conv adds learnable detail
    on top, and a final pointwise conv mixes channels.

    Spatial conv is hard-coded to ``kernel=(1, 3, 3)`` for ``dims == 3``,
    matching K-VAE: per-frame, no temporal mixing through the upsample itself.
    Temporal correlation is restored by surrounding ResBlocks.

    Args:
        channels: Number of input (and output) channels.
        dims: Tensor dimensionality — 2 for 4D, 3 for 5D.
        factor: Spatial upscale factor for H and W.
        with_linear: Append a final 1x1 (or 1x1x1) pointwise conv. Disable
            when this module is composed inside a parent that owns its own
            post-mixer (e.g. ``PXSv2HybridUpsampleND``). Defaults to ``True``.
        padding_mode: Edge handling for the residual conv. ``"reflect"`` is the
            historical default; ``"zeros"`` matches the production K-VAE, whose
            ``CachedPXSUpsample`` resolves ``padding_mode or 'reflect'`` against a
            sidecar that passes ``'zeros'``.
    """

    def __init__(
        self,
        channels: int,
        dims: int,
        factor: int = 2,
        *,
        with_linear: bool = True,
        padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize PXSv2UpsampleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.dims = dims
        self.factor = factor

        if dims == DIMS_2:
            self.spatial_conv: nn.Module = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            )
            self.linear: nn.Module = nn.Conv2d(channels, channels, kernel_size=1) if with_linear else nn.Identity()
        else:
            self.spatial_conv = nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                padding_mode=padding_mode,
            )
            self.linear = nn.Conv3d(channels, channels, kernel_size=1) if with_linear else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``Linear( NN_2x(x) + Conv·NN_2x(x) )``."""
        up = spatial_nearest_2x(x, self.dims, self.factor)
        return self.linear(up + self.spatial_conv(up))


class PXSv2HybridUpsampleND(nn.Module):
    """Hybrid upsample: ICNR-PXS branch parallel to PXS v2 branch, mixed by 1x1.

    Combines the learnable sub-pixel-conv branch (``PixelShuffleND``, optionally
    ICNR-initialized so it starts as nearest) with the K-VAE PXS v2 branch
    (``nearest + Conv_(1,3,3) residual``), then mixes both via a final 1x1
    pointwise conv.

    Args:
        channels: Number of input (and output) channels.
        dims: Tensor dimensionality — 2 for 4D, 3 for 5D.
        factor: Spatial upscale factor for H and W.
        temporal_mix: Forwarded to the inner ``PixelShuffleND`` (controls 3x3x3
            vs (1,3,3) kernel for ``dims == 3``). The PXS v2 branch always uses
            (1,3,3) regardless. Defaults to ``True``.
        icnr: Apply ICNR initialization to the inner ``PixelShuffleND`` so it
            starts as nearest, matching the PXS v2 branch and removing
            checkerboard at init. Defaults to ``True``.
        temporal_padding: Forwarded to the inner ``PixelShuffleND``.
        padding_mode: Forwarded to the inner ``PXSv2UpsampleND``.
    """

    def __init__(
        self,
        channels: int,
        dims: int,
        factor: int = 2,
        *,
        temporal_mix: bool = True,
        icnr: bool = True,
        temporal_padding: TemporalPadding = "zeros",
        padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize PXSv2HybridUpsampleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.pxs = PixelShuffleND(
            channels,
            dims=dims,
            factor=factor,
            temporal_mix=temporal_mix,
            icnr=icnr,
            temporal_padding=temporal_padding,
        )
        self.v3 = PXSv2UpsampleND(
            channels,
            dims=dims,
            factor=factor,
            with_linear=False,
            padding_mode=padding_mode,
        )
        if dims == DIMS_2:
            self.linear: nn.Module = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.linear = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``Linear( PXS(x) + (NN_2x + Conv·NN_2x)(x) )``."""
        return self.linear(self.pxs(x) + self.v3(x))


# Construction and execution helpers for the cascaded multi-scale model.


STAGE_FACTOR = 2


@dataclass
class UpsampleSpec:
    """Configuration shared by the two spatial upsample stages."""

    mode: Literal["pixel_shuffle", "bilinear", "pxs_v2", "pxs_v2_hybrid"]
    dims: int
    temporal_mix: bool
    icnr: bool
    temporal_padding: TemporalPadding = "zeros"
    padding_mode: UpsamplePaddingMode = "reflect"


@dataclass
class ResidualStackSpec:
    """Configuration shared by residual stacks at all model resolutions.

    in_channels: input width of the stack's first block; None means use channels.
    """

    channels: int
    mid_channels: int
    total_blocks: int
    stochastic_depth_rate: float
    dims: int
    layer_scale_init: float | None
    grn: bool
    depthwise: bool
    kernel_size: int
    zq_dim: int | None
    in_channels: int | None = None
    temporal_padding: TemporalPadding = "zeros"


class BlockSequence(NamedTuple):
    """Non-registering residual-block execution view."""

    blocks: nn.Sequential


def build_stem(
    in_channels: int,
    hidden_channels: int,
    dims: int,
    *,
    bare: bool,
    temporal_padding: TemporalPadding = "zeros",
) -> nn.Sequential:
    """Build an input projection: bare conv, or the legacy conv + RMSNorm + SiLU."""
    conv = make_conv(dims, temporal_padding)
    projection = conv(in_channels, hidden_channels, kernel_size=3, padding=1)
    if bare:
        return nn.Sequential(projection)
    return nn.Sequential(projection, RMSNorm(hidden_channels, dims=dims), nn.SiLU())


def build_output_head(
    hidden_channels: int,
    out_channels: int,
    dims: int,
    *,
    zq_dim: int | None,
    temporal_padding: TemporalPadding = "zeros",
) -> nn.Sequential:
    """Build a norm + SiLU + conv output head, zq-conditioned when ``zq_dim`` is set."""
    conv = make_conv(dims, temporal_padding)
    norm: nn.Module = RMSNorm(hidden_channels, dims=dims)
    if zq_dim is not None:
        norm = ModulatedRMSNorm(hidden_channels, zq_dim, dims=dims)
    return nn.Sequential(norm, nn.SiLU(), conv(hidden_channels, out_channels, kernel_size=3, padding=1))


def apply_output_head(head: nn.Sequential, x: Tensor, zq: Tensor | None) -> Tensor:
    """Run an output head, threading ``zq`` through a modulated first norm."""
    norm = head[0]
    x = norm(x, zq) if isinstance(norm, ModulatedRMSNorm) else norm(x)
    x = head[1](x)
    return head[2](x)


def build_stage_upsample(channels: int, spec: UpsampleSpec) -> nn.Module:
    """Build one 2x spatial upsample stage."""
    if spec.mode == "pixel_shuffle":
        return PixelShuffleND(
            channels,
            dims=spec.dims,
            factor=STAGE_FACTOR,
            temporal_mix=spec.temporal_mix,
            icnr=spec.icnr,
            temporal_padding=spec.temporal_padding,
        )
    if spec.mode == "bilinear":
        return BilinearUpsampleND(dims=spec.dims, factor=STAGE_FACTOR)
    if spec.mode == "pxs_v2":
        return PXSv2UpsampleND(channels, dims=spec.dims, factor=STAGE_FACTOR, padding_mode=spec.padding_mode)
    return PXSv2HybridUpsampleND(
        channels,
        dims=spec.dims,
        factor=STAGE_FACTOR,
        temporal_mix=spec.temporal_mix,
        icnr=spec.icnr,
        temporal_padding=spec.temporal_padding,
        padding_mode=spec.padding_mode,
    )


def build_residual_stack(count: int, start_index: int, spec: ResidualStackSpec) -> nn.Sequential:
    """Build residual blocks with linearly scheduled stochastic depth."""
    blocks: list[nn.Module] = []
    for index in range(count):
        drop_probability = spec.stochastic_depth_rate * (start_index + index) / max(spec.total_blocks - 1, 1)
        block_in = spec.in_channels if index == 0 and spec.in_channels is not None else spec.channels
        blocks.append(
            ResidualBlock(
                block_in,
                out_channels=spec.channels,
                mid_channels=spec.mid_channels,
                dims=spec.dims,
                layer_scale_init=spec.layer_scale_init,
                grn=spec.grn,
                stochastic_depth_prob=drop_probability,
                depthwise=spec.depthwise,
                kernel_size=spec.kernel_size,
                zq_dim=spec.zq_dim,
                temporal_padding=spec.temporal_padding,
            )
        )
    return nn.Sequential(*blocks)


def apply_block_sequence(x: Tensor, zq: Tensor | None, sequence: BlockSequence) -> Tensor:
    """Run a residual-block stack."""
    for block in sequence.blocks:
        x = block(x, zq) if zq is not None else block(x)
    return x


# Scale-specific x2 adaptation modules for the cascaded latent upscaler.


X2TailMode = Literal["shared", "scale_specific_norm", "private", "private_full"]
X2FinisherMode = Literal["none", "linear", "pxs_residual"]


class SharedSecondStage(NamedTuple):
    """Non-registering view of the x4 second-stage modules."""

    upsample: nn.Module
    blocks: nn.Sequential
    output_proj: nn.Sequential
    mid_blocks: nn.Sequential | None = None


class ScaleSpecificBlockNorms(nn.Module):
    """Private normalization copies for one shared residual block."""

    def __init__(self, block: ResidualBlock) -> None:
        """Copy the modulated norms from a non-depthwise residual block."""
        super().__init__()
        if block.zq_dim is None or block.depthwise:
            msg = "scale-specific norms require non-depthwise modulated residual blocks"
            raise ValueError(msg)
        self.norm1 = copy.deepcopy(block.norm1)
        self.norm2 = copy.deepcopy(block.norm2)


def require_pxs_upsample(source: nn.Module) -> PXSv2UpsampleND:
    """Require a pxs_v2 upsample stage as the finisher donor."""
    if not isinstance(source, PXSv2UpsampleND):
        msg = f"x2_finisher requires a PXSv2UpsampleND source, got {type(source).__name__}"
        raise TypeError(msg)
    return source


class X2Finisher(nn.Module):
    """The upsample-stage conv pair applied at the adapter grid without the nearest-2x resize."""

    def __init__(self, source: nn.Module, mode: X2FinisherMode) -> None:
        """Clone the finisher convs from a pxs_v2 upsample stage."""
        super().__init__()
        if mode == "none":
            msg = "X2Finisher must not be built with x2_finisher='none'"
            raise ValueError(msg)
        donor = require_pxs_upsample(source)
        self.mode = mode
        self.linear = copy.deepcopy(donor.linear)
        self.spatial_conv = copy.deepcopy(donor.spatial_conv) if mode == "pxs_residual" else None

    def forward(self, x: Tensor) -> Tensor:
        """Apply the cloned convs on the unchanged grid."""
        if self.spatial_conv is not None:
            x = x + self.spatial_conv(x)
        return self.linear(x)


class X2Branch(nn.Module):
    """Modules exclusive to the frozen-e115 x2 adaptation path."""

    def __init__(
        self,
        adapter: nn.Sequential,
        tail_mode: X2TailMode,
        shared_stage: SharedSecondStage,
        *,
        dims: int,
        finisher: X2Finisher | None = None,
    ) -> None:
        """Build the selected scale-specific capacity around a shared stage."""
        super().__init__()
        self.adapter = adapter
        self.tail_mode = tail_mode
        self.dims = dims
        self.finisher = finisher

        self.post_norms: nn.ModuleList | None = None
        self.output_norm: RMSNorm | None = None
        self.private_mid_blocks: nn.Sequential | None = None
        self.private_upsample: nn.Module | None = None
        self.private_blocks: nn.Sequential | None = None
        self.private_output_proj: nn.Sequential | None = None

        if tail_mode == "scale_specific_norm":
            self.post_norms = nn.ModuleList([ScaleSpecificBlockNorms(block) for block in shared_stage.blocks])  # type: ignore
            self.output_norm = copy.deepcopy(shared_stage.output_proj[0])  # type: ignore
        elif tail_mode in ("private", "private_full"):
            self.private_upsample = copy.deepcopy(shared_stage.upsample)
            self.private_blocks = copy.deepcopy(shared_stage.blocks)
            self.private_output_proj = copy.deepcopy(shared_stage.output_proj)
            if tail_mode == "private_full":
                if shared_stage.mid_blocks is None:
                    msg = "x2_tail_mode='private_full' requires the shared stage to expose mid_blocks"
                    raise ValueError(msg)
                self.private_mid_blocks = copy.deepcopy(shared_stage.mid_blocks)

    def private_stage(self, shared_stage: SharedSecondStage) -> SharedSecondStage:
        """Select the shared stage or the registered private copy."""
        if self.tail_mode not in ("private", "private_full"):
            return shared_stage
        if self.private_upsample is None or self.private_blocks is None or self.private_output_proj is None:
            msg = "private x2 tail modules are not initialized"
            raise RuntimeError(msg)
        return SharedSecondStage(
            upsample=self.private_upsample,
            blocks=self.private_blocks,
            output_proj=self.private_output_proj,
        )

    def apply_post_blocks(self, x: Tensor, zq: Tensor | None, stage: SharedSecondStage) -> Tensor:
        """Apply post blocks with shared or x2-specific normalization."""
        for index, block in enumerate(stage.blocks):
            if self.tail_mode == "scale_specific_norm":
                if zq is None or self.post_norms is None:
                    msg = "scale-specific normalization requires zq and private norms"
                    raise RuntimeError(msg)
                norms = self.post_norms[index]
                x = block.forward_with_norms(x, zq, norm1=norms.norm1, norm2=norms.norm2)  # type: ignore
            elif zq is None:
                x = block(x)
            else:
                x = block(x, zq)
        return x

    def apply_output_projection(self, x: Tensor, zq: Tensor | None, stage: SharedSecondStage) -> Tensor:
        """Project features with a shared or scale-specific final norm."""
        if self.tail_mode != "scale_specific_norm":
            return apply_output_head(stage.output_proj, x, zq)
        if self.output_norm is None:
            msg = "scale-specific output norm is not initialized"
            raise RuntimeError(msg)
        x = self.output_norm(x)
        x = stage.output_proj[1](x)
        return stage.output_proj[2](x)

    def apply_block_stack(self, blocks: nn.Sequential, x: Tensor, zq: Tensor | None) -> Tensor:
        """Run a residual stack with the branch's zq-threading convention."""
        for block in blocks:
            x = block(x, zq) if zq is not None else block(x)
        return x

    def apply_adapter(self, x: Tensor, zq: Tensor | None) -> Tensor:
        """Run the x2-only residual adapter."""
        return self.apply_block_stack(self.adapter, x, zq)

    def apply_mid_blocks(self, x: Tensor, zq: Tensor | None) -> Tensor:
        """Run the private mid blocks ahead of the second-stage upsample (private_full only)."""
        if self.private_mid_blocks is None:
            return x
        return self.apply_block_stack(self.private_mid_blocks, x, zq)

    def forward_second_stage(
        self,
        x: Tensor,
        zq: Tensor | None,
        batch_size: int,
        frames: int,
        shared_stage: SharedSecondStage,
    ) -> Tensor:
        """Run the selected x2 second stage without registering shared aliases."""
        stage = self.private_stage(shared_stage)
        if self.finisher is not None:
            x = self.finisher(x)
        x = self.apply_mid_blocks(x, zq)
        x = stage.upsample(x)
        x = self.apply_post_blocks(x, zq, stage)
        x = self.apply_output_projection(x, zq, stage)
        if self.dims == 2:
            return split_batch_time(x, batch_size, frames)
        return x


# Multi-scale cascaded upsampler: two 2x stages with intermediate supervision.


class MultiScaleUpsampler(nn.Module):
    """Cascaded 2x+2x VAE-latent upsampler with an optional x2 entry.

    The x4 route keeps the original two-stage architecture. The x2 route enters
    after ``mid_blocks`` through a private stem, optional residual adapter, and
    a shared, scale-normalized, or private second stage. ``x2_tail_mode``
    ``'private_full'`` instead routes through the branch's own warm copies of
    ``mid_blocks`` and the second stage, behind an optional finisher cloned
    from ``upsample_1``.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        bottleneck_channels: int | None = None,
        num_pre_blocks: int = 2,
        num_mid_blocks: int = 6,
        num_post_blocks: int = 8,
        *,
        input_skip: bool = True,
        upsample_mode: UpsampleMode = "pixel_shuffle",
        temporal_mix: bool = True,
        icnr: bool = False,
        dims: int = 3,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        layer_scale_init: float | None = None,
        grn: bool = False,
        stochastic_depth_rate: float = 0.0,
        depthwise: bool = False,
        zq_dim: int | None = None,
        bare_stem: bool = False,
        modulated_output_proj: bool = False,
        global_skip: bool = False,
        enable_x2_entry: bool = False,
        x2_adapter_blocks: int = 0,
        x2_tail_mode: X2TailMode = "shared",
        x2_adapter_sources: tuple[int, ...] | None = None,
        x2_finisher: X2FinisherMode = "none",
        stage_channels: tuple[int, int, int] | None = None,
        temporal_padding: TemporalPadding = "zeros",
        upsample_padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize MultiScaleUpsampler with the given architecture parameters."""
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.upscale_factor = 4
        self.input_skip = input_skip
        self.dims = dims
        self.zq_dim = zq_dim
        self.global_skip = global_skip
        self.enable_x2_entry = enable_x2_entry
        self.x2_tail_mode = x2_tail_mode
        self.x2_adapter_sources = x2_adapter_sources

        if modulated_output_proj and zq_dim is None:
            msg = "modulated_output_proj requires zq_dim (modulated_norm)"
            raise ValueError(msg)
        if enable_x2_entry and global_skip:
            msg = "global_skip does not support the x2 entry"
            raise ValueError(msg)
        if enable_x2_entry and modulated_output_proj and x2_tail_mode != "private_full":
            msg = "modulated_output_proj with the x2 entry requires x2_tail_mode='private_full'"
            raise ValueError(msg)
        if x2_adapter_sources is not None and len(x2_adapter_sources) != x2_adapter_blocks:
            msg = "x2_adapter_sources must list one pre_blocks index per adapter block"
            raise ValueError(msg)
        if x2_adapter_sources is not None and any(not 0 <= index < num_pre_blocks for index in x2_adapter_sources):
            msg = "x2_adapter_sources indices must be within [0, num_pre_blocks)"
            raise ValueError(msg)
        if stage_channels is not None:
            if enable_x2_entry and x2_tail_mode != "private_full":
                msg = "stage_channels with enable_x2_entry requires x2_tail_mode='private_full'"
                raise ValueError(msg)
            if input_skip:
                msg = "stage_channels is incompatible with input_skip"
                raise ValueError(msg)
            if bottleneck_channels is not None:
                msg = "stage_channels is incompatible with bottleneck_channels"
                raise ValueError(msg)
            if hidden_channels != stage_channels[0]:
                msg = f"hidden_channels ({hidden_channels}) must equal stage_channels[0] ({stage_channels[0]})"
                raise ValueError(msg)
        self.stage_channels = stage_channels
        w1, w2, w3 = stage_channels if stage_channels is not None else (hidden_channels,) * 3

        if input_skip and hidden_channels % in_channels != 0:
            msg = (
                f"hidden_channels ({hidden_channels}) must be divisible by in_channels ({in_channels}) for input_skip"
            )
            raise ValueError(msg)

        # Projection layers
        self.input_proj = build_stem(in_channels, w1, dims, bare=bare_stem, temporal_padding=temporal_padding)
        self.mid_output_head = build_output_head(w2, in_channels, dims, zq_dim=None, temporal_padding=temporal_padding)
        self.output_proj = build_output_head(
            w3,
            in_channels,
            dims,
            zq_dim=zq_dim if modulated_output_proj else None,
            temporal_padding=temporal_padding,
        )

        # Two 2x upsample stages
        upsample_spec = UpsampleSpec(
            dims=dims,
            mode=upsample_mode,
            temporal_mix=temporal_mix,
            icnr=icnr,
            temporal_padding=temporal_padding,
            padding_mode=upsample_padding_mode,
        )
        self.upsample_1 = build_stage_upsample(w1, upsample_spec)
        self.upsample_2 = build_stage_upsample(w2, upsample_spec)

        # Residual stacks with linear stochastic depth across all stages; the
        # first block of a stage narrows from the previous stage width.
        total_blocks = num_pre_blocks + num_mid_blocks + num_post_blocks

        def stage_spec(channels: int, first_in: int | None) -> ResidualStackSpec:
            return ResidualStackSpec(
                channels=channels,
                mid_channels=bottleneck_channels if bottleneck_channels is not None else channels * expand_ratio,
                total_blocks=total_blocks,
                stochastic_depth_rate=stochastic_depth_rate,
                dims=dims,
                layer_scale_init=layer_scale_init,
                grn=grn,
                depthwise=depthwise,
                kernel_size=kernel_size,
                zq_dim=zq_dim,
                in_channels=first_in,
                temporal_padding=temporal_padding,
            )

        self.pre_blocks = build_residual_stack(num_pre_blocks, 0, stage_spec(w1, None))
        self.mid_blocks = build_residual_stack(
            num_mid_blocks, num_pre_blocks, stage_spec(w2, w1 if w1 != w2 else None)
        )
        self.post_blocks = build_residual_stack(
            num_post_blocks, num_pre_blocks + num_mid_blocks, stage_spec(w3, w2 if w2 != w3 else None)
        )

        # Build x2-exclusive modules after the complete x4 path so enabling an x2
        # entry does not perturb the base-path RNG initialization.
        self.mid_input_proj: nn.Sequential | None = None
        self.x2_branch: X2Branch | None = None
        if enable_x2_entry:
            self.mid_input_proj = build_stem(
                in_channels, hidden_channels, dims, bare=bare_stem, temporal_padding=temporal_padding
            )
            adapter_spec = replace(
                stage_spec(w1, None),
                total_blocks=max(x2_adapter_blocks, 1),
                stochastic_depth_rate=0.0,
            )
            adapter = build_residual_stack(x2_adapter_blocks, 0, adapter_spec)
            for block in adapter:
                block.zero_init_residual()  # type: ignore[attr-defined]
            finisher = X2Finisher(self.upsample_1, x2_finisher) if x2_finisher != "none" else None
            self.x2_branch = X2Branch(
                adapter,
                x2_tail_mode,
                self.shared_second_stage(),
                dims=dims,
                finisher=finisher,
            )

        if global_skip:
            # Zero heads so step 0 reproduces the nearest base exactly at both scales.
            for head in (self.mid_output_head, self.output_proj):
                final_conv = head[-1]
                nn.init.zeros_(final_conv.weight)  # type: ignore[arg-type]
                nn.init.zeros_(final_conv.bias)  # type: ignore[arg-type]

    def shared_second_stage(self) -> SharedSecondStage:
        """Return a non-registering view of the e115-compatible second stage."""
        return SharedSecondStage(
            upsample=self.upsample_2,
            blocks=self.post_blocks,
            output_proj=self.output_proj,
            mid_blocks=self.mid_blocks,
        )

    def _project_with_skip(self, x: Tensor, proj: nn.Module) -> Tensor:
        """Apply a projection and optionally add the channel-repeated input skip."""
        projected = proj(x)
        if self.input_skip:
            repeats = projected.shape[1] // x.shape[1]
            projected = projected + x.repeat_interleave(repeats, dim=1)
        return projected

    def _head_x4(self, x: Tensor, zq: Tensor | None) -> Tensor:
        """x4 entry: ``input_proj -> input_skip -> pre_blocks -> upsample_1`` (-> 2x grid)."""
        x = self._project_with_skip(x, self.input_proj)
        x = apply_block_sequence(x, zq, BlockSequence(self.pre_blocks))
        return self.upsample_1(x)

    def _head_x2(self, x: Tensor, zq: Tensor | None) -> Tensor:
        """Project a genuine x2 latent and apply the scale-specific adapter."""
        if self.mid_input_proj is None or self.x2_branch is None:
            msg = "x2 head requires enable_x2_entry=True"
            raise ValueError(msg)
        x = self._project_with_skip(x, self.mid_input_proj)
        return self.x2_branch.apply_adapter(x, zq)

    def _tail(
        self,
        x: Tensor,
        zq: Tensor | None,
        b: int,
        t: int,
        *,
        return_intermediates: bool,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """x4 continuation: ``mid_blocks -> [2x head] -> second stage``."""
        x = apply_block_sequence(x, zq, BlockSequence(self.mid_blocks))

        # 2x supervision branch (only when the caller will use it)
        z_2x = None
        if return_intermediates:
            z_2x_inner = self.mid_output_head(x)
            z_2x = split_batch_time(z_2x_inner, b, t) if self.dims == 2 else z_2x_inner

        z_4x = self._second_stage(x, zq, b, t)
        if return_intermediates:
            return {"2x": z_2x, "4x": z_4x}  # type: ignore[return-value]
        return z_4x

    def _second_stage(self, x: Tensor, zq: Tensor | None, b: int, t: int) -> Tensor:
        """Shared second stage: ``upsample_2 -> post_blocks -> output_proj`` — 2x of the current grid."""
        x = self.upsample_2(x)
        x = apply_block_sequence(x, zq, BlockSequence(self.post_blocks))
        x = apply_output_head(self.output_proj, x, zq)
        return split_batch_time(x, b, t) if self.dims == 2 else x

    def forward(
        self,
        z: torch.Tensor,
        *,
        entry: Literal["x4", "x2"] = "x4",
        return_intermediates: bool | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Upsample through the x4 cascade or the configured x2 entry."""
        if entry not in ("x4", "x2"):
            msg = f"entry must be 'x4' or 'x2', got {entry!r}"
            raise ValueError(msg)
        if entry == "x2" and not self.enable_x2_entry:
            msg = "forward(entry='x2') requires the model to be built with enable_x2_entry=True"
            raise ValueError(msg)
        if entry == "x2" and return_intermediates:
            msg = "return_intermediates=True is not supported for entry='x2' — the x2 path has no 2x sub-target"
            raise ValueError(msg)
        if return_intermediates is None:
            return_intermediates = self.training

        b, _, t, _, _ = z.shape
        x = merge_batch_time(z) if self.dims == 2 else z
        zq = x if self.zq_dim is not None else None

        if entry == "x2":
            if self.x2_branch is None:
                msg = "x2 branch requires enable_x2_entry=True"
                raise ValueError(msg)
            x = self._head_x2(x, zq)
            return self.x2_branch.forward_second_stage(x, zq, b, t, self.shared_second_stage())
        x = self._head_x4(x, zq)
        out = self._tail(x, zq, b, t, return_intermediates=return_intermediates)
        if not self.global_skip:
            return out
        return self._add_nearest_base(out, z)

    def _add_nearest_base(
        self,
        out: dict[str, torch.Tensor] | torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Add the parameter-free nearest base of the input latent to each output."""
        if isinstance(out, dict):
            return {
                "2x": out["2x"] + spatial_nearest_2x(z, DIMS_3, STAGE_FACTOR),
                "4x": out["4x"] + spatial_nearest_2x(z, DIMS_3, self.upscale_factor),
            }
        return out + spatial_nearest_2x(z, DIMS_3, self.upscale_factor)


# Factory function for building an upsampler model.


def build_upsampler(config: Mapping[str, Any]) -> MultiScaleUpsampler:
    """Create a `MultiScaleUpsampler` from a serialized model config.

    Args:
        config: Multi-scale model configuration, as stored under `latent_upscaler/config.json`'s
            `models[i].model`. Fields absent from older published configs fall back to
            `MultiScaleUpsampler`'s own defaults.

    Returns:
        Initialized upsampler module.

    Raises:
        ValueError: If `config["architecture"]` is not `"multi_scale"`.
    """
    if config["architecture"] != "multi_scale":
        raise ValueError(f"unsupported latent-upscaler architecture={config['architecture']!r}")
    stage_channels = config["stage_channels"]
    x2_adapter_sources = config.get("x2_adapter_sources")
    return MultiScaleUpsampler(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        bottleneck_channels=config["bottleneck_channels"],
        num_pre_blocks=config["num_pre_blocks"],
        num_mid_blocks=config["num_mid_blocks"],
        num_post_blocks=config["num_post_blocks"],
        input_skip=config["input_skip"],
        upsample_mode=config["upsample_mode"],
        temporal_mix=config["temporal_mix"],
        icnr=config.get("icnr", False),
        dims=config["dims"],
        expand_ratio=config["expand_ratio"],
        kernel_size=config["kernel_size"],
        layer_scale_init=config["layer_scale_init"],
        grn=config["grn"],
        stochastic_depth_rate=config["stochastic_depth_rate"],
        depthwise=config["depthwise"],
        zq_dim=config["in_channels"] if config["modulated_norm"] else None,
        bare_stem=config["bare_stem"],
        modulated_output_proj=config["modulated_output_proj"],
        global_skip=config.get("global_skip", False),
        enable_x2_entry=config["enable_x2_entry"],
        x2_adapter_blocks=config.get("x2_adapter_blocks", 0),
        x2_tail_mode=config.get("x2_tail_mode", "shared"),
        x2_adapter_sources=tuple(x2_adapter_sources) if x2_adapter_sources is not None else None,
        x2_finisher=config.get("x2_finisher", "none"),
        stage_channels=tuple(stage_channels) if stage_channels is not None else None,
        temporal_padding=config["temporal_padding"],
        upsample_padding_mode=config["upsample_padding_mode"],
    )


class Kandinsky6SRLatentUpscalerBank(ModelMixin, ConfigMixin):
    """Diffusers wrapper around the self-contained x2/x4 latent-upscaler bank.

    Args:
        models (`list[dict]`): Serialized upscaler definitions. Each definition
            contains a ``target_scale`` and a validated ``model`` configuration.
        scaling_factor (`float`, *optional*, defaults to 1.0): Factor applied to
            latents before they are passed to an upscaler.
    """

    _no_split_modules = ["MultiScaleUpsampler"]

    @register_to_config
    def __init__(
        self,
        models: Sequence[Mapping[str, Any]],
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self._models = nn.ModuleDict()
        for spec in models:
            target_scale = str(spec["target_scale"])
            if not target_scale.endswith("x"):
                target_scale = f"{target_scale}x"
            if target_scale in self._models:
                raise ValueError(f"duplicate latent upscaler target scale: {target_scale}")
            upscaler = build_upsampler(spec["model"])
            upscaler.target_scale = target_scale
            upscaler.scaling_factor = float(scaling_factor)
            self._models[target_scale] = upscaler

    def for_scale(self, scale: int) -> nn.Module | None:
        """Return the bank entry matching ``scale``."""
        key = f"{int(scale)}x"
        return self._models[key] if key in self._models else None

    @property
    def scales(self) -> tuple[int, ...]:
        """Return all serialized upscale factors."""
        return tuple(sorted(int(key.removesuffix("x")) for key in self._models))

    def forward(
        self,
        latent: Tensor,
        *,
        entry: str | None = None,
        return_intermediates: bool | None = None,
    ) -> Any:
        """Forward one bank entry, primarily for direct component use.

        Args:
            latent (`torch.Tensor`): Latent tensor to upscale.
            entry (`str`, *optional*): Entry name such as ``"x2"`` or ``"2x"``.
                Required when the bank contains multiple scales.
            return_intermediates (`bool`, *optional*): Whether to return
                intermediate upscaler outputs when supported by the entry.

        Returns:
            `torch.Tensor` or `dict`: Upscaled latent, or the selected entry's
            intermediate-output structure.
        """
        if entry is None:
            if len(self._models) != 1:
                raise ValueError("entry is required when the latent-upscaler bank has multiple scales")
            entry = next(iter(self._models))
        key = entry.removeprefix("x")
        if not key.endswith("x"):
            key = f"{key}x"
        model = self._models[key]
        kwargs: dict[str, Any] = {"entry": f"x{key.removesuffix('x')}"}
        if return_intermediates is not None:
            kwargs["return_intermediates"] = return_intermediates
        return model(latent, **kwargs)


__all__ = ["Kandinsky6SRLatentUpscalerBank"]
