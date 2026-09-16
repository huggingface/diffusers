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

from __future__ import annotations

import copy
import functools
import importlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import torch
from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional


def rearrange(*args: Any, **kwargs: Any) -> Any:
    """Load einops only when the latent upscaler is actually executed."""
    try:
        from einops import rearrange as _rearrange  # noqa: PLC0415
    except (ImportError, OSError) as exc:
        raise RuntimeError("SR latent upscaler requires einops. Install with `pip install einops`.") from exc
    return _rearrange(*args, **kwargs)


def _model_config(value: Mapping[str, Any]) -> Any:
    """Convert serialized model mappings to attribute-accessible config objects."""
    if not isinstance(value, Mapping):
        raise TypeError(f"latent upscaler model config must be a mapping, got {type(value).__name__}")

    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return SimpleNamespace(**{str(key): convert(val) for key, val in item.items()})
        if isinstance(item, list):
            return tuple(convert(val) for val in item)
        return item

    config = convert(value)
    if getattr(config, "architecture", None) == "multi_scale":
        # Older/published multi-scale configs omit fields whose native
        # constructor defaults are sufficient.
        defaults = {
            "icnr": False,
            "global_skip": False,
            "x2_adapter_blocks": 0,
            "x2_tail_mode": "shared",
            "x2_adapter_sources": None,
            "x2_finisher": "none",
            "motion_attention": None,
        }
        for key, default in defaults.items():
            if not hasattr(config, key):
                setattr(config, key, default)
    return config


UpsampleMode = str
UpsamplePosition = str
FlatModelConfig = Any
MultiScaleModelConfig = Any


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


# Convolution primitives shared by the latent upsampler architectures.


DIMS_2 = 2
DIMS_3 = 3

TemporalPadding = Literal["zeros", "replicate", "causal"]
UpsamplePaddingMode = Literal["reflect", "zeros"]


def as_triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Expand a scalar kernel/padding spec into an explicit ``(t, h, w)`` triple."""
    return (value, value, value) if isinstance(value, int) else value


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
        pad_t, pad_h, pad_w = as_triple(padding)
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
        self.shortcut = nn.Identity() if in_channels == out_channels else conv(in_channels, out_channels, kernel_size=1)
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


# Exact merging of flash-style attention branches with a closed-form null branch.


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def merge_math(
    outputs: Sequence[Tensor],
    lses: Sequence[Tensor],
    null_value: Tensor,
    null_logits: Tensor,
) -> tuple[Tensor, Tensor]:
    """Combine attention branches and the null token into one softmax, in fp32.

    Args:
        outputs: Branch outputs shaped ``(batch, seq, heads, dim)``.
        lses: Branch log-sum-exps shaped ``(batch, seq, heads)``.
        null_value: Null-token value shaped ``(heads, dim)``.
        null_logits: Null-token logits shaped ``(batch, seq, heads)``.

    Returns:
        Merged output ``(batch, seq, heads, dim)`` and total lse ``(batch, seq, heads)``.
    """
    lse_stack = torch.stack([*[lse.float() for lse in lses], null_logits.float()], dim=0)
    total_lse = torch.logsumexp(lse_stack, dim=0)
    weights = torch.exp(lse_stack - total_lse)
    merged = weights[-1].unsqueeze(-1) * null_value.float()
    for weight, output in zip(weights[:-1], outputs, strict=True):
        merged = merged + weight.unsqueeze(-1) * output.float()
    return merged, total_lse


@functools.cache
def compiled_merge_math() -> Callable[..., tuple[Tensor, Tensor]]:
    """Compile the merge math once and reuse it across calls."""
    return torch.compile(merge_math, fullgraph=True)


class MergeWithNullBranch(Function):
    """Merge kernel attention branches with the null branch, with exact gradients.

    Flash-style kernels (NATTEN ``na2d``) return ``(output, lse)`` but their
    backward ignores incoming lse gradients, so plain autograd through a softmax
    merge silently drops part of their q/k/v gradients. This op follows the
    ring-attention convention instead: each kernel branch receives the full
    upstream gradient while its saved output/lse tensors are overwritten in
    place with the merged values, letting the kernel backward reconstruct exact
    global gradients. The null branch is single-key attention, so its gradients
    have a closed form and are computed here directly.

    The in-place overwrite means branch tensors must not be read after backward,
    and the op is incompatible with gradient recomputation (activation
    checkpointing) and double backward.
    """

    @staticmethod
    def forward(ctx: Any, *args: Any) -> Tensor:
        """Merge branches; see :func:`merge_attention_branches` for the layout of ``args``."""
        null_value, null_logits, num_branches, use_compile, *branch_tensors = args
        outputs = branch_tensors[:num_branches]
        lses = branch_tensors[num_branches:]
        math_fn = compiled_merge_math() if use_compile else merge_math
        merged, total_lse = math_fn(outputs, lses, null_value, null_logits)
        ctx.num_branches = num_branches
        ctx.save_for_backward(null_value, null_logits, merged, total_lse, *branch_tensors)
        target = outputs[0] if outputs else null_value
        return merged.to(target.dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        """Route full gradients to kernel branches and closed-form ones to the null pair."""
        num_branches = ctx.num_branches
        null_value, null_logits, merged, total_lse, *branch_tensors = ctx.saved_tensors
        outputs = branch_tensors[:num_branches]
        lses = branch_tensors[num_branches:]
        grad = grad_out.float()
        null_weight = torch.exp(null_logits.float() - total_lse)
        d_null_value = torch.einsum("bsh,bshd->hd", null_weight, grad)
        d_null_logits = null_weight * (
            torch.einsum("bshd,hd->bsh", grad, null_value.float()) - (grad * merged).sum(dim=-1)
        )
        for output, lse in zip(outputs, lses, strict=True):
            output.data.copy_(merged.to(output.dtype))
            lse.data.copy_(total_lse.to(lse.dtype))
        return (
            d_null_value.to(null_value.dtype),
            d_null_logits.to(null_logits.dtype),
            None,
            None,
            *(grad_out for _ in outputs),
            *(None for _ in lses),
        )


def merge_attention_branches(
    outputs: Sequence[Tensor],
    lses: Sequence[Tensor],
    null_value: Tensor,
    null_logits: Tensor,
    *,
    torch_compile: bool = False,
) -> Tensor:
    """Merge kernel attention branches with the closed-form null branch.

    Args:
        outputs: Kernel branch outputs shaped ``(batch, seq, heads, dim)``; may be empty.
        lses: Matching branch log-sum-exps shaped ``(batch, seq, heads)``.
        null_value: Null-token value shaped ``(heads, dim)``.
        null_logits: Null-token logits shaped ``(batch, seq, heads)``.
        torch_compile: Compile the merge math with ``torch.compile``.

    Returns:
        Merged attention output shaped ``(batch, seq, heads, dim)`` in the branch dtype
        (or the null-value dtype when no branches are given).
    """
    return MergeWithNullBranch.apply(null_value, null_logits, len(outputs), torch_compile, *outputs, *lses)


# Runtime helpers shared by latent-upscaler model components.


if TYPE_CHECKING:
    from collections.abc import Callable


def forward_with_checkpointing(
    module: Callable[..., Tensor],
    *inputs: Tensor,
    use_checkpointing: bool = False,
) -> Tensor:
    """Run a callable directly or through non-reentrant activation checkpointing."""
    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(module, *inputs, use_reentrant=False)
    return module(*inputs)


# Spatial upsampling operations: pixel-shuffle, bilinear, and K-VAE PXS v2.


if TYPE_CHECKING:
    from collections.abc import Callable

DIMS_2 = 2
DIMS_3 = 3
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
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = functional.interpolate(x, scale_factor=factor, mode="nearest")
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
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
        """Expand channels via convolution, then rearrange into spatial dimensions."""
        x = self.conv(x)
        if self.dims == DIMS_3:
            return rearrange(x, "b (c p1 p2) t h w -> b c t (h p1) (w p2)", p1=self.factor, p2=self.factor)
        return rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=self.factor, p2=self.factor)


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
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = functional.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False)
            return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
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


# Motion-correspondence attention for latent video features.


if TYPE_CHECKING:
    from types import ModuleType

MotionAttentionBackend = Literal["sdpa", "natten"]
NattenBackend = Literal["cutlass-fna", "hopper-fna", "flex-fna"]


@dataclass
class MotionCorrespondenceSpec:
    """Construction parameters for one motion-correspondence block."""

    channels: int
    num_heads: int
    head_dim: int
    spatial_kernel_size: int
    temporal_offsets: tuple[int, ...]
    backend: MotionAttentionBackend
    natten_backend: NattenBackend = "hopper-fna"
    merge_compile: bool = False


def temporal_neighbor_indices(
    query_index: int,
    num_frames: int,
    temporal_offsets: tuple[int, ...],
) -> tuple[int, ...]:
    """Return valid neighbor-frame indices without shifting or duplication.

    Args:
        query_index: Query frame index.
        num_frames: Number of frames in the clip.
        temporal_offsets: Allowed non-zero offsets from the query frame.

    Returns:
        Valid neighbor indices in the configured offset order.
    """
    return tuple(query_index + offset for offset in temporal_offsets if 0 <= query_index + offset < num_frames)


def spatial_neighborhood_mask(
    height: int,
    width: int,
    kernel_size: int,
    device: torch.device,
) -> Tensor:
    """Build the shifted sliding-window mask used by neighborhood attention.

    Args:
        height: Feature-map height.
        width: Feature-map width.
        kernel_size: Odd spatial neighborhood size.
        device: Device for the returned mask.

    Returns:
        Boolean mask shaped ``(height * width, height * width)``.
    """
    if kernel_size > min(height, width):
        msg = f"spatial kernel {kernel_size} exceeds feature map {height}x{width}"
        raise ValueError(msg)

    query_h, query_w = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    key_h, key_w = query_h.flatten(), query_w.flatten()
    radius = kernel_size // 2
    start_h = (query_h.flatten() - radius).clamp(min=0, max=height - kernel_size)
    start_w = (query_w.flatten() - radius).clamp(min=0, max=width - kernel_size)
    inside_h = (key_h[None, :] >= start_h[:, None]) & (key_h[None, :] < start_h[:, None] + kernel_size)
    inside_w = (key_w[None, :] >= start_w[:, None]) & (key_w[None, :] < start_w[:, None] + kernel_size)
    return inside_h & inside_w


def split_rope_dims(head_dim: int) -> tuple[int, int, int]:
    """Split a head dimension into temporal, height, and width chunks.

    Leftover rotary pairs go to the spatial axes: correspondence is a spatial
    matching task, while temporal positions only span the short offset window.
    """
    if head_dim < 6 or head_dim % 2 != 0:
        msg = f"3D RoPE requires an even head_dim >= 6, got {head_dim}"
        raise ValueError(msg)
    pairs, remainder = divmod(head_dim // 2, 3)
    pair_counts = tuple(pairs + int(axis >= 3 - remainder) for axis in range(3))
    return 2 * pair_counts[0], 2 * pair_counts[1], 2 * pair_counts[2]


class AxialRoPE3D(nn.Module):
    """Axial rotary embeddings for ``(time, height, width)`` coordinates."""

    def __init__(self, head_dim: int, max_period: float = 10000.0) -> None:
        """Initialize per-axis rotary frequencies."""
        super().__init__()
        self.axis_dims = split_rope_dims(head_dim)
        for axis_name, axis_dim in zip(("time", "height", "width"), self.axis_dims, strict=True):
            self.register_buffer(
                f"{axis_name}_frequencies",
                get_freqs(axis_dim // 2, max_period),
                persistent=False,
            )

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """Apply rotary embeddings and return the original working dtype."""
        working_dtype = x.dtype
        chunks = x.to(torch.float32).split(self.axis_dims, dim=-1)
        axes = (1, 2, 3)
        frequencies = (self.time_frequencies, self.height_frequencies, self.width_frequencies)
        rotated = [
            self.rotate_axis(chunk, axis, frequency)
            for chunk, axis, frequency in zip(chunks, axes, frequencies, strict=True)
        ]
        return torch.cat(rotated, dim=-1).to(working_dtype)

    @staticmethod
    def rotate_axis(x: Tensor, axis: int, frequencies: Tensor) -> Tensor:
        """Rotate one channel chunk according to positions along one axis."""
        positions = torch.arange(x.shape[axis], device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, frequencies.to(device=x.device))
        broadcast_shape = [1] * x.ndim
        broadcast_shape[axis] = positions.numel()
        broadcast_shape[-1] = frequencies.numel()
        cosine = angles.cos().reshape(broadcast_shape)
        sine = angles.sin().reshape(broadcast_shape)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        first = pairs[..., 0] * cosine - pairs[..., 1] * sine
        second = pairs[..., 0] * sine + pairs[..., 1] * cosine
        return torch.stack((first, second), dim=-1).flatten(-2)


class MotionCorrespondenceBlock(nn.Module):
    """Cross-frame spatial correspondence with an explicit no-match token."""

    def __init__(self, spec: MotionCorrespondenceSpec) -> None:
        """Initialize a motion-correspondence residual block."""
        super().__init__()
        if spec.channels != spec.num_heads * spec.head_dim:
            msg = f"channels must equal num_heads * head_dim, got {spec.channels} != {spec.num_heads} * {spec.head_dim}"
            raise ValueError(msg)
        if spec.spatial_kernel_size < 3 or spec.spatial_kernel_size % 2 == 0:
            msg = "spatial_kernel_size must be odd and at least 3"
            raise ValueError(msg)
        if (
            not spec.temporal_offsets
            or 0 in spec.temporal_offsets
            or tuple(sorted(set(spec.temporal_offsets))) != spec.temporal_offsets
            or set(spec.temporal_offsets) != {-offset for offset in spec.temporal_offsets}
        ):
            msg = "temporal_offsets must be non-zero, unique, sorted, and symmetric"
            raise ValueError(msg)
        self.channels = spec.channels
        self.num_heads = spec.num_heads
        self.head_dim = spec.head_dim
        self.spatial_kernel_size = spec.spatial_kernel_size
        self.temporal_offsets = spec.temporal_offsets
        self.backend = spec.backend
        self.natten_backend = spec.natten_backend
        self.merge_compile = spec.merge_compile

        self.norm = nn.RMSNorm(spec.channels)
        self.to_qkv = nn.Linear(spec.channels, 3 * spec.channels)
        self.q_norm = nn.RMSNorm(spec.head_dim)
        self.k_norm = nn.RMSNorm(spec.head_dim)
        self.rope = AxialRoPE3D(spec.head_dim)
        self.null_key = nn.Parameter(torch.randn(spec.num_heads, spec.head_dim) * 0.02)
        self.null_value = nn.Parameter(torch.zeros(spec.num_heads, spec.head_dim))
        self.to_out = nn.Linear(spec.channels, spec.channels)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply cross-frame correspondence to ``(B,C,T,H,W)`` features."""
        if x.shape[2] == 1:
            return x
        features = rearrange(x, "b c t h w -> b t h w c")
        # fp32 norm inputs keep fused rms_norm kernels under bf16 autocast (same as DiT QK-norm).
        qkv = self.to_qkv(self.norm(features.float()).type_as(features))
        q, k, v = [
            rearrange(part, "b t h w (nh d) -> b t h w nh d", nh=self.num_heads) for part in qkv.chunk(3, dim=-1)
        ]
        q_unrotated = self.q_norm(q.float()).type_as(q)
        q = self.rope(q_unrotated)
        k = self.rope(self.k_norm(k.float()).type_as(k))
        attention = self.compute_sdpa_attention if self.backend == "sdpa" else self.compute_natten_attention
        context = attention(q, k, v, q_unrotated)
        output = self.to_out(rearrange(context, "b t h w nh d -> b t h w (nh d)"))
        return x + rearrange(output, "b t h w c -> b c t h w")

    def null_match_logits(self, q_unrotated: Tensor) -> Tensor:
        """Score queries against the null token in the shared attention scale.

        Uses pre-RoPE queries: the null token has no position, so pairing it with
        a rotated query would make the no-match threshold depend on where the
        query sits in the clip.
        """
        logits = torch.einsum("...nd,nd->...n", q_unrotated.float(), self.null_key.float())
        return logits * self.head_dim**-0.5

    def compute_sdpa_attention(self, q: Tensor, k: Tensor, v: Tensor, q_unrotated: Tensor) -> Tensor:
        """Compute the reference operation with PyTorch SDPA."""
        batch, frames, height, width, _heads, dim = q.shape
        spatial_mask = spatial_neighborhood_mask(height, width, self.spatial_kernel_size, q.device)
        null_logits = self.null_match_logits(q_unrotated)
        frame_outputs: list[Tensor] = []
        for query_index in range(frames):
            neighbor_indices = temporal_neighbor_indices(query_index, frames, self.temporal_offsets)
            q_frame = rearrange(q[:, query_index], "b h w nh d -> b nh (h w) d")
            k_frames = rearrange(k[:, neighbor_indices], "b t h w nh d -> b nh (t h w) d")
            v_frames = rearrange(v[:, neighbor_indices], "b t h w nh d -> b nh (t h w) d")
            # The null column carries a zero key, so its logit comes solely from the
            # additive bias; SDPA applies the bias after q·k scaling, hence pre-scaled logits.
            null_key = torch.zeros(batch, self.num_heads, 1, dim, dtype=q.dtype, device=q.device)
            null_value = self.null_value.to(q)[None, :, None, :].expand(batch, -1, 1, -1)
            keys = torch.cat((k_frames, null_key), dim=2)
            values = torch.cat((v_frames, null_value), dim=2)
            bias = self.sdpa_attention_bias(spatial_mask, null_logits[:, query_index], len(neighbor_indices))
            attended = functional.scaled_dot_product_attention(q_frame, keys, values, attn_mask=bias.to(q.dtype))
            frame_outputs.append(rearrange(attended, "b nh (h w) d -> b h w nh d", h=height, w=width))
        return torch.stack(frame_outputs, dim=1)

    def sdpa_attention_bias(self, spatial_mask: Tensor, frame_null_logits: Tensor, num_neighbors: int) -> Tensor:
        """Build the additive bias: window gating for real keys, null logits in the last column."""
        window_bias = torch.where(spatial_mask.repeat(1, num_neighbors), 0.0, float("-inf"))
        null_bias = rearrange(frame_null_logits, "b h w nh -> b nh (h w) 1")
        window_bias = window_bias[None, None].expand(*null_bias.shape[:2], -1, -1)
        return torch.cat((window_bias, null_bias), dim=-1)

    def compute_natten_attention(self, q: Tensor, k: Tensor, v: Tensor, q_unrotated: Tensor) -> Tensor:
        """Compute correspondence with fused 2D neighborhood-attention kernels."""
        if q.device.type != "cuda":
            msg = "NATTEN motion attention requires CUDA"
            raise RuntimeError(msg)
        natten = load_natten()
        batch, frames, height, width, _heads, _dim = q.shape
        if self.spatial_kernel_size > min(height, width):
            msg = f"spatial kernel {self.spatial_kernel_size} exceeds feature map {height}x{width}"
            raise ValueError(msg)
        null_logits = self.null_match_logits(q_unrotated)

        grouped_queries: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for query_index in range(frames):
            neighbors = temporal_neighbor_indices(query_index, frames, self.temporal_offsets)
            grouped_queries[len(neighbors)].append((query_index, neighbors))

        frame_outputs: list[Tensor | None] = [None] * frames
        for group in grouped_queries.values():
            query_indices = tuple(query_index for query_index, _neighbors in group)
            q_group = rearrange(q[:, query_indices], "b t h w nh d -> (b t) h w nh d").contiguous()
            group_outputs: list[Tensor] = []
            group_lse: list[Tensor] = []
            for neighbor_position in range(len(group[0][1])):
                neighbor_indices = tuple(neighbors[neighbor_position] for _query_index, neighbors in group)
                k_group = rearrange(k[:, neighbor_indices], "b t h w nh d -> (b t) h w nh d").contiguous()
                v_group = rearrange(v[:, neighbor_indices], "b t h w nh d -> (b t) h w nh d").contiguous()
                output, lse = natten.na2d(
                    q_group,
                    k_group,
                    v_group,
                    kernel_size=(self.spatial_kernel_size, self.spatial_kernel_size),
                    backend=self.natten_backend,
                    return_lse=True,
                )
                group_outputs.append(output.flatten(1, 2))
                group_lse.append(lse.flatten(1, 2))
            merged = merge_attention_branches(
                group_outputs,
                group_lse,
                self.null_value.to(q),
                rearrange(null_logits[:, query_indices], "b t h w nh -> (b t) (h w) nh"),
                torch_compile=self.merge_compile,
            )
            merged = rearrange(
                merged,
                "(b t) (h w) nh d -> b t h w nh d",
                b=batch,
                t=len(query_indices),
                h=height,
                w=width,
            )
            for group_index, query_index in enumerate(query_indices):
                frame_outputs[query_index] = merged[:, group_index]
        if any(output is None for output in frame_outputs):
            msg = "NATTEN correspondence did not produce every query frame"
            raise RuntimeError(msg)
        return torch.stack([output for output in frame_outputs if output is not None], dim=1)


def load_natten() -> ModuleType:
    """Import NATTEN lazily and require its fused CUDA extension."""
    try:
        natten = importlib.import_module("natten")
    except ImportError as error:
        msg = "NATTEN backend requested, but the natten package is not installed"
        raise RuntimeError(msg) from error
    if not getattr(natten, "HAS_LIBNATTEN", False):
        msg = "NATTEN backend requested, but fused libnatten kernels are unavailable"
        raise RuntimeError(msg)
    return natten


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


@dataclass
class MotionAttentionSpec:
    """Construction and placement contract for motion-correspondence blocks."""

    after_mid_blocks: tuple[int, ...]
    block: MotionCorrespondenceSpec


class BlockSequence(NamedTuple):
    """Non-registering residual and motion-attention execution view."""

    blocks: nn.Sequential
    motion_blocks: nn.ModuleList | None = None
    motion_after_blocks: tuple[int, ...] = ()


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


def build_motion_attention_stack(spec: MotionAttentionSpec) -> nn.ModuleList:
    """Build one motion block for every configured mid-stage placement."""
    return nn.ModuleList([MotionCorrespondenceBlock(spec.block) for _placement in spec.after_mid_blocks])


def apply_block_sequence(
    x: Tensor,
    zq: Tensor | None,
    sequence: BlockSequence,
    *,
    use_checkpointing: bool,
) -> Tensor:
    """Run residual blocks with scheduled motion correspondence."""
    for index, block in enumerate(sequence.blocks):
        inputs = (x, zq) if zq is not None else (x,)
        x = forward_with_checkpointing(block, *inputs, use_checkpointing=use_checkpointing)
        block_number = index + 1
        if sequence.motion_blocks is not None and block_number in sequence.motion_after_blocks:
            motion_index = sequence.motion_after_blocks.index(block_number)
            x = forward_with_checkpointing(
                sequence.motion_blocks[motion_index],
                x,
                # NATTEN merge_attentions custom autograd is incompatible with
                # PyTorch non-reentrant activation checkpointing.
                use_checkpointing=False,
            )
    return x


# Convolutional latent-space spatial upsampler with configurable upsampling strategy.


class ConvLatentUpsampler(nn.Module):
    """Lightweight convolutional spatial upsampler for VAE latents.

    Upsampling strategies (controlled by ``upsample_mode``):
        pixel_shuffle: Learned channel-to-space rearrangement (default).
        bilinear: Parameter-free bilinear interpolation.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        bottleneck_channels: int | None = None,
        num_pre_residual_blocks: int = 0,
        num_residual_blocks: int = 16,
        upscale_factor: int = 2,
        *,
        input_skip: bool = True,
        upsample_mode: UpsampleMode = "pixel_shuffle",
        upsample_position: UpsamplePosition = "after_projection",
        temporal_mix: bool = True,
        icnr: bool = False,
        gradient_checkpointing: bool = False,
        dims: int = 2,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        layer_scale_init: float | None = None,
        grn: bool = False,
        stochastic_depth_rate: float = 0.0,
        depthwise: bool = False,
        stem_channels: int | None = None,
        zq_dim: int | None = None,
        temporal_padding: TemporalPadding = "zeros",
        upsample_padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize the upsampler.

        Args:
            in_channels: Number of VAE latent channels.
            hidden_channels: Internal width of residual blocks.
            bottleneck_channels: Mid-channel bottleneck width for ResidualBlocks (None = hidden_channels).
            num_pre_residual_blocks: Number of residual blocks before upsampling (default 0).
            num_residual_blocks: Number of post-upsample residual refinement blocks.
            upscale_factor: Spatial upscale factor for pixel_shuffle and bilinear modes.
            input_skip: Add ``repeat_interleave`` skip connection from input to ``conv_in`` output.
            upsample_mode: Upsampling strategy — "pixel_shuffle" or "bilinear" (parameter-free interpolation).
            upsample_position: Where to apply upsampling — "before_projection" or "after_projection".
            temporal_mix: Whether pixel-shuffle Conv3d mixes across the temporal dimension.
                When False, uses (1,3,3) kernel. Only affects dims=3 + pixel_shuffle.
            icnr: Apply ICNR initialization to the sub-pixel-conv weight inside the
                ``pixel_shuffle`` and ``pxs_v2_hybrid`` upsample modes. Removes
                checkerboard at init. Ignored by ``bilinear`` and ``pxs_v2``.
            gradient_checkpointing: Enable gradient checkpointing for residual blocks.
            dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
            expand_ratio: Expansion ratio for inverted bottleneck (used when ``bottleneck_channels`` is None).
            kernel_size: Kernel size for depthwise convolutions.
            layer_scale_init: Initial LayerScale gamma value. ``None`` disables LayerScale.
            grn: Whether to insert Global Response Normalization in residual blocks.
            stochastic_depth_rate: Maximum drop rate for stochastic depth (linearly scheduled).
            depthwise: Use depthwise separable convolutions in residual blocks.
            stem_channels: When set, pre-residual blocks and input projection operate at this
                narrower width, with a ResidualBlock transition to ``hidden_channels`` after upsampling.
            zq_dim: When set, every ``ResidualBlock`` uses ``ModulatedRMSNorm`` conditioned on
                a side tensor (the LQ latent), re-injecting LQ structure at every block via FiLM.
            temporal_padding: How ``dims == 3`` convolutions extend T — ``"zeros"``
                (default) or ``"replicate"``, which repeats the edge frame as K-VAE does.
            upsample_padding_mode: Edge handling for the ``pxs_v2`` residual conv —
                ``"reflect"`` (default) or ``"zeros"``, matching the production K-VAE.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_pre_residual_blocks = num_pre_residual_blocks
        self.num_residual_blocks = num_residual_blocks
        self.upscale_factor = upscale_factor
        self.upsample_mode = upsample_mode
        self.upsample_position = upsample_position
        self.input_skip = input_skip
        self.gradient_checkpointing = gradient_checkpointing
        self.dims = dims
        self.zq_dim = zq_dim

        # Channel widths: stem (pre-upsample) vs main (post-upsample)
        proj_channels = stem_channels if stem_channels is not None else hidden_channels
        post_bottleneck = bottleneck_channels if bottleneck_channels is not None else hidden_channels * expand_ratio
        pre_bottleneck = bottleneck_channels if bottleneck_channels is not None else proj_channels * expand_ratio
        self.bottleneck_channels = post_bottleneck

        # Projection layers
        conv = make_conv(dims, temporal_padding)
        self.input_proj = nn.Sequential(
            conv(in_channels, proj_channels, kernel_size=3, padding=1),
            RMSNorm(proj_channels, dims=dims),
            nn.SiLU(),
        )
        self.output_proj = nn.Sequential(
            RMSNorm(hidden_channels, dims=dims),
            nn.SiLU(),
            conv(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

        # Spatial upsampling
        upsample_channels = in_channels if upsample_position == "before_projection" else proj_channels
        if upsample_mode == "pixel_shuffle":
            self.upsample: nn.Module = PixelShuffleND(
                upsample_channels,
                dims=dims,
                factor=upscale_factor,
                temporal_mix=temporal_mix,
                icnr=icnr,
                temporal_padding=temporal_padding,
            )
        elif upsample_mode == "bilinear":
            self.upsample = BilinearUpsampleND(dims=dims, factor=upscale_factor)
        elif upsample_mode == "pxs_v2":
            self.upsample = PXSv2UpsampleND(
                upsample_channels,
                dims=dims,
                factor=upscale_factor,
                padding_mode=upsample_padding_mode,
            )
        else:
            self.upsample = PXSv2HybridUpsampleND(
                upsample_channels,
                dims=dims,
                factor=upscale_factor,
                temporal_mix=temporal_mix,
                icnr=icnr,
                temporal_padding=temporal_padding,
                padding_mode=upsample_padding_mode,
            )

        # Channel transition: bridges stem width → hidden width after upsampling
        self.channel_transition: ResidualBlock | None = None
        if stem_channels is not None:
            self.channel_transition = ResidualBlock(
                proj_channels,
                out_channels=hidden_channels,
                mid_channels=post_bottleneck,
                dims=dims,
                grn=grn,
                zq_dim=zq_dim,
                temporal_padding=temporal_padding,
            )

        # Residual stacks
        total_blocks = num_pre_residual_blocks + num_residual_blocks
        shared_block_kwargs = {
            "total_blocks": total_blocks,
            "stochastic_depth_rate": stochastic_depth_rate,
            "dims": dims,
            "layer_scale_init": layer_scale_init,
            "grn": grn,
            "depthwise": depthwise,
            "kernel_size": kernel_size,
            "zq_dim": zq_dim,
            "temporal_padding": temporal_padding,
        }
        self.pre_residual_blocks = self._build_residual_stack(
            num_pre_residual_blocks,
            proj_channels,
            pre_bottleneck,
            start_idx=0,
            **shared_block_kwargs,  # type: ignore[arg-type]
        )
        self.residual_blocks = self._build_residual_stack(
            num_residual_blocks,
            hidden_channels,
            post_bottleneck,
            start_idx=num_pre_residual_blocks,
            **shared_block_kwargs,  # type: ignore[arg-type]
        )

    def _build_residual_stack(
        self,
        count: int,
        channels: int,
        mid_channels: int,
        *,
        start_idx: int,
        total_blocks: int,
        stochastic_depth_rate: float,
        dims: int,
        layer_scale_init: float | None,
        grn: bool,
        depthwise: bool,
        kernel_size: int,
        zq_dim: int | None = None,
        temporal_padding: TemporalPadding = "zeros",
    ) -> nn.Sequential:
        """Build a stack of ResidualBlocks with linearly scheduled stochastic depth."""
        blocks: list[nn.Module] = []
        for i in range(count):
            drop_prob = stochastic_depth_rate * (start_idx + i) / max(total_blocks - 1, 1)
            blocks.append(
                ResidualBlock(
                    channels,
                    mid_channels=mid_channels,
                    dims=dims,
                    layer_scale_init=layer_scale_init,
                    grn=grn,
                    stochastic_depth_prob=drop_prob,
                    depthwise=depthwise,
                    kernel_size=kernel_size,
                    zq_dim=zq_dim,
                    temporal_padding=temporal_padding,
                )
            )
        return nn.Sequential(*blocks)

    def _apply_blocks(
        self,
        x: Tensor,
        blocks: nn.Sequential,
        *,
        use_checkpointing: bool,
        zq: Tensor | None = None,
    ) -> Tensor:
        """Apply a residual stack with optional modulation."""
        for block in blocks:
            if zq is not None:
                x = forward_with_checkpointing(block, x, zq, use_checkpointing=use_checkpointing)
            else:
                x = forward_with_checkpointing(block, x, use_checkpointing=use_checkpointing)
        return x

    def _apply_input_projection(self, x: Tensor) -> Tensor:
        z_skip = x
        x = self.input_proj(x)
        if self.input_skip:
            repeats = x.shape[1] // z_skip.shape[1]
            x = x + z_skip.repeat_interleave(repeats, dim=1)
        return x

    @property
    def last_layer_weight(self) -> torch.Tensor:
        """Weight of the final output conv (used by adaptive GAN-weight balancing)."""
        return self.output_proj[-1].weight  # type: ignore

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Upsample latent tensor spatially.

        Args:
            z: Input latent tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Upsampled tensor of shape ``(B, C, T, H', W')``.
        """
        b, _, t, _, _ = z.shape
        x = rearrange(z, "b c t h w -> (b t) c h w") if self.dims == 2 else z
        zq = x if self.zq_dim is not None else None
        if self.upsample_position == "before_projection":
            x = self.upsample(x)
        x = self._apply_input_projection(x)
        use_ckpt = self.training and self.gradient_checkpointing
        x = self._apply_blocks(
            x,
            self.pre_residual_blocks,
            use_checkpointing=use_ckpt,
            zq=zq,
        )
        if self.upsample_position == "after_projection":
            x = self.upsample(x)
        if self.channel_transition is not None:
            if zq is not None:
                x = forward_with_checkpointing(self.channel_transition, x, zq, use_checkpointing=use_ckpt)
            else:
                x = forward_with_checkpointing(self.channel_transition, x, use_checkpointing=use_ckpt)
        x = self._apply_blocks(
            x,
            self.residual_blocks,
            use_checkpointing=use_ckpt,
            zq=zq,
        )
        x = self.output_proj(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t) if self.dims == 2 else x


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

    def copy_from(self, block: ResidualBlock) -> None:
        """Reset the private norms from their shared x4 counterparts."""
        self.norm1.load_state_dict(block.norm1.state_dict())
        self.norm2.load_state_dict(block.norm2.state_dict())


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

    def copy_from(self, source: nn.Module) -> None:
        """Reset the finisher convs from their x4 upsample counterparts."""
        donor = require_pxs_upsample(source)
        self.linear.load_state_dict(donor.linear.state_dict())
        if self.spatial_conv is not None:
            self.spatial_conv.load_state_dict(donor.spatial_conv.state_dict())

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

    def initialize_from_x4(self, shared_stage: SharedSecondStage) -> None:
        """Copy scale-specific tail state from the loaded x4 baseline."""
        if self.tail_mode == "scale_specific_norm":
            if self.post_norms is None or self.output_norm is None:
                msg = "scale-specific normalization modules are not initialized"
                raise RuntimeError(msg)
            for norms, block in zip(self.post_norms, shared_stage.blocks, strict=True):
                norms.copy_from(block)
            self.output_norm.load_state_dict(shared_stage.output_proj[0].state_dict())
        elif self.tail_mode in ("private", "private_full"):
            if self.private_upsample is None or self.private_blocks is None or self.private_output_proj is None:
                msg = "private x2 tail modules are not initialized"
                raise RuntimeError(msg)
            self.private_upsample.load_state_dict(shared_stage.upsample.state_dict())
            self.private_blocks.load_state_dict(shared_stage.blocks.state_dict())
            self.private_output_proj.load_state_dict(shared_stage.output_proj.state_dict())
            if self.private_mid_blocks is not None:
                if shared_stage.mid_blocks is None:
                    msg = "private mid blocks require the shared stage to expose mid_blocks"
                    raise RuntimeError(msg)
                self.private_mid_blocks.load_state_dict(shared_stage.mid_blocks.state_dict())

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

    def apply_post_blocks(
        self,
        x: Tensor,
        zq: Tensor | None,
        stage: SharedSecondStage,
        *,
        use_checkpointing: bool,
    ) -> Tensor:
        """Apply post blocks with shared or x2-specific normalization."""
        for index, block in enumerate(stage.blocks):
            if self.tail_mode == "scale_specific_norm":
                if zq is None or self.post_norms is None:
                    msg = "scale-specific normalization requires zq and private norms"
                    raise RuntimeError(msg)
                norms = self.post_norms[index]
                block_call = partial(block.forward_with_norms, norm1=norms.norm1, norm2=norms.norm2)  # type: ignore
                x = forward_with_checkpointing(block_call, x, zq, use_checkpointing=use_checkpointing)
            elif zq is None:
                x = forward_with_checkpointing(block, x, use_checkpointing=use_checkpointing)
            else:
                x = forward_with_checkpointing(block, x, zq, use_checkpointing=use_checkpointing)
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

    def apply_block_stack(
        self, blocks: nn.Sequential, x: Tensor, zq: Tensor | None, *, use_checkpointing: bool
    ) -> Tensor:
        """Run a residual stack with the branch's zq-threading convention."""
        for block in blocks:
            if zq is None:
                x = forward_with_checkpointing(block, x, use_checkpointing=use_checkpointing)
            else:
                x = forward_with_checkpointing(block, x, zq, use_checkpointing=use_checkpointing)
        return x

    def apply_adapter(self, x: Tensor, zq: Tensor | None, *, use_checkpointing: bool) -> Tensor:
        """Run the x2-only residual adapter."""
        return self.apply_block_stack(self.adapter, x, zq, use_checkpointing=use_checkpointing)

    def apply_mid_blocks(self, x: Tensor, zq: Tensor | None, *, use_checkpointing: bool) -> Tensor:
        """Run the private mid blocks ahead of the second-stage upsample (private_full only)."""
        if self.private_mid_blocks is None:
            return x
        return self.apply_block_stack(self.private_mid_blocks, x, zq, use_checkpointing=use_checkpointing)

    def forward_second_stage(
        self,
        x: Tensor,
        zq: Tensor | None,
        batch_size: int,
        frames: int,
        shared_stage: SharedSecondStage,
        *,
        use_checkpointing: bool,
    ) -> Tensor:
        """Run the selected x2 second stage without registering shared aliases."""
        stage = self.private_stage(shared_stage)
        if self.finisher is not None:
            x = self.finisher(x)
        x = self.apply_mid_blocks(x, zq, use_checkpointing=use_checkpointing)
        x = stage.upsample(x)
        x = self.apply_post_blocks(x, zq, stage, use_checkpointing=use_checkpointing)
        x = self.apply_output_projection(x, zq, stage)
        if self.dims == 2:
            return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size, t=frames)
        return x


# Multi-scale cascaded upsampler: two 2x stages with intermediate supervision.


# The two projections back to latent space. They are scale-specific, so a warm
# start from another network cannot supply them and a warm-started run may want
# to fit them before letting gradients reach the rest.
HEAD_PREFIXES = ("output_proj.", "mid_output_head.")


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
        gradient_checkpointing: bool = False,
        dims: int = 3,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        layer_scale_init: float | None = None,
        grn: bool = False,
        stochastic_depth_rate: float = 0.0,
        depthwise: bool = False,
        motion_attention: MotionAttentionSpec | None = None,
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
        self.gradient_checkpointing = gradient_checkpointing
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
            if motion_attention is not None:
                msg = "stage_channels is incompatible with motion_attention"
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
        self.motion_after_mid_blocks = motion_attention.after_mid_blocks if motion_attention is not None else ()

        if input_skip and hidden_channels % in_channels != 0:
            msg = f"hidden_channels ({hidden_channels}) must be divisible by in_channels ({in_channels}) for input_skip"
            raise ValueError(msg)
        if motion_attention is not None and dims != 3:
            msg = "motion_attention requires dims=3"
            raise ValueError(msg)
        if motion_attention is not None and enable_x2_entry:
            msg = "motion_attention Round 23 is x4-only and cannot enable the x2 entry"
            raise ValueError(msg)
        if motion_attention is not None and not motion_attention.after_mid_blocks:
            msg = "motion_attention requires at least one mid-stage placement"
            raise ValueError(msg)
        if motion_attention is not None and motion_attention.after_mid_blocks[-1] > num_mid_blocks:
            msg = "motion attention placement exceeds the mid residual stack"
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
        self.mid_blocks = build_residual_stack(num_mid_blocks, num_pre_blocks, stage_spec(w2, w1 if w1 != w2 else None))
        self.post_blocks = build_residual_stack(
            num_post_blocks, num_pre_blocks + num_mid_blocks, stage_spec(w3, w2 if w2 != w3 else None)
        )

        self.mid_motion_blocks: nn.ModuleList | None = None
        if motion_attention is not None:
            self.mid_motion_blocks = build_motion_attention_stack(motion_attention)

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

    @property
    def last_layer_weight(self) -> torch.Tensor:
        """Weight of the final 4x-output conv (used by adaptive GAN-weight balancing).

        The GAN-loss runs only on the 4x output, so the relevant last layer is
        ``output_proj[-1]`` rather than the 2x ``mid_output_head``.
        """
        return self.output_proj[-1].weight  # type: ignore

    def shared_second_stage(self) -> SharedSecondStage:
        """Return a non-registering view of the e115-compatible second stage."""
        return SharedSecondStage(
            upsample=self.upsample_2,
            blocks=self.post_blocks,
            output_proj=self.output_proj,
            mid_blocks=self.mid_blocks,
        )

    @staticmethod
    def is_x2_state_key(key: str) -> bool:
        """Return whether a state key belongs exclusively to the x2 path."""
        return key.startswith(("mid_input_proj.", "x2_branch."))

    def initialize_x2_from_x4(self) -> None:
        """Initialize x2-exclusive modules from the currently loaded x4 path."""
        if self.mid_input_proj is None or self.x2_branch is None:
            msg = "x2 initialization requires enable_x2_entry=True"
            raise ValueError(msg)
        self.mid_input_proj.load_state_dict(self.input_proj.state_dict())
        self.x2_branch.initialize_from_x4(self.shared_second_stage())
        if self.x2_adapter_sources is not None:
            for block, source_index in zip(self.x2_branch.adapter, self.x2_adapter_sources, strict=True):
                block.load_state_dict(self.pre_blocks[source_index].state_dict())
        if self.x2_branch.finisher is not None:
            self.x2_branch.finisher.copy_from(self.upsample_1)

    def configure_x2_finetuning(self) -> None:
        """Freeze the e115 path and expose only x2-exclusive parameters."""
        if self.mid_input_proj is None or self.x2_branch is None:
            msg = "x2 fine-tuning requires enable_x2_entry=True"
            raise ValueError(msg)
        self.requires_grad_(requires_grad=False)
        self.mid_input_proj.requires_grad_(requires_grad=True)
        self.x2_branch.requires_grad_(requires_grad=True)

    def _project_with_skip(self, x: Tensor, proj: nn.Module) -> Tensor:
        """Apply a projection and optionally add the channel-repeated input skip."""
        projected = proj(x)
        if self.input_skip:
            repeats = projected.shape[1] // x.shape[1]
            projected = projected + x.repeat_interleave(repeats, dim=1)
        return projected

    def _head_x4(self, x: Tensor, zq: Tensor | None, *, use_ckpt: bool) -> Tensor:
        """x4 entry: ``input_proj -> input_skip -> pre_blocks -> upsample_1`` (-> 2x grid)."""
        x = self._project_with_skip(x, self.input_proj)
        x = apply_block_sequence(
            x,
            zq,
            BlockSequence(self.pre_blocks),
            use_checkpointing=use_ckpt,
        )
        return self.upsample_1(x)

    def _head_x2(self, x: Tensor, zq: Tensor | None, *, use_ckpt: bool) -> Tensor:
        """Project a genuine x2 latent and apply the scale-specific adapter."""
        if self.mid_input_proj is None or self.x2_branch is None:
            msg = "x2 head requires enable_x2_entry=True"
            raise ValueError(msg)
        x = self._project_with_skip(x, self.mid_input_proj)
        return self.x2_branch.apply_adapter(x, zq, use_checkpointing=use_ckpt)

    def _tail(
        self,
        x: Tensor,
        zq: Tensor | None,
        b: int,
        t: int,
        *,
        use_ckpt: bool,
        return_intermediates: bool,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """x4 continuation: ``mid_blocks -> [2x head] -> second stage``."""
        x = apply_block_sequence(
            x,
            zq,
            BlockSequence(self.mid_blocks, self.mid_motion_blocks, self.motion_after_mid_blocks),
            use_checkpointing=use_ckpt,
        )

        # 2x supervision branch (only when the caller will use it)
        z_2x = None
        if return_intermediates:
            z_2x_inner = self.mid_output_head(x)
            z_2x = rearrange(z_2x_inner, "(b t) c h w -> b c t h w", b=b, t=t) if self.dims == 2 else z_2x_inner

        z_4x = self._second_stage(x, zq, b, t, use_ckpt=use_ckpt)
        if return_intermediates:
            return {"2x": z_2x, "4x": z_4x}  # type: ignore[return-value]
        return z_4x

    def _second_stage(self, x: Tensor, zq: Tensor | None, b: int, t: int, *, use_ckpt: bool) -> Tensor:
        """Shared second stage: ``upsample_2 -> post_blocks -> output_proj`` — 2x of the current grid."""
        x = self.upsample_2(x)
        x = apply_block_sequence(x, zq, BlockSequence(self.post_blocks), use_checkpointing=use_ckpt)
        x = apply_output_head(self.output_proj, x, zq)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t) if self.dims == 2 else x

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
        x = rearrange(z, "b c t h w -> (b t) c h w") if self.dims == 2 else z
        zq = x if self.zq_dim is not None else None
        use_ckpt = self.training and self.gradient_checkpointing

        if entry == "x2":
            if self.x2_branch is None:
                msg = "x2 branch requires enable_x2_entry=True"
                raise ValueError(msg)
            x = self._head_x2(x, zq, use_ckpt=use_ckpt)
            return self.x2_branch.forward_second_stage(
                x,
                zq,
                b,
                t,
                self.shared_second_stage(),
                use_checkpointing=use_ckpt,
            )
        x = self._head_x4(x, zq, use_ckpt=use_ckpt)
        out = self._tail(x, zq, b, t, use_ckpt=use_ckpt, return_intermediates=return_intermediates)
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


# Factory functions for building upsampler models.


if TYPE_CHECKING:
    from torch import nn


def build_upsampler(config: FlatModelConfig | MultiScaleModelConfig) -> nn.Module:
    """Create an upsampler module from config.

    Dispatches on config type: ``FlatModelConfig`` builds a
    ``ConvLatentUpsampler``; ``MultiScaleModelConfig`` builds a
    ``MultiScaleUpsampler``.

    Args:
        config: Model configuration (flat or multi-scale).

    Returns:
        Initialized upsampler module.
    """
    if config.architecture == "multi_scale":
        return _build_multi_scale(config)
    return _build_flat(config)


def _build_flat(config: FlatModelConfig) -> ConvLatentUpsampler:
    """Build a flat (single-stage) ConvLatentUpsampler from config."""
    return ConvLatentUpsampler(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        bottleneck_channels=config.bottleneck_channels,
        num_pre_residual_blocks=config.num_pre_residual_blocks,
        num_residual_blocks=config.num_residual_blocks,
        upscale_factor=config.upscale_factor,
        input_skip=config.input_skip,
        upsample_mode=config.upsample_mode,
        upsample_position=config.upsample_position,
        temporal_mix=config.temporal_mix,
        icnr=config.icnr,
        gradient_checkpointing=config.gradient_checkpointing,
        dims=config.dims,
        expand_ratio=config.expand_ratio,
        kernel_size=config.kernel_size,
        layer_scale_init=config.layer_scale_init,
        grn=config.grn,
        stochastic_depth_rate=config.stochastic_depth_rate,
        depthwise=config.depthwise,
        stem_channels=config.stem_channels,
        zq_dim=config.in_channels if config.modulated_norm else None,
        temporal_padding=config.temporal_padding,
        upsample_padding_mode=config.upsample_padding_mode,
    )


def _build_multi_scale(config: MultiScaleModelConfig) -> MultiScaleUpsampler:
    """Build a cascaded 2x+2x MultiScaleUpsampler from config."""
    motion_attention = build_motion_attention_spec(config)
    return MultiScaleUpsampler(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        bottleneck_channels=config.bottleneck_channels,
        num_pre_blocks=config.num_pre_blocks,
        num_mid_blocks=config.num_mid_blocks,
        num_post_blocks=config.num_post_blocks,
        input_skip=config.input_skip,
        upsample_mode=config.upsample_mode,
        temporal_mix=config.temporal_mix,
        icnr=config.icnr,
        gradient_checkpointing=config.gradient_checkpointing,
        dims=config.dims,
        expand_ratio=config.expand_ratio,
        kernel_size=config.kernel_size,
        layer_scale_init=config.layer_scale_init,
        grn=config.grn,
        stochastic_depth_rate=config.stochastic_depth_rate,
        depthwise=config.depthwise,
        motion_attention=motion_attention,
        zq_dim=config.in_channels if config.modulated_norm else None,
        bare_stem=config.bare_stem,
        modulated_output_proj=config.modulated_output_proj,
        global_skip=config.global_skip,
        enable_x2_entry=config.enable_x2_entry,
        x2_adapter_blocks=config.x2_adapter_blocks,
        x2_tail_mode=config.x2_tail_mode,
        x2_adapter_sources=config.x2_adapter_sources,
        x2_finisher=config.x2_finisher,
        stage_channels=config.stage_channels,
        temporal_padding=config.temporal_padding,
        upsample_padding_mode=config.upsample_padding_mode,
    )


def build_motion_attention_spec(config: MultiScaleModelConfig) -> MotionAttentionSpec | None:
    """Resolve the nested motion-attention config into a construction spec."""
    motion = config.motion_attention
    if motion is None:
        return None
    return MotionAttentionSpec(
        after_mid_blocks=motion.after_mid_blocks,
        block=MotionCorrespondenceSpec(
            channels=config.hidden_channels,
            spatial_kernel_size=motion.spatial_kernel_size,
            temporal_offsets=motion.temporal_offsets,
            num_heads=motion.num_heads,
            head_dim=config.hidden_channels // motion.num_heads,
            backend=motion.backend,
            natten_backend=motion.natten_backend,
            merge_compile=motion.merge_compile,
        ),
    )


# Published Diffusers configs may omit the optional motion-attention section.
# Keep the native factory unchanged while accepting the same config shape in
# the serialized Diffusers adapter.
def build_motion_attention_spec(config: MultiScaleModelConfig) -> MotionAttentionSpec | None:
    """Resolve optional motion-attention settings from a serialized config."""
    motion = getattr(config, "motion_attention", None)
    if motion is None:
        return None
    return MotionAttentionSpec(
        after_mid_blocks=motion.after_mid_blocks,
        block=MotionCorrespondenceSpec(
            channels=config.hidden_channels,
            spatial_kernel_size=motion.spatial_kernel_size,
            temporal_offsets=motion.temporal_offsets,
            num_heads=motion.num_heads,
            head_dim=config.hidden_channels // motion.num_heads,
            backend=motion.backend,
            natten_backend=motion.natten_backend,
            merge_compile=motion.merge_compile,
        ),
    )


class Kandinsky6SRLatentUpscalerBank(ModelMixin, ConfigMixin):
    """Diffusers wrapper around the self-contained x2/x4 latent-upscaler bank.

    Args:
        models (`list[dict]`): Serialized upscaler definitions. Each definition
            contains a ``target_scale`` and a validated ``model`` configuration.
        scaling_factor (`float`, *optional*, defaults to 1.0): Factor applied to
            latents before they are passed to an upscaler.
    """

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
            model_config = _model_config(spec["model"])
            upscaler = build_upsampler(model_config)
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
        kwargs: dict[str, Any] = {}
        if return_intermediates is not None:
            kwargs["return_intermediates"] = return_intermediates
        if hasattr(model, "forward"):
            kwargs["entry"] = f"x{key.removesuffix('x')}"
        try:
            return model(latent, **kwargs)
        except TypeError:
            kwargs.pop("entry", None)
            return model(latent, **kwargs)


__all__ = ["Kandinsky6SRLatentUpscalerBank"]
