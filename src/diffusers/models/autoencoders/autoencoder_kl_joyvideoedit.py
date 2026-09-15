# Copyright 2026 The JoyAI-Video-Edit Team and The HuggingFace Team. All rights reserved.
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
from ...utils.accelerate_utils import apply_forward_hook
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


CACHE_T = 1


class JoyVideoEditRMSNorm(nn.Module):
    r"""
    Channel-first RMS normalization (no learnable bias) used throughout the JoyVideoEdit VAE.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden_states, dim=1) * self.scale * self.gamma


class JoyVideoEditCausalConv3d(nn.Conv3d):
    r"""
    A 3D convolution that is causal in the temporal dimension by construction of its *input*, rather than by padding
    zeros at the front like a standard causal conv.

    Every forward call appends a duplicated copy of the last input frame at the end of the temporal axis before
    convolving (`torch.cat([front, hidden_states, hidden_states[:, :, -1:, :, :]], dim=2)`), where `front` is either
    the last `CACHE_T` frame(s) of the previous chunk (`cache_x`) or, for the very first chunk, the tensor itself
    (which must have a single temporal frame in that case). Padding is applied block-internally rather than as a single
    zero-pad at the very start of the sequence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        assert self.padding[0] == 1, "Causal padding only supports padding of 1 in the temporal dimension."
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0)
        self.padding = (0, 0, 0)

    def forward(self, hidden_states: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        if cache_x is not None:
            front = cache_x.to(hidden_states.device)
        else:
            assert hidden_states.shape[2] == 1, (
                f"Input temporal dimension is expected to be 1 when cache_x is None, got {hidden_states.shape}."
            )
            front = hidden_states
        hidden_states = torch.cat([front, hidden_states, hidden_states[:, :, -1:, :, :]], dim=2)
        hidden_states = F.pad(hidden_states, self._padding)
        return super().forward(hidden_states)


class JoyVideoEditResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = JoyVideoEditRMSNorm(channels)
        self.conv1 = JoyVideoEditCausalConv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = JoyVideoEditRMSNorm(channels)
        self.conv2 = JoyVideoEditCausalConv3d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv1(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv2(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        return hidden_states + residual


class JoyVideoEditVAEAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "JoyVideoEditAttentionBlock", hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        batch_size, channels, num_frames, height, width = hidden_states.shape

        hidden_states = attn.norm(hidden_states)
        query = attn.q(hidden_states)
        key = attn.k(hidden_states)
        value = attn.v(hidden_states)

        # "b c t h w -> (b t) (h w) 1 c"
        query = query.permute(0, 2, 3, 4, 1).contiguous().reshape(batch_size * num_frames, height * width, 1, channels)
        key = key.permute(0, 2, 3, 4, 1).contiguous().reshape(batch_size * num_frames, height * width, 1, channels)
        value = value.permute(0, 2, 3, 4, 1).contiguous().reshape(batch_size * num_frames, height * width, 1, channels)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # "(b t) (h w) 1 c -> b c t h w"
        hidden_states = (
            hidden_states.reshape(batch_size, num_frames, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()
        )

        hidden_states = attn.proj_out(hidden_states)
        return identity + hidden_states


class JoyVideoEditAttentionBlock(nn.Module, AttentionModuleMixin):
    r"""
    Single-head spatial self-attention applied independently to every frame, with separate query, key, value, and
    output projections.
    """

    _default_processor_cls = JoyVideoEditVAEAttnProcessor
    _available_processors = [JoyVideoEditVAEAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = JoyVideoEditRMSNorm(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.set_processor(self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class JoyVideoEditDownsampleBlock(nn.Module):
    r"""
    Causal-conv downsample followed by a space-to-depth fold: the conv reduces the channel count to `out_channels //
    factor`, then `factor = 8` (temporal) or `4` (spatial-only) neighboring positions are folded into the channel
    dimension so the output has `out_channels` channels at `1/factor` the number of positions. A mean-pooled version of
    the *input* (folded the same way) is added back as a residual shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int, temporal_downsample: bool) -> None:
        super().__init__()
        factor = 8 if temporal_downsample else 4
        self.conv = JoyVideoEditCausalConv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.temporal_downsample = temporal_downsample
        self.group_size = factor * in_channels // out_channels

    @staticmethod
    def _space_to_depth(hidden_states: torch.Tensor, factor_t: int) -> torch.Tensor:
        # einops: "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=factor_t, r2=2, r3=2
        batch_size, channels, num_frames, height, width = hidden_states.shape
        num_frames, height, width = num_frames // factor_t, height // 2, width // 2
        hidden_states = hidden_states.reshape(batch_size, channels, num_frames, factor_t, height, 2, width, 2)
        hidden_states = hidden_states.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous()
        return hidden_states.reshape(batch_size, factor_t * 4 * channels, num_frames, height, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        feat_cache: list,
        feat_idx: list[int],
        first_chunk: bool = False,
    ) -> torch.Tensor:
        factor_t = 2 if self.temporal_downsample else 1

        if self.temporal_downsample and first_chunk:
            shortcut = torch.cat([hidden_states[:, :, :1, :, :], hidden_states], dim=2)
        else:
            shortcut = hidden_states
        shortcut = self._space_to_depth(shortcut, factor_t)

        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        if self.temporal_downsample and first_chunk:
            hidden_states = torch.cat([hidden_states[:, :, :1, :, :], hidden_states], dim=2)
        hidden_states = self._space_to_depth(hidden_states, factor_t)

        batch_size, channels, num_frames, height, width = shortcut.shape
        shortcut = shortcut.view(batch_size, hidden_states.shape[1], self.group_size, num_frames, height, width).mean(
            dim=2
        )
        return hidden_states + shortcut


class JoyVideoEditUpsampleBlock(nn.Module):
    r"""
    Causal-conv upsample followed by a depth-to-space unfold: the conv raises the channel count to `out_channels *
    factor`, then unfolded into `factor = 8` (temporal) or `4` (spatial-only) neighboring positions. A
    `repeat_interleave`-based upsample of the input (unfolded the same way) is added back as a residual shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int, temporal_upsample: bool) -> None:
        super().__init__()
        factor = 8 if temporal_upsample else 4
        self.conv = JoyVideoEditCausalConv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

        self.temporal_upsample = temporal_upsample
        self.repeats = factor * out_channels // in_channels

    @staticmethod
    def _depth_to_space(hidden_states: torch.Tensor, factor_t: int) -> torch.Tensor:
        # einops: "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=factor_t, r2=2, r3=2
        batch_size, folded_channels, num_frames, height, width = hidden_states.shape
        channels = folded_channels // (factor_t * 4)
        hidden_states = hidden_states.reshape(batch_size, factor_t, 2, 2, channels, num_frames, height, width)
        hidden_states = hidden_states.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
        return hidden_states.reshape(batch_size, channels, num_frames * factor_t, height * 2, width * 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        feat_cache: list,
        feat_idx: list[int],
        first_chunk: bool = False,
    ) -> torch.Tensor:
        factor_t = 2 if self.temporal_upsample else 1

        shortcut = hidden_states.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = self._depth_to_space(shortcut, factor_t)

        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        hidden_states = self._depth_to_space(hidden_states, factor_t)

        hidden_states = hidden_states + shortcut
        if self.temporal_upsample and first_chunk:
            hidden_states = hidden_states[:, :, 1:, :, :]

        return hidden_states


class JoyVideoEditEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_downsample: tuple[bool, ...],
    ) -> None:
        super().__init__()

        self.conv_in = JoyVideoEditCausalConv3d(in_channels, block_in_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(num_res_blocks):
                self.down_blocks.append(JoyVideoEditResidualBlock(channels=block_in))

            if i_level != len(block_in_channels) - 1:
                block_out = block_in_channels[i_level + 1]
                self.down_blocks.append(JoyVideoEditDownsampleBlock(block_in, block_out, temporal_downsample[i_level]))

        self.mid_blocks = nn.ModuleList(
            [
                JoyVideoEditResidualBlock(channels=block_in),
                JoyVideoEditAttentionBlock(block_in),
                JoyVideoEditResidualBlock(channels=block_in),
            ]
        )

        self.norm_out = JoyVideoEditRMSNorm(block_in)
        self.conv_out = JoyVideoEditCausalConv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        feat_cache: list,
        feat_idx: list[int],
        first_chunk: bool = False,
    ) -> torch.Tensor:
        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv_in(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        for block in self.down_blocks:
            if isinstance(block, JoyVideoEditDownsampleBlock):
                hidden_states = block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
            else:
                hidden_states = block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        for block in self.mid_blocks:
            if isinstance(block, JoyVideoEditResidualBlock):
                hidden_states = block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv_out(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        return hidden_states


class JoyVideoEditDecoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_upsample: tuple[bool, ...],
    ) -> None:
        super().__init__()

        block_in = block_in_channels[0]
        self.conv_in = JoyVideoEditCausalConv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid_blocks = nn.ModuleList(
            [
                JoyVideoEditResidualBlock(channels=block_in),
                JoyVideoEditAttentionBlock(block_in),
                JoyVideoEditResidualBlock(channels=block_in),
            ]
        )

        self.up_blocks = nn.ModuleList([])
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(JoyVideoEditResidualBlock(channels=block_in))

            if i_level != len(block_in_channels) - 1:
                block_out = block_in_channels[i_level + 1]
                self.up_blocks.append(JoyVideoEditUpsampleBlock(block_in, block_out, temporal_upsample[i_level]))

        self.norm_out = JoyVideoEditRMSNorm(block_in)
        self.conv_out = JoyVideoEditCausalConv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        feat_cache: list,
        feat_idx: list[int],
        first_chunk: bool = False,
    ) -> torch.Tensor:
        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv_in(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        for block in self.mid_blocks:
            if isinstance(block, JoyVideoEditResidualBlock):
                hidden_states = block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                hidden_states = block(hidden_states)

        for block in self.up_blocks:
            if isinstance(block, JoyVideoEditUpsampleBlock):
                hidden_states = block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
            else:
                hidden_states = block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        idx = feat_idx[0]
        cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
        hidden_states = self.conv_out(hidden_states, cache_x=feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        return hidden_states


class JoyVideoEditStem(nn.Module):
    """Extra 3/2 spatial down-projection applied to pixels before the encoder backbone.

    `pixel_unshuffle(stride)` folds a `stride x stride` neighborhood into channels, a 1x1 conv reprojects, then
    `pixel_shuffle(group)` unfolds a `group x group` block back to space, giving a net spatial factor of `group /
    stride = 2 / 3`. Combined with the backbone's `patch_size * 2 ** (len(temporal_downsample) - 1) = 16`, the VAE
    reaches an effective spatial compression of 24. Frames are folded into the batch dim so the 2D shuffles act
    per-frame.
    """

    def __init__(self, channels: int, stride: int = 3, group: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.group = group
        self.proj = nn.Conv2d(channels * stride * stride, channels * group * group, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape
        out_height, out_width = height * self.group // self.stride, width * self.group // self.stride
        z = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        z = F.pixel_unshuffle(z, self.stride)
        z = self.proj(z)
        z = F.pixel_shuffle(z, self.group)
        return z.reshape(batch_size, num_frames, channels, out_height, out_width).permute(0, 2, 1, 3, 4)


class JoyVideoEditHeadResBlock(nn.Module):
    """Depthwise-separable residual block used inside `JoyVideoEditHead`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.act(self.dw(x)))


class JoyVideoEditHead(nn.Module):
    """Learned 3/2 spatial up-projection applied after the decoder backbone, inverting `JoyVideoEditStem`.

    Bilinearly upsamples the decoded pixels by `scale` and adds a learned residual refinement. Frames are folded into
    the batch dim so the 2D convolutions act per-frame.
    """

    def __init__(
        self, channels: int, scale: float = 1.5, hidden: int = 32, num_blocks: int = 4, mid_channels: int = 12
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.conv_in = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.blocks = nn.Sequential(*[JoyVideoEditHeadResBlock(hidden) for _ in range(num_blocks)])
        self.reduce = nn.Conv2d(hidden, mid_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape
        out_height, out_width = round(height * self.scale), round(width * self.scale)
        z = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        f = self.act(self.conv_in(z))
        f = self.blocks(f)
        f = self.reduce(f)
        f = F.interpolate(f, size=(out_height, out_width), mode="bilinear", align_corners=False)
        residual = self.conv_out(f)
        base = F.interpolate(z, size=(out_height, out_width), mode="bilinear", align_corners=False)
        out = (base + residual).clamp(-1.0, 1.0)
        return out.reshape(batch_size, num_frames, channels, out_height, out_width).permute(0, 2, 1, 3, 4)


def patchify(hidden_states: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return hidden_states
    # einops: "b c t (h r1) (w r2) -> b (c r1 r2) t h w", r1=patch_size, r2=patch_size
    batch_size, channels, num_frames, height, width = hidden_states.shape
    height, width = height // patch_size, width // patch_size
    hidden_states = hidden_states.reshape(batch_size, channels, num_frames, height, patch_size, width, patch_size)
    hidden_states = hidden_states.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    return hidden_states.reshape(batch_size, channels * patch_size * patch_size, num_frames, height, width)


def unpatchify(hidden_states: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return hidden_states
    # einops: "b (r1 r2 c) t h w -> b c t (h r1) (w r2)", r1=patch_size, r2=patch_size
    batch_size, folded_channels, num_frames, height, width = hidden_states.shape
    channels = folded_channels // (patch_size * patch_size)
    hidden_states = hidden_states.reshape(batch_size, patch_size, patch_size, channels, num_frames, height, width)
    hidden_states = hidden_states.permute(0, 3, 4, 5, 1, 6, 2).contiguous()
    return hidden_states.reshape(batch_size, channels, num_frames, height * patch_size, width * patch_size)


class AutoencoderKLJoyVideoEdit(ModelMixin, AttentionMixin, AutoencoderMixin, ConfigMixin):
    r"""
    A causal, chunk-streamable VAE with KL loss for encoding videos into latents and decoding latent representations
    into videos, used by the JoyVideoEdit pipeline.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False
    _repeated_blocks = None
    _group_offload_block_modules = ["stem", "encoder", "decoder", "head"]
    # keys to ignore when AlignDeviceHook moves inputs/outputs between devices
    # these are shared mutable state modified in-place
    _skip_keys = ["feat_cache", "feat_idx"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: int = 2,
        latent_channels: int = 64,
        layers_per_block: int = 2,
        block_in_channels: tuple[int, ...] = (128, 256, 512, 1024),
        temporal_downsample: tuple[bool, ...] = (True, True, True, False),
        chunk_size: int = 48,
        latents_mean: list[float] = [
            0.003708,
            0.018799,
            -0.049072,
            -1.171875,
            0.064453,
            0.648438,
            -0.507812,
            0.030273,
            -0.090332,
            0.10498,
            -0.18457,
            0.667969,
            -0.863281,
            -0.12793,
            0.000511,
            0.472656,
            -0.636719,
            0.761719,
            0.170898,
            -0.482422,
            0.267578,
            0.092285,
            -0.066406,
            -0.002029,
            0.201172,
            0.026489,
            -0.073242,
            0.016479,
            -0.449219,
            0.070312,
            -0.423828,
            0.804688,
            -1.773438,
            -0.117676,
            0.010986,
            -0.092285,
            -0.003448,
            -0.133789,
            -0.230469,
            -0.410156,
            -0.292969,
            0.414062,
            -0.150391,
            -0.045654,
            -0.213867,
            -0.126953,
            -0.062012,
            -1.039062,
            0.058838,
            -0.015442,
            -0.054932,
            0.100098,
            -0.112793,
            0.0177,
            0.213867,
            -0.003906,
            0.172852,
            0.003281,
            -0.257812,
            0.010071,
            0.008362,
            -0.163086,
            0.126953,
            -1.34375,
        ],
        latents_std: list[float] = [
            0.5625,
            1.710938,
            0.695312,
            2.453125,
            0.769531,
            3.265625,
            3.140625,
            0.835938,
            0.570312,
            0.757812,
            0.925781,
            2.046875,
            2.171875,
            0.503906,
            1.53125,
            1.03125,
            1.90625,
            2.375,
            0.5625,
            0.964844,
            0.699219,
            0.648438,
            3.890625,
            0.707031,
            2.265625,
            0.878906,
            0.550781,
            0.451172,
            2.46875,
            0.53125,
            1.914062,
            3.234375,
            4.65625,
            1.1875,
            0.65625,
            0.738281,
            0.851562,
            0.71875,
            0.796875,
            2.78125,
            1.445312,
            0.589844,
            0.535156,
            0.628906,
            0.734375,
            0.597656,
            0.921875,
            3.09375,
            0.585938,
            0.527344,
            0.570312,
            1.84375,
            0.574219,
            0.617188,
            0.65625,
            0.75,
            0.601562,
            0.539062,
            1.664062,
            0.777344,
            0.507812,
            0.652344,
            0.699219,
            2.8125,
        ],
    ) -> None:
        super().__init__()

        if len(temporal_downsample) != len(block_in_channels):
            raise ValueError(
                "`temporal_downsample` must have one value per block in `block_in_channels`, got "
                f"{len(temporal_downsample)} and {len(block_in_channels)}."
            )
        if temporal_downsample[-1]:
            raise ValueError(
                "The last value must be `False` because the final encoder/decoder block has no temporal "
                "downsample/upsample layer."
            )

        # The encoder/decoder backbone compresses space by `patch_size * 2 ** (len(temporal_downsample) - 1)`; the
        # extra `JoyVideoEditStem` / `JoyVideoEditHead` around it multiply that by 3/2, so pixels are compressed 24x.
        self.backbone_spatial_ratio = patch_size * 2 ** (len(temporal_downsample) - 1)
        self.spatial_compression_ratio = self.backbone_spatial_ratio * 3 // 2
        self.temporal_compression_ratio = 2 ** sum(temporal_downsample[:-1])
        if chunk_size <= 0 or chunk_size % self.temporal_compression_ratio != 0:
            raise ValueError(
                f"`chunk_size` must be a positive multiple of the temporal compression ratio "
                f"({self.temporal_compression_ratio}), got {chunk_size}."
            )

        self.stem = JoyVideoEditStem(in_channels)
        self.encoder = JoyVideoEditEncoder(
            in_channels=in_channels * patch_size**2,
            z_channels=latent_channels,
            num_res_blocks=layers_per_block,
            block_in_channels=block_in_channels,
            temporal_downsample=temporal_downsample,
        )
        self.decoder = JoyVideoEditDecoder(
            z_channels=latent_channels,
            out_channels=out_channels * patch_size**2,
            num_res_blocks=layers_per_block,
            block_in_channels=tuple(reversed(block_in_channels)),
            temporal_upsample=temporal_downsample,
        )
        self.head = JoyVideoEditHead(out_channels)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch
        # dimension to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "encoder": sum(isinstance(m, JoyVideoEditCausalConv3d) for m in self.encoder.modules()),
            "decoder": sum(isinstance(m, JoyVideoEditCausalConv3d) for m in self.decoder.modules()),
        }
        self.clear_cache()

    def clear_cache(self) -> None:
        self._enc_conv_idx = [0]
        self._dec_conv_idx = [0]
        self._enc_feat_map = [None] * self._cached_conv_counts["encoder"]
        self._dec_feat_map = [None] * self._cached_conv_counts["decoder"]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = patchify(x, self.config.patch_size)

        self.clear_cache()
        out = []
        num_chunks = 1 + math.ceil((x.shape[2] - 1) / self.config.chunk_size)
        for i in range(num_chunks):
            self._enc_conv_idx = [0]
            if i == 0:
                chunk = self.encoder(
                    x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx, first_chunk=True
                )
            else:
                start = 1 + (i - 1) * self.config.chunk_size
                end = 1 + i * self.config.chunk_size
                chunk = self.encoder(
                    x[:, :, start:end, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    first_chunk=False,
                )
            out.append(chunk)
        out = torch.cat(out, dim=2)
        self.clear_cache()
        return out

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        r"""
        Encode a batch of videos into latents.

        Args:
            x (`torch.Tensor`): Input batch of videos, shape `(batch, channels, frames, height, width)`. `frames`
                must equal `self.temporal_compression_ratio * n + 1` for some integer `n`, and `height` / `width` must
                be divisible by `self.spatial_compression_ratio`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.modeling_outputs.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        _, _, num_frames, height, width = x.shape
        if (num_frames - 1) % self.temporal_compression_ratio != 0:
            raise ValueError(f"Temporal dimension must be {self.temporal_compression_ratio}n + 1, got {x.shape}.")
        if height % self.spatial_compression_ratio != 0 or width % self.spatial_compression_ratio != 0:
            raise ValueError(
                f"Spatial dimensions must be divisible by {self.spatial_compression_ratio}, got {x.shape}."
            )

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        _, _, num_latent_frames, _, _ = z.shape
        latent_chunk_size = self.config.chunk_size // self.temporal_compression_ratio

        self.clear_cache()
        decoded = []
        num_chunks = 1 + math.ceil((num_latent_frames - 1) / latent_chunk_size)
        for i in range(num_chunks):
            self._dec_conv_idx = [0]
            if i == 0:
                chunk = self.decoder(
                    z[:, :, :1, :, :], feat_cache=self._dec_feat_map, feat_idx=self._dec_conv_idx, first_chunk=True
                )
            else:
                start = 1 + (i - 1) * latent_chunk_size
                end = 1 + i * latent_chunk_size
                chunk = self.decoder(
                    z[:, :, start:end, :, :],
                    feat_cache=self._dec_feat_map,
                    feat_idx=self._dec_conv_idx,
                    first_chunk=False,
                )
            decoded.append(chunk)
        decoded = torch.cat(decoded, dim=2)
        self.clear_cache()

        decoded = unpatchify(decoded, self.config.patch_size)
        return self.head(decoded)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
        r"""
        Decode a batch of latents into videos.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vae.DecoderOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | torch.Tensor:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling
                deterministic.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                If `return_dict` is True, a [`~models.autoencoders.vae.DecoderOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
