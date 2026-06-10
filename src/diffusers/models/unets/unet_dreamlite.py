# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
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
DreamLite UNet model and its constituent 2D blocks.

This single file mirrors the structure used by recent diffusers transformer model files: it defines all DreamLite
building blocks (Down / Mid / Up) and the top-level :class:`DreamLiteUNetModel` together.

Compared to the upstream ``unet_2d_blocks`` Down/Mid/Up cross-attention blocks, the DreamLite variants additionally
thread the following knobs:

- ``use_sep_conv``: replace standard convs in :class:`ResnetBlock2DDreamLite` with depthwise-separable convs
  (mobile-friendly).
- ``qk_norm``, ``num_kv_heads``, ``ff_mult``: propagated into :class:`DreamLiteTransformer2DModel` /
  :class:`BasicTransformerBlockDreamLite`.

The two "no self-attention" variants hard-code ``use_self_attention=False`` in their
:class:`DreamLiteTransformer2DModel` calls.

The U-Net itself defaults its attention processors to :class:`DreamLiteAttnProcessor2_0` (GQA-aware SDPA), which is
required because the upstream ``AttnProcessor2_0`` does not handle ``kv_heads != heads`` correctly.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import register_to_config
from ..activations import get_activation
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..downsampling import Downsample2D as _CoreDownsample2D
from ..downsampling import downsample_2d
from ..normalization import RMSNorm
from ..transformers.dual_transformer_2d import DualTransformer2DModel
from ..transformers.transformer_2d_dreamlite import DreamLiteTransformer2DModel
from ..upsampling import Upsample2D as _CoreUpsample2D
from ..upsampling import upsample_2d
from .unet_2d_blocks import Downsample2D, Upsample2D, apply_freeu
from .unet_2d_condition import UNet2DConditionModel


# ---------------------------------------------------------------------------
# Building blocks (resnet + attention processor)
# ---------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution used by DreamLite mobile-friendly ResNet blocks.

    A depthwise convolution (groups == in_channels) followed by a 1x1 pointwise convolution. The pointwise output
    channel count is multiplied by `expand_ratio` to support inverted-residual style expansion / contraction inside
    [`ResnetBlock2DDreamLite`].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        expand_ratio: float = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, int(out_channels * expand_ratio), kernel_size=1, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.depthwise(hidden_states)
        hidden_states = self.pointwise(hidden_states)
        return hidden_states


class ResnetBlock2DDreamLite(nn.Module):
    r"""
    A ResNet block used by DreamLite. Mirrors [`diffusers.models.resnet.ResnetBlock2D`] with one extra option:

        use_sep_conv (`bool`, *optional*, defaults to `False`):
            Replace the two 3x3 convolutions with [`DepthwiseSeparableConv`]. The first conv expands the channel count
            by 2x; the second conv contracts it back. Used by the mobile-friendly DreamLite checkpoints.

    All other parameters behave identically to [`diffusers.models.resnet.ResnetBlock2D`].
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        use_sep_conv: bool = False,
    ):
        super().__init__()
        if time_embedding_norm in ("ada_group", "spatial"):
            raise ValueError(
                f"`time_embedding_norm`={time_embedding_norm!r} is not supported by `ResnetBlock2DDreamLite`. "
                "Use `diffusers.models.resnet.ResnetBlockCondNorm2D` instead."
            )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        # Inverted-residual style expansion when `use_sep_conv=True`: conv1 expands channels by 2x,
        # conv2 contracts them back. For the standard branch this is just a regular 3x3 conv.
        if use_sep_conv:
            expand_ratio = 2
            self.conv1 = DepthwiseSeparableConv(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, expand_ratio=expand_ratio
            )
            out_channels = out_channels * expand_ratio
        else:
            expand_ratio = 1
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm}")
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        if use_sep_conv:
            self.conv2 = DepthwiseSeparableConv(
                out_channels,
                conv_2d_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                expand_ratio=1 / expand_ratio,
            )
            conv_2d_out_channels = conv_2d_out_channels // expand_ratio
        else:
            self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = _CoreUpsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = _CoreDownsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(f"`temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}")
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            # Only call .contiguous() under training, to avoid DDP gradient-stride warnings while keeping
            # inference fast (especially on CPU). Mirrors the upstream fix from huggingface/diffusers#12975.
            if self.training:
                input_tensor = input_tensor.contiguous()
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class DreamLiteAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention with Grouped Query Attention (GQA / MQA) support.

    Identical to :class:`AttnProcessor2_0` except the key/value reshape branch correctly handles ``attn.kv_heads !=
    attn.heads`` by reshaping K/V to ``kv_heads`` and then ``repeat_interleave``-ing them up to ``attn.heads``. This is
    required by the DreamLite UNet, which combines GQA with ``qk_norm`` — a combination the default
    :class:`AttnProcessor2_0` does not handle. SDPA is delegated to :func:`dispatch_attention_fn` so any of the
    diffusers attention backends (native PyTorch SDPA, FlashAttention, etc.) can be used.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # --- GQA-aware reshape (the only real difference vs AttnProcessor2_0) ---
        # ``dispatch_attention_fn`` expects (batch, seq, heads, head_dim) — keep Q/K/V in that layout
        # and let the dispatched backend handle the transpose internally.
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if kv_heads != attn.heads:
            # GQA / MQA: repeat K/V heads up to query heads for SDPA.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=2, output_size=key.shape[2] * heads_per_kv_head)
            value = torch.repeat_interleave(
                value, heads_per_kv_head, dim=2, output_size=value.shape[2] * heads_per_kv_head
            )
        # ------------------------------------------------------------------------

        # the output of sdp = (batch, seq_len, num_heads, head_dim)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ---------------------------------------------------------------------------
# Mid block
# ---------------------------------------------------------------------------
class DreamLiteUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int | tuple[int] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_groups_out: int | None = None,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        # DreamLite extras
        qk_norm: str | None = None,
        use_sep_conv: bool = False,
        ff_mult: int = 4,
        num_kv_heads: int | None = None,
        num_mid_layers: int = 1,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnet_groups_out = resnet_groups_out or resnet_groups

        resnets = [
            ResnetBlock2DDreamLite(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                groups_out=resnet_groups_out,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_sep_conv=use_sep_conv,
            )
        ]
        attentions = []

        for i in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    DreamLiteTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups_out,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        num_kv_heads=num_kv_heads,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2DDreamLite(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups_out,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_sep_conv=use_sep_conv,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = resnet(hidden_states, temb)

        return hidden_states


# ---------------------------------------------------------------------------
# Down blocks
# ---------------------------------------------------------------------------
class DreamLiteCrossAttnDownBlock2D(nn.Module):
    """DreamLite down block with both self- and cross-attention in each transformer layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int | tuple[int] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        # DreamLite extras
        qk_norm: str | None = None,
        use_sep_conv: bool = False,
        ff_mult: int = 4,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DDreamLite(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_sep_conv=use_sep_conv,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    DreamLiteTransformer2DModel(
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        num_kv_heads=num_kv_heads,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        additional_residuals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DreamLiteCrossAttnNoSelfAttnDownBlock2D(nn.Module):
    """DreamLite down block with cross-attention only (self-attention is removed)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int | tuple[int] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        # DreamLite extras
        qk_norm: str | None = None,
        use_sep_conv: bool = False,
        ff_mult: int = 4,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DDreamLite(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_sep_conv=use_sep_conv,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    DreamLiteTransformer2DModel(
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        num_kv_heads=num_kv_heads,
                        # DreamLite "remove self-attention" path:
                        use_self_attention=False,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        additional_residuals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DreamLiteDownBlock2D(nn.Module):
    """DreamLite plain resnet-only down block (no attention)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        use_sep_conv: bool = False,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2DDreamLite(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_sep_conv=use_sep_conv,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


# ---------------------------------------------------------------------------
# Up blocks
# ---------------------------------------------------------------------------
class DreamLiteCrossAttnUpBlock2D(nn.Module):
    """DreamLite up block with both self- and cross-attention in each transformer layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int | tuple[int] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        # DreamLite extras
        qk_norm: str | None = None,
        use_sep_conv: bool = False,
        ff_mult: int = 4,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DDreamLite(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_sep_conv=use_sep_conv,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    DreamLiteTransformer2DModel(
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        num_kv_heads=num_kv_heads,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        upsample_size: int | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class DreamLiteCrossAttnNoSelfAttnUpBlock2D(nn.Module):
    """DreamLite up block with cross-attention only (self-attention is removed)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int | tuple[int] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        # DreamLite extras
        qk_norm: str | None = None,
        use_sep_conv: bool = False,
        ff_mult: int = 4,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DDreamLite(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_sep_conv=use_sep_conv,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    DreamLiteTransformer2DModel(
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        qk_norm=qk_norm,
                        ff_mult=ff_mult,
                        num_kv_heads=num_kv_heads,
                        # DreamLite "remove self-attention" path:
                        use_self_attention=False,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        upsample_size: int | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class DreamLiteUpBlock2D(nn.Module):
    """DreamLite plain resnet-only up block (no attention)."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        use_sep_conv: bool = False,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2DDreamLite(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_sep_conv=use_sep_conv,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor | None = None,
        upsample_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


# ---------------------------------------------------------------------------
# Local block dispatch (DreamLite-only)
#
# The string ``down_block_type`` / ``up_block_type`` / ``mid_block_type`` keys
# persisted in saved checkpoints' ``config.json`` mirror the Python class names
# defined above. The ``carlofkl/DreamLite-{base,mobile}`` Hub repos
# (``diffusers`` branch) ship configs that use these exact keys.
# ---------------------------------------------------------------------------
def _get_down_block_dreamlite(
    down_block_type: str,
    *,
    num_layers,
    transformer_layers_per_block,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    resnet_groups,
    cross_attention_dim,
    num_attention_heads,
    downsample_padding,
    dual_cross_attention,
    use_linear_projection,
    only_cross_attention,
    upcast_attention,
    resnet_time_scale_shift,
    attention_type,
    dropout,
    qk_norm,
    use_sep_conv,
    ff_mult,
    num_kv_heads,
):
    if down_block_type == "DreamLiteDownBlock2D":
        return DreamLiteDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_sep_conv=use_sep_conv,
        )
    if down_block_type in ("DreamLiteCrossAttnDownBlock2D", "DreamLiteCrossAttnNoSelfAttnDownBlock2D"):
        if cross_attention_dim is None:
            raise ValueError(f"cross_attention_dim must be specified for {down_block_type}")
        cls = (
            DreamLiteCrossAttnDownBlock2D
            if down_block_type == "DreamLiteCrossAttnDownBlock2D"
            else DreamLiteCrossAttnNoSelfAttnDownBlock2D
        )
        return cls(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
            qk_norm=qk_norm,
            use_sep_conv=use_sep_conv,
            ff_mult=ff_mult,
            num_kv_heads=num_kv_heads,
        )
    raise ValueError(f"DreamLite does not support down_block_type={down_block_type!r}")


def _get_mid_block_dreamlite(
    mid_block_type,
    *,
    temb_channels,
    in_channels,
    resnet_eps,
    resnet_act_fn,
    resnet_groups,
    output_scale_factor,
    transformer_layers_per_block,
    num_attention_heads,
    cross_attention_dim,
    dual_cross_attention,
    use_linear_projection,
    upcast_attention,
    resnet_time_scale_shift,
    attention_type,
    dropout,
    qk_norm,
    use_sep_conv,
    ff_mult,
    num_kv_heads,
    num_mid_layers=1,
):
    if mid_block_type is None:
        return None
    if mid_block_type == "DreamLiteUNetMidBlock2DCrossAttn":
        return DreamLiteUNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
            qk_norm=qk_norm,
            use_sep_conv=use_sep_conv,
            ff_mult=ff_mult,
            num_kv_heads=num_kv_heads,
            num_layers=num_mid_layers,
        )
    raise ValueError(f"DreamLite does not support mid_block_type={mid_block_type!r}")


def _get_up_block_dreamlite(
    up_block_type,
    *,
    num_layers,
    transformer_layers_per_block,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    resolution_idx,
    resnet_groups,
    cross_attention_dim,
    num_attention_heads,
    dual_cross_attention,
    use_linear_projection,
    only_cross_attention,
    upcast_attention,
    resnet_time_scale_shift,
    attention_type,
    dropout,
    qk_norm,
    use_sep_conv,
    ff_mult,
    num_kv_heads,
):
    if up_block_type == "DreamLiteUpBlock2D":
        return DreamLiteUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_sep_conv=use_sep_conv,
        )
    if up_block_type in ("DreamLiteCrossAttnUpBlock2D", "DreamLiteCrossAttnNoSelfAttnUpBlock2D"):
        if cross_attention_dim is None:
            raise ValueError(f"cross_attention_dim must be specified for {up_block_type}")
        cls = (
            DreamLiteCrossAttnUpBlock2D
            if up_block_type == "DreamLiteCrossAttnUpBlock2D"
            else DreamLiteCrossAttnNoSelfAttnUpBlock2D
        )
        return cls(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
            qk_norm=qk_norm,
            use_sep_conv=use_sep_conv,
            ff_mult=ff_mult,
            num_kv_heads=num_kv_heads,
        )
    raise ValueError(f"DreamLite does not support up_block_type={up_block_type!r}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DreamLiteUNetModel(UNet2DConditionModel):
    r"""
    DreamLite variant of :class:`UNet2DConditionModel`.

    Differences vs the parent class:

    * Down / Mid / Up blocks are dispatched to the DreamLite variants defined above, which support depthwise-separable
      convolutions in resnets and Grouped Query Attention with RMSNorm ``qk_norm`` in attention.
    * ``default_attn_processor`` returns :class:`DreamLiteAttnProcessor2_0` so SDPA is GQA-aware out of the box.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int | tuple[int, int] | None = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: tuple[str, ...] = (
            "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
            "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
            "DreamLiteCrossAttnDownBlock2D",
        ),
        mid_block_type: str | None = "DreamLiteUNetMidBlock2DCrossAttn",
        up_block_types: tuple[str, ...] = (
            "DreamLiteCrossAttnUpBlock2D",
            "DreamLiteCrossAttnNoSelfAttnUpBlock2D",
            "DreamLiteUpBlock2D",
        ),
        only_cross_attention: bool | tuple[bool, ...] = False,
        block_out_channels: tuple[int, ...] = (320, 640, 1280),
        layers_per_block: int | tuple[int, ...] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: int | None = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int | tuple[int, ...] = 2048,
        transformer_layers_per_block: int | tuple[int, ...] | tuple[tuple, ...] = 1,
        reverse_transformer_layers_per_block: tuple[tuple[int, ...], ...] | None = None,
        encoder_hid_dim: int | None = None,
        encoder_hid_dim_type: str | None = None,
        attention_head_dim: int | tuple[int, ...] = 64,
        num_attention_heads: int | tuple[int, ...] | None = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: str | None = None,
        addition_embed_type: str | None = None,
        addition_time_embed_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: int | None = None,
        time_embedding_act_fn: str | None = None,
        timestep_post_act: str | None = None,
        time_cond_proj_dim: int | None = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: int | None = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: bool | None = None,
        cross_attention_norm: str | None = None,
        addition_embed_type_num_heads: int = 64,
        # ---- DreamLite extras ----
        qk_norm: str | None = "rms_norm",
        use_sep_conv: bool = True,
        ff_mult: int = 6,
        num_kv_heads: int | None = 1,
        num_mid_layers: int = 1,
    ):
        # NOTE: deliberately skip UNet2DConditionModel.__init__ and call nn.Module directly,
        # because we replicate the body with DreamLite block dispatch.
        nn.Module.__init__(self)

        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via "
                "`num_attention_heads` because of a naming issue as described in "
                "https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. "
                "Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )
        num_attention_heads = num_attention_heads or attention_head_dim

        # Reuse parent helpers (they only touch self, no super().__init__ required).
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
        )

        self.projection_class_embeddings_input_dim = projection_class_embeddings_input_dim

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )

        from ..embeddings import TimestepEmbedding  # local import to avoid cycle

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self._set_encoder_hid_proj(
            encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )
        self._set_class_embedding(
            class_embed_type,
            act_fn=act_fn,
            num_class_embeds=num_class_embeds,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )
        self._set_add_embedding(
            addition_embed_type,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

        self.time_embed_act = None if time_embedding_act_fn is None else get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # Normalize per-stage args
        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)
        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim * 2 if class_embeddings_concat else time_embed_dim

        # ---- Down ----
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(
                _get_down_block_dreamlite(
                    down_block_type,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    downsample_padding=downsample_padding,
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    use_sep_conv=use_sep_conv,
                    ff_mult=ff_mult,
                    num_kv_heads=num_kv_heads,
                )
            )

        # ---- Mid ----
        self.mid_block = _get_mid_block_dreamlite(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            dropout=dropout,
            qk_norm=qk_norm,
            use_sep_conv=use_sep_conv,
            ff_mult=ff_mult,
            num_kv_heads=num_kv_heads,
            num_mid_layers=num_mid_layers,
        )

        # ---- Up ----
        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            self.up_blocks.append(
                _get_up_block_dreamlite(
                    up_block_type,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resolution_idx=i,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    use_sep_conv=use_sep_conv,
                    ff_mult=ff_mult,
                    num_kv_heads=num_kv_heads,
                )
            )

        # ---- Out ----
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
            self.conv_act = get_activation(act_fn)
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

        self._set_pos_net_if_use_gligen(attention_type=attention_type, cross_attention_dim=cross_attention_dim)

        # ---- DreamLite: install GQA-aware processor everywhere ----
        for module in self.modules():
            if isinstance(module, Attention):
                module.set_processor(DreamLiteAttnProcessor2_0())

    # ----- override default processor so set_attn_processor("default") restores GQA ----
    @property
    def default_attn_processor(self):  # type: ignore[override]
        return DreamLiteAttnProcessor2_0()

    def set_default_attn_processor(self):  # type: ignore[override]
        """Reinstall :class:`DreamLiteAttnProcessor2_0` everywhere.

        The parent implementation only knows about the diffusers stock processor sets and would raise
        for our GQA-aware processor; override so utilities that round-trip through this method (CPU
        offload, save/load, layerwise casting, ...) keep working unchanged.
        """
        self.set_attn_processor(DreamLiteAttnProcessor2_0())

    # ----- DreamLite extension: support `text_proj_rms` encoder_hid_proj -----
    def _set_encoder_hid_proj(  # type: ignore[override]
        self,
        encoder_hid_dim_type,
        cross_attention_dim,
        encoder_hid_dim,
    ):
        """
        Override to support DreamLite's `text_proj_rms` variant (Linear → RMSNorm). All other variants fall back to the
        parent implementation, preserving full compatibility with upstream configs (`text_proj`, `text_image_proj`,
        `image_proj`, ...).
        """
        if encoder_hid_dim_type == "text_proj_rms":
            if encoder_hid_dim is None:
                raise ValueError(
                    "`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to 'text_proj_rms'."
                )
            self.encoder_hid_proj = nn.Sequential(
                nn.Linear(encoder_hid_dim, cross_attention_dim),
                RMSNorm(cross_attention_dim, eps=1e-5, elementwise_affine=True),
            )
            return
        super()._set_encoder_hid_proj(
            encoder_hid_dim_type=encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )

    # ----- DreamLite extension: dispatch `text_proj_rms` like `text_proj` -----
    def process_encoder_hidden_states(  # type: ignore[override]
        self, encoder_hidden_states, added_cond_kwargs
    ):
        """
        For `text_proj_rms`, the projection is a plain `nn.Sequential` applied to `encoder_hidden_states` (same call
        signature as `text_proj`). All other variants are delegated to the parent.
        """
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj_rms":
            return self.encoder_hid_proj(encoder_hidden_states)
        return super().process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )

    # ----- DreamLite extension: support `addition_embed_type == "time"` -----
    def _set_add_embedding(  # type: ignore[override]
        self,
        addition_embed_type,
        addition_embed_type_num_heads,
        addition_time_embed_dim,
        flip_sin_to_cos,
        freq_shift,
        cross_attention_dim,
        encoder_hid_dim,
        projection_class_embeddings_input_dim,
        time_embed_dim,
    ):
        """
        Override to support DreamLite's `addition_embed_type == "time"` variant (same module layout as `text_time` but
        `get_aug_embed` does not require `text_embeds`). All other variants delegate to the parent implementation.
        """
        if addition_embed_type == "time":
            from ..embeddings import TimestepEmbedding, Timesteps

            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
            return
        super()._set_add_embedding(
            addition_embed_type=addition_embed_type,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

    # ----- DreamLite extension: dispatch `addition_embed_type == "time"` -----
    def get_aug_embed(  # type: ignore[override]
        self, emb, encoder_hidden_states, added_cond_kwargs
    ):
        """
        For `addition_embed_type == "time"`, build aug_emb from `time_ids` only (no `text_embeds`). All other variants
        are delegated to the parent.
        """
        if self.config.addition_embed_type == "time":
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'time' "
                    "which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((-1, self.config.projection_class_embeddings_input_dim))
            add_embeds = time_embeds.to(emb.dtype)
            return self.add_embedding(add_embeds)
        return super().get_aug_embed(
            emb=emb,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )


__all__ = [
    "DreamLiteUNetModel",
    "DreamLiteUNetMidBlock2DCrossAttn",
    "DreamLiteCrossAttnDownBlock2D",
    "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
    "DreamLiteCrossAttnUpBlock2D",
    "DreamLiteCrossAttnNoSelfAttnUpBlock2D",
    "DreamLiteDownBlock2D",
    "DreamLiteUpBlock2D",
]
