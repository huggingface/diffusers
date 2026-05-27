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
DreamLite-specific UNet 2D blocks.

These mirror the upstream ``unet_2d_blocks`` Down/Mid/Up cross-attention blocks but additionally thread the
DreamLite-specific knobs:

- ``use_sep_conv``: replace standard convs in ResnetBlock2DDreamLite with depthwise-separable convs (mobile-friendly).
- ``qk_norm``, ``num_kv_heads``, ``ff_mult``: propagated into DreamLiteTransformer2DModel / BasicTransformerBlock.
- ``RemoveSelfAttn`` variants hard-code ``use_self_attention=False`` in their DreamLiteTransformer2DModel calls.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..attention_processor import Attention  # noqa: F401  (re-export friendliness)
from ..resnet_dreamlite import ResnetBlock2DDreamLite
from ..transformers.dual_transformer_2d import DualTransformer2DModel
from ..transformers.transformer_2d_dreamlite import DreamLiteTransformer2DModel
from .unet_2d_blocks import Downsample2D, Upsample2D, apply_freeu


# ---------------------------------------------------------------------------
# Mid block
# ---------------------------------------------------------------------------
class UNetMidBlock2DCrossAttnDreamLite(nn.Module):
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
def _make_down_block_class(class_name: str, *, remove_self_attn: bool):
    class _DownBlock(nn.Module):
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
                    tf_kwargs = {
                        "num_attention_heads": num_attention_heads,
                        "attention_head_dim": out_channels // num_attention_heads,
                        "in_channels": out_channels,
                        "num_layers": transformer_layers_per_block[i],
                        "cross_attention_dim": cross_attention_dim,
                        "norm_num_groups": resnet_groups,
                        "use_linear_projection": use_linear_projection,
                        "only_cross_attention": only_cross_attention,
                        "upcast_attention": upcast_attention,
                        "attention_type": attention_type,
                        "qk_norm": qk_norm,
                        "ff_mult": ff_mult,
                        "num_kv_heads": num_kv_heads,
                    }
                    if remove_self_attn:
                        tf_kwargs["use_self_attention"] = False
                    attentions.append(DreamLiteTransformer2DModel(**tf_kwargs))
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

    _DownBlock.__name__ = class_name
    _DownBlock.__qualname__ = class_name
    return _DownBlock


CrossAttnDownBlock2DDreamLite = _make_down_block_class("CrossAttnDownBlock2DDreamLite", remove_self_attn=False)
CrossAttnDownRemoveSelfAttnBlock2DDreamLite = _make_down_block_class(
    "CrossAttnDownRemoveSelfAttnBlock2DDreamLite", remove_self_attn=True
)


# ---------------------------------------------------------------------------
# Up blocks
# ---------------------------------------------------------------------------
def _make_up_block_class(class_name: str, *, remove_self_attn: bool):
    class _UpBlock(nn.Module):
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
                    tf_kwargs = {
                        "num_attention_heads": num_attention_heads,
                        "attention_head_dim": out_channels // num_attention_heads,
                        "in_channels": out_channels,
                        "num_layers": transformer_layers_per_block[i],
                        "cross_attention_dim": cross_attention_dim,
                        "norm_num_groups": resnet_groups,
                        "use_linear_projection": use_linear_projection,
                        "only_cross_attention": only_cross_attention,
                        "upcast_attention": upcast_attention,
                        "attention_type": attention_type,
                        "qk_norm": qk_norm,
                        "ff_mult": ff_mult,
                        "num_kv_heads": num_kv_heads,
                    }
                    if remove_self_attn:
                        tf_kwargs["use_self_attention"] = False
                    attentions.append(DreamLiteTransformer2DModel(**tf_kwargs))
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

    _UpBlock.__name__ = class_name
    _UpBlock.__qualname__ = class_name
    return _UpBlock


CrossAttnUpBlock2DDreamLite = _make_up_block_class("CrossAttnUpBlock2DDreamLite", remove_self_attn=False)
CrossAttnUpRemoveSelfAttnBlock2DV1DreamLite = _make_up_block_class(
    "CrossAttnUpRemoveSelfAttnBlock2DV1DreamLite", remove_self_attn=True
)


# ---------------------------------------------------------------------------
# Plain resnet-only blocks (no attention)
# ---------------------------------------------------------------------------
class DownBlock2DDreamLite(nn.Module):
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


class UpBlock2DDreamLite(nn.Module):
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


__all__ = [
    "UNetMidBlock2DCrossAttnDreamLite",
    "CrossAttnDownBlock2DDreamLite",
    "CrossAttnDownRemoveSelfAttnBlock2DDreamLite",
    "CrossAttnUpBlock2DDreamLite",
    "CrossAttnUpRemoveSelfAttnBlock2DV1DreamLite",
    "DownBlock2DDreamLite",
    "UpBlock2DDreamLite",
]
