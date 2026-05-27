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
DreamLite UNet model.

This module defines :class:`DreamLiteUNetModel`, a subclass of :class:`UNet2DConditionModel` that:

* swaps every Down / Mid / Up block for the DreamLite variants defined in :mod:`unet_2d_blocks_dreamlite`, which
  support ``use_sep_conv``, ``qk_norm``, ``num_kv_heads`` and ``ff_mult``;
* defaults its attention processors to :class:`DreamLiteAttnProcessor2_0` (GQA-aware SDPA), which is required because
  the upstream ``AttnProcessor2_0`` does not handle ``kv_heads != heads`` correctly.

Everything else (forward pass, time / class / additional / encoder-hid embeddings, conv-in / conv-out, GLIGEN
positional net, etc.) is inherited unchanged from :class:`UNet2DConditionModel`.
"""

from __future__ import annotations

from torch import nn

from ...configuration_utils import register_to_config
from ..activations import get_activation
from ..attention_processor import Attention, DreamLiteAttnProcessor2_0
from ..normalization import RMSNorm
from .unet_2d_blocks_dreamlite import (
    CrossAttnDownBlock2DDreamLite,
    CrossAttnDownRemoveSelfAttnBlock2DDreamLite,
    CrossAttnUpBlock2DDreamLite,
    CrossAttnUpRemoveSelfAttnBlock2DV1DreamLite,
    DownBlock2DDreamLite,
    UNetMidBlock2DCrossAttnDreamLite,
    UpBlock2DDreamLite,
)
from .unet_2d_condition import UNet2DConditionModel


# ---------------------------------------------------------------------------
# Local block dispatch (DreamLite-only)
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
    if down_block_type == "DownBlock2D":
        return DownBlock2DDreamLite(
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
    if down_block_type in (
        "CrossAttnDownBlock2D",
        "CrossAttnDownRemoveSelfAttnBlock2D",
    ):
        if cross_attention_dim is None:
            raise ValueError(f"cross_attention_dim must be specified for {down_block_type}")
        cls = (
            CrossAttnDownBlock2DDreamLite
            if down_block_type == "CrossAttnDownBlock2D"
            else CrossAttnDownRemoveSelfAttnBlock2DDreamLite
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
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        return UNetMidBlock2DCrossAttnDreamLite(
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
    if up_block_type == "UpBlock2D":
        return UpBlock2DDreamLite(
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
    if up_block_type in (
        "CrossAttnUpBlock2D",
        "CrossAttnUpRemoveSelfAttnBlock2DV1",
    ):
        if cross_attention_dim is None:
            raise ValueError(f"cross_attention_dim must be specified for {up_block_type}")
        cls = (
            CrossAttnUpBlock2DDreamLite
            if up_block_type == "CrossAttnUpBlock2D"
            else CrossAttnUpRemoveSelfAttnBlock2DV1DreamLite
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

    * Down / Mid / Up blocks are dispatched to the DreamLite variants (``unet_2d_blocks_dreamlite``), which support
      depthwise-separable convolutions in resnets and Grouped Query Attention with RMSNorm ``qk_norm`` in attention.
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
            "CrossAttnDownRemoveSelfAttnBlock2D",
            "CrossAttnDownRemoveSelfAttnBlock2D",
            "CrossAttnDownBlock2D",
        ),
        mid_block_type: str | None = "UNetMidBlock2DCrossAttn",
        up_block_types: tuple[str, ...] = (
            "CrossAttnUpBlock2D",
            "CrossAttnUpRemoveSelfAttnBlock2DV1",
            "UpBlock2D",
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


__all__ = ["DreamLiteUNetModel"]
