from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import is_torch_version
from ..utils.torch_utils import apply_freeu
from .attention_processor import Attention
from .dual_transformer_2d import DualTransformer2DModel
from .embeddings import PositionalEmbedding
from .modeling_utils import ModelMixin
from .resnet import Downsample2D, ResnetBlock2D, Upsample2D
from .transformer_2d import Transformer2DModel
from .transformer_temporal import TransformerTemporalModel, TransformerTemporalMotionModel


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    num_attention_heads,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    motion_norm_num_groups=32,
    motion_cross_attention_dim=None,
    motion_num_attention_heads=8,
    motion_attention_bias=False,
    motion_activation_fn="geglu",
    motion_max_seq_length=32,
):
    if down_block_type == "DownBlockMotion":
        return DownBlockMotion(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            motion_norm_num_groups=motion_norm_num_groups,
            motion_cross_attention_dim=motion_cross_attention_dim,
            motion_num_attention_heads=motion_num_attention_heads,
            motion_attention_bias=motion_attention_bias,
            motion_activation_fn=motion_activation_fn,
            motion_max_seq_length=motion_max_seq_length,
        )
    elif down_block_type == "CrossAttnDownBlockMotion":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlockMotion")
        return CrossAttnDownBlockMotion(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
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
            resnet_time_scale_shift=resnet_time_scale_shift,
            motion_norm_num_groups=motion_norm_num_groups,
            motion_cross_attention_dim=motion_cross_attention_dim,
            motion_num_attention_heads=motion_num_attention_heads,
            motion_attention_bias=motion_attention_bias,
            motion_activation_fn=motion_activation_fn,
            motion_max_seq_length=motion_max_seq_length,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    num_attention_heads,
    resolution_idx=None,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    motion_norm_num_groups=32,
    motion_cross_attention_dim=None,
    motion_num_attention_heads=8,
    motion_attention_bias=False,
    motion_activation_fn="geglu",
    motion_max_seq_length=32,
):
    if up_block_type == "UpBlockMotion":
        return UpBlockMotion(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            motion_norm_num_groups=motion_norm_num_groups,
            motion_cross_attention_dim=motion_cross_attention_dim,
            motion_num_attention_heads=motion_num_attention_heads,
            motion_attention_bias=motion_attention_bias,
            motion_activation_fn=motion_activation_fn,
            motion_max_seq_length=motion_max_seq_length,
        )
    elif up_block_type == "CrossAttnUpBlockMotion":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlockMotion")
        return CrossAttnUpBlockMotion(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
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
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            motion_norm_num_groups=motion_norm_num_groups,
            motion_cross_attention_dim=motion_cross_attention_dim,
            motion_num_attention_heads=motion_num_attention_heads,
            motion_attention_bias=motion_attention_bias,
            motion_activation_fn=motion_activation_fn,
            motion_max_seq_length=motion_max_seq_length,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class MotionAttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, in_channels, max_seq_length=32):
        super().__init__()
        self.pos_embed = PositionalEmbedding(embed_dim=in_channels, max_seq_length=max_seq_length)

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states
        hidden_states = self.pos_embed(hidden_states)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MotionAttnProcessor2_0(nn.Module):
    r"""
    Attention Processor for performing attention-related computations in the Motion Modules.
    """

    def __init__(self, in_channels, max_seq_length=32):
        super().__init__()
        self.pos_embed = PositionalEmbedding(embed_dim=in_channels, max_seq_length=max_seq_length)

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        hidden_states = self.pos_embed(hidden_states)

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
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MotionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        norm_num_groups=32,
        cross_attention_dim=None,
        activation_fn="geglu",
        attention_bias=False,
        num_attention_heads=8,
        max_seq_length=24,
    ) -> None:
        super().__init__()

        self.temporal_transformer = TransformerTemporalMotionModel(
            in_channels=in_channels,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=in_channels // num_attention_heads,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            num_attention_heads=num_attention_heads,
        )
        self.use_cross_attention = cross_attention_dim is not None

    def forward(self, hidden_states, encoder_hidden_states=None, num_frames=1, cross_attention_kwargs=None):
        encoder_hidden_states = encoder_hidden_states if self.use_cross_attention else None
        hidden_states = self.temporal_transformer(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            num_frames=num_frames,
        ).sample

        return hidden_states


class MotionModules(nn.Module):
    def __init__(
        self,
        in_channels,
        layers_per_block=2,
        num_attention_heads=8,
        attention_bias=False,
        cross_attention_dim=None,
        activation_fn="geglu",
        norm_num_groups=32,
        max_seq_length=32,
    ):
        super().__init__()
        self.motion_modules = nn.ModuleList([])

        for i in range(layers_per_block):
            self.motion_modules.append(
                MotionBlock(
                    in_channels=in_channels,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    num_attention_heads=num_attention_heads,
                    max_seq_length=max_seq_length,
                )
            )


class MotionAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        block_out_channels=(320, 640, 1280, 1280),
        motion_layers_per_block=2,
        motion_mid_block_layers_per_block=1,
        motion_num_attention_heads=8,
        motion_attention_bias=False,
        motion_cross_attention_dim=None,
        motion_activation_fn="geglu",
        motion_norm_num_groups=32,
        motion_max_seq_length=32,
    ):
        """Container to store AnimateDiff Motion Modules

        Args:
            block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each UNet block.
            layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
            num_attention_heads (`int`, *optional*):
            The number of heads to use in each attention layer.
            attention_bias (bool, optional, defaults to False): Whether to include bias in attention layers.
            cross_attention_dim (int, optional, Defaults to None): Set in order to use cross attention.
            activation_fn (str, optional, Defaults to "geglu"): Activation Function.
            norm_num_groups (int, optional): _description_. Defaults to 32.
            max_seq_length (int, optional): _description_. Defaults to 24.
        """

        super().__init__()
        down_blocks = []
        up_blocks = []

        for i, channel in enumerate(block_out_channels):
            output_channel = block_out_channels[i]
            down_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=motion_cross_attention_dim,
                    activation_fn=motion_activation_fn,
                    attention_bias=motion_attention_bias,
                    num_attention_heads=motion_num_attention_heads,
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=motion_layers_per_block,
                )
            )

        self.mid_block = MotionModules(
            in_channels=block_out_channels[-1],
            norm_num_groups=motion_norm_num_groups,
            cross_attention_dim=motion_cross_attention_dim,
            activation_fn=motion_activation_fn,
            attention_bias=motion_attention_bias,
            num_attention_heads=motion_num_attention_heads,
            layers_per_block=motion_mid_block_layers_per_block,
            max_seq_length=motion_max_seq_length,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, channel in enumerate(reversed_block_out_channels):
            output_channel = reversed_block_out_channels[i]
            up_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=motion_cross_attention_dim,
                    activation_fn=motion_activation_fn,
                    attention_bias=motion_attention_bias,
                    num_attention_heads=motion_num_attention_heads,
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=motion_layers_per_block + 1,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, sample):
        pass


class DownBlockMotion(nn.Module):
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
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        motion_norm_num_groups=32,
        motion_cross_attention_dim=None,
        motion_num_attention_heads=8,
        motion_attention_bias=False,
        motion_activation_fn="geglu",
        motion_max_seq_length=32,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = MotionModules(
            in_channels=out_channels,
            norm_num_groups=motion_norm_num_groups,
            cross_attention_dim=motion_cross_attention_dim,
            activation_fn=motion_activation_fn,
            attention_bias=motion_attention_bias,
            num_attention_heads=motion_num_attention_heads,
            max_seq_length=motion_max_seq_length,
            layers_per_block=num_layers,
        ).motion_modules

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, scale: float = 1.0, num_frames=1):
        output_states = ()

        blocks = zip(self.resnets, self.motion_modules)
        for resnet, motion_module in blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, scale
                    )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(motion_module), hidden_states.requires_grad_(), temb, num_frames
                )

            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)
                hidden_states = motion_module(hidden_states, num_frames=num_frames)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        attention_type="default",
        motion_norm_num_groups=32,
        motion_cross_attention_dim=None,
        motion_num_attention_heads=8,
        motion_attention_bias=False,
        motion_activation_fn="geglu",
        motion_max_seq_length=32,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
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
        self.motion_modules = MotionModules(
            in_channels=out_channels,
            norm_num_groups=motion_norm_num_groups,
            cross_attention_dim=motion_cross_attention_dim,
            activation_fn=motion_activation_fn,
            attention_bias=motion_attention_bias,
            num_attention_heads=motion_num_attention_heads,
            max_seq_length=motion_max_seq_length,
            layers_per_block=num_layers,
        ).motion_modules

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        encoder_attention_mask=None,
        cross_attention_kwargs=None,
        additional_residuals=None,
    ):
        output_states = ()

        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        blocks = list(zip(self.resnets, self.attentions, self.motion_modules))
        for i, (resnet, attn, motion_module) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = motion_module(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_frames=num_frames,
                )

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=lora_scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlockMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        attention_type="default",
        motion_norm_num_groups=32,
        motion_cross_attention_dim=None,
        motion_num_attention_heads=8,
        motion_attention_bias=False,
        motion_activation_fn="geglu",
        motion_max_seq_length=32,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
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
                )
            )

            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
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
        self.motion_modules = MotionModules(
            in_channels=out_channels,
            norm_num_groups=motion_norm_num_groups,
            cross_attention_dim=motion_cross_attention_dim,
            activation_fn=motion_activation_fn,
            attention_bias=motion_attention_bias,
            num_attention_heads=motion_num_attention_heads,
            max_seq_length=motion_max_seq_length,
            layers_per_block=num_layers,
        ).motion_modules

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        num_frames=1,
    ):
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        blocks = zip(self.resnets, self.attentions, self.motion_modules)
        for resnet, attn, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
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

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = motion_module(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_frames=num_frames,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)

        return hidden_states


class UpBlockMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        motion_norm_num_groups=32,
        motion_cross_attention_dim=None,
        motion_num_attention_heads=8,
        motion_attention_bias=False,
        motion_activation_fn="geglu",
        motion_max_seq_length=32,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
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
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = MotionModules(
            in_channels=out_channels,
            norm_num_groups=motion_norm_num_groups,
            cross_attention_dim=motion_cross_attention_dim,
            activation_fn=motion_activation_fn,
            attention_bias=motion_attention_bias,
            num_attention_heads=motion_num_attention_heads,
            max_seq_length=motion_max_seq_length,
            layers_per_block=num_layers,
        ).motion_modules

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, scale: float = 1.0, num_frames=1
    ):
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        blocks = zip(self.resnets, self.motion_modules)

        for resnet, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
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

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                )

            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)
                hidden_states = motion_module(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

        return hidden_states


class UNetMidBlockCrossAttnMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        attention_type="default",
        motion_norm_num_groups=32,
        motion_cross_attention_dim=None,
        motion_num_attention_heads=8,
        motion_attention_bias=False,
        motion_activation_fn="geglu",
        motion_max_seq_length=32,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = MotionModules(
            in_channels=in_channels,
            norm_num_groups=motion_norm_num_groups,
            cross_attention_dim=motion_cross_attention_dim,
            activation_fn=motion_activation_fn,
            attention_bias=motion_attention_bias,
            num_attention_heads=motion_num_attention_heads,
            max_seq_length=motion_max_seq_length,
            layers_per_block=num_layers,
        ).motion_modules

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        num_frames=1,
    ) -> torch.FloatTensor:
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)

        blocks = zip(self.attentions, self.resnets[1:], self.motion_modules)
        for attn, resnet, motion_module in blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(motion_module),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = motion_module(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_frames=num_frames,
                )
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)

        return hidden_states
