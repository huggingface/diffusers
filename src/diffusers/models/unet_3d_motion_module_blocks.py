# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py

import torch
from torch import nn

from .attention import Transformer3DModel  # TODO: create Transformer3DModel class
from .motion_module import get_motion_module

import pdb


def rearrange_0(tensor):
    b, c, f, h, w = tensor.size()
    tensor = tensor.view(b * f, c, h, w)
    return tensor  # b c f h w -> (b f) c h w


def rearrange_1(tensor, f):
    bf, c, h, w = tensor.size()
    b = bf // f
    tensor = tensor.view(b, f, c, h, w)

    return tensor  # (b f) c h w -> b c f h w


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange_0(x)
        x = super().forward(x)
        x = rearrange_1(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # if self.use_conv:
        #     if self.name == "conv":
        #         hidden_states = self.conv(hidden_states)
        #     else:
        #         hidden_states = self.Conv2d_0(hidden_states)
        hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            raise NotImplementedError

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
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
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
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
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
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
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
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
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
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
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
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
        motion_modules = []

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=in_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )
            resnets.append(
                ResnetBlock3D(
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
        self.motion_modules = nn.ModuleList(motion_modules)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet, motion_module in zip(self.attentions, self.resnets[1:], self.motion_modules):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
            hidden_states = (
                motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                if motion_module is not None
                else hidden_states
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
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
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
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
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None):
        output_states = ()

        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )[0]
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )

            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

                # add motion module
                hidden_states = (
                    motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                    if motion_module is not None
                    else hidden_states
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
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
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
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
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()

        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

                # add motion module
                hidden_states = (
                    motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                    if motion_module is not None
                    else hidden_states
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
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
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
    ):
        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )[0]
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )

            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

                # add motion module
                hidden_states = (
                    motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                    if motion_module is not None
                    else hidden_states
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
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
        add_upsample=True,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
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
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        encoder_hidden_states=None,
    ):
        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = (
                    motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                    if motion_module is not None
                    else hidden_states
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
