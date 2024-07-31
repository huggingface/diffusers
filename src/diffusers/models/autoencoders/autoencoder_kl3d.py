from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


## == Basic Block of 3D VAE Model design in CogVideoX === ###


## Draft of block
# class DownEncoderBlock3D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         pad_mode: str = "first",
#     ):
#         super().__init__()
#         resnets = []
#
#         for i in range(num_layers):
#             resnets.append(
#                 CogVideoXResnetBlock3D(
#                     in_channels=in_channels if i == 0 else out_channels,
#                     out_channels=out_channels,
#                     temb_channels=0,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     non_linearity=resnet_act_fn,
#                     conv_shortcut=resnet_pre_norm,
#                     pad_mode=pad_mode,
#                 )
#             )
#             in_channels = out_channels
#
#         self.resnets = nn.ModuleList(resnets)
#         self.downsampler = DownSample3D(in_channels=out_channels, out_channels=out_channels) if num_layers > 0 else None
#
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         for resnet in self.resnets:
#             hidden_states = resnet(hidden_states, temb=None)
#
#         if self.downsampler is not None:
#             hidden_states = self.downsampler(hidden_states)
#
#         return hidden_states
#
#
# class Encoder3D(nn.Module):
#     def __init__(
#             self,
#             in_channels: int = 3,
#             out_channels: int = 16,
#             down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
#             block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
#             layers_per_block: int = 3,
#             act_fn: str = "silu",
#             norm_num_groups: int = 32,
#             dropout: float = 0.0,
#             resolution: int = 256,
#             double_z: bool = True,
#             pad_mode: str = "first",
#             temporal_compress_times: int = 4,
#     ):
#         super().__init__()
#         self.act_fn = get_activation(act_fn)
#         self.num_resolutions = len(block_out_channels)
#         self.layers_per_block = layers_per_block
#         self.resolution = resolution
#
#         # log2 of temporal_compress_times
#         self.temporal_compress_level = int(np.log2(temporal_compress_times))
#
#         self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
#
#         self.down_blocks = nn.ModuleList()
#         self.downsamples = nn.ModuleList()
#
#         for i_level in range(self.num_resolutions):
#             block_in = block_out_channels[i_level - 1] if i_level > 0 else block_out_channels[0]
#             block_out = block_out_channels[i_level]
#             is_final_block = i_level == self.num_resolutions - 1
#
#             down_block = DownEncoderBlock3D(
#                 in_channels=block_in,
#                 out_channels=block_out,
#                 num_layers=self.layers_per_block,
#                 dropout=dropout,
#                 resnet_eps=1e-6,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 resnet_pre_norm=True,
#                 pad_mode=pad_mode,
#             )
#             self.down_blocks.append(down_block)
#
#             if not is_final_block:
#                 compress_time = i_level < self.temporal_compress_level
#                 self.downsamples.append(
#                     DownSample3D(in_channels=block_out, out_channels=block_out, compress_time=compress_time)
#                 )
#
#         # middle
#         block_in = block_out_channels[-1]
#         self.mid_block_1 = CogVideoXResnetBlock3D(
#             in_channels=block_in,
#             out_channels=block_in,
#             non_linearity=act_fn,
#             temb_channels=0,
#             groups=norm_num_groups,
#             dropout=dropout,
#             pad_mode=pad_mode,
#         )
#         self.mid_block_2 = CogVideoXResnetBlock3D(
#             in_channels=block_in,
#             out_channels=block_in,
#             non_linearity=act_fn,
#             temb_channels=0,
#             groups=norm_num_groups,
#             dropout=dropout,
#             pad_mode=pad_mode,
#         )
#
#         # out
#         self.norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
#         self.conv_act = get_activation(act_fn)
#
#         conv_out_channels = 2 * out_channels if double_z else out_channels
#         self.conv_out = CogVideoXCausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3, pad_mode=pad_mode)
#
#     def forward(self, sample: torch.Tensor) -> torch.Tensor:
#         temb = None
#
#         # DownSampling
#         sample = self.conv_in(sample)
#         for i_level in range(self.num_resolutions):
#             sample = self.down_blocks[i_level](sample)
#             if i_level < len(self.downsamples):
#                 sample = self.downsamples[i_level](sample)
#
#         sample = self.mid_block_1(sample, temb)
#         sample = self.mid_block_2(sample, temb)
#
#         # post-process
#         sample = self.norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)
#
#         return sample


# Todo: zRzRzRzRzRzRzR Move it to cogvideox model file since pr#2 has been merged
class CogVideoXSaveConv3d(nn.Conv3d):
    """
    A 3D convolution layer that splits the input tensor into smaller parts to avoid OOM in CogVideoX Model.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3

        # Set to 2GB, suitable for CuDNN
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input, part_num, dim=2)

            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


# Todo: zRzRzRzRzRzRzR Move it to cogvideox model file since pr#2 has been merged
class CogVideoXCausalConv3d(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        def cast_tuple(t, length=1):
            return t if isinstance(t, tuple) else ((t,) * length)

        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = CogVideoXSaveConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.conv_cache = None

    def forward(self, x):
        if self.pad_mode == "constant":
            causal_padding_3d = (self.time_pad, 0, self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_3d, mode="constant", value=0)
        elif self.pad_mode == "first":
            pad_x = torch.cat([x[:, :, :1]] * self.time_pad, dim=2)
            x = torch.cat([pad_x, x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode="constant", value=0)
        elif self.pad_mode == "reflect":
            reflect_x = x[:, :, 1 : self.time_pad + 1, :, :].flip(dims=[2])
            if reflect_x.shape[2] < self.time_pad:
                reflect_x = torch.cat(
                    [torch.zeros_like(x[:, :, :1, :, :])] * (self.time_pad - reflect_x.shape[2]) + [reflect_x], dim=2
                )
            x = torch.cat([reflect_x, x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode="constant", value=0)
        else:
            raise ValueError("Invalid pad mode")
        if self.time_pad != 0 and self.conv_cache is None:
            self.conv_cache = x[:, :, -self.time_pad :].detach().clone().cpu()
            return self.conv(x)
        elif self.time_pad != 0 and self.conv_cache is not None:
            x = torch.cat([self.conv_cache.to(x.device), x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode="constant", value=0)
            self.conv_cache = None
            return self.conv(x)

        return self.conv(x)


# Todo: zRzRzRzRzRzRzR Move it to cogvideox model file since pr#2 has been merged
class CogVideoXSpatialNorm3D(nn.Module):
    r"""
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002. This implementation is specific
    to 3D-video like data.

    CogVideoXSaveConv3d is used instead of nn.Conv3d to avoid OOM in CogVideoX Model.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv = CogVideoXCausalConv3d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=0)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if zq.shape[2] > 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])
            zq = self.conv(zq)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class CogVideoXResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.non_linearity = get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
            )
        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels, out_features=out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
                )
            else:
                self.nin_shortcut = CogVideoXSaveConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(
        self, input_tensor: torch.Tensor, temb: torch.Tensor, zq: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> torch.Tensor:
        hidden_states = input_tensor
        if zq is not None:
            hidden_states = self.norm1(hidden_states, zq)
        else:
            hidden_states = self.norm1(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.non_linearity(temb))[:, :, None, None, None]

        if zq is not None:
            hidden_states = self.norm2(hidden_states, zq)
        else:
            hidden_states = self.norm2(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                input_tensor = self.conv_shortcut(input_tensor)
            else:
                input_tensor = self.nin_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


# Todo: zRzRzRzRzRzRzR Move it to cogvideox model file since pr#2 has been merged
class CogVideoXUpSample3D(nn.Module):
    r"""
    Add compress_time option to the `UpSample` layer of a variational autoencoder that upsamples its input in CogVideoX
    Model.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        compress_time (`bool`, *optional*, defaults to `False`):
            Whether to compress the time dimension.
    """

    def __init__(self, in_channels: int, out_channels: int, compress_time: bool = False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            if x.shape[2] > 1 and x.shape[2] % 2 == 1:
                # split first frame
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                x_first = F.interpolate(x_first, scale_factor=2.0)
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                x = torch.cat([x_first, x_rest], dim=2)
            elif x.shape[2] > 1:
                x = F.interpolate(x, scale_factor=2.0)
            else:
                x = x.squeeze(2)
                x = F.interpolate(x, scale_factor=2.0)
                x = x[:, :, None, :, :]
        else:
            # only interpolate 2D
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.interpolate(x, scale_factor=2.0)
            x = x.reshape(b, t, c, *x.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)

        return x


# Todo: Create vae_3d.py such as vae.py file?
class CogVideoXDownSample3D(nn.Module):
    r"""
    Add compress_time option to the `DownSample` layer of a variational autoencoder that downsamples its input in
    CogVideoX Model.

    Args:
        in_channels (`int`, *optional*):
            The number of input channels.
        out_channels (`int`, *optional*):
            The number of output channels.
        compress_time (`bool`, *optional*, defaults to `False`):
            Whether to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            b, c, t, h, w = x.shape
            x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)

            if x.shape[-1] % 2 == 1:
                # split first frame
                x_first, x_rest = x[..., 0], x[..., 1:]

                if x_rest.shape[-1] > 0:
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = x.reshape(b, h, w, c, x.shape[-1]).permute(0, 3, 4, 1, 2)

            else:
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                x = x.reshape(b, h, w, c, x.shape[-1]).permute(0, 3, 4, 1, 2)

        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)

        return x


class Encoder3D(nn.Module):
    r"""
    The `Encoder3D` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        pad_mode: str = "first",
        temporal_compress_times: int = 4,
    ):
        super().__init__()
        self.act_fn = get_activation(act_fn)
        self.num_resolutions = len(block_out_channels)
        self.layers_per_block = layers_per_block
        self.resolution = resolution

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)

        curr_res = resolution
        in_ch_mult = (block_out_channels[0],) + tuple(block_out_channels)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()

            block_in = in_ch_mult[i_level]
            block_out = block_out_channels[i_level]

            for i_block in range(self.layers_per_block):
                block.append(
                    CogVideoXResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        non_linearity=act_fn,
                        dropout=dropout,
                        groups=norm_num_groups,
                        pad_mode=pad_mode,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block

            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = CogVideoXDownSample3D(
                        in_channels=block_in, out_channels=block_in, compress_time=True
                    )
                else:
                    down.downsample = CogVideoXDownSample3D(
                        in_channels=block_in, out_channels=block_in, compress_time=False
                    )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        block_in = in_ch_mult[-1]
        self.mid.block_1 = CogVideoXResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            non_linearity=act_fn,
            temb_channels=0,
            groups=norm_num_groups,
            dropout=dropout,
            pad_mode=pad_mode,
        )
        self.mid.block_2 = CogVideoXResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            non_linearity=act_fn,
            temb_channels=0,
            groups=norm_num_groups,
            dropout=dropout,
            pad_mode=pad_mode,
        )

        self.norm_out = nn.GroupNorm(num_channels=block_in, num_groups=norm_num_groups, eps=1e-6)
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CogVideoXCausalConv3d(
            block_in, conv_out_channels if double_z else out_channels, kernel_size=3, pad_mode=pad_mode
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # timestep embedding

        temb = None

        # DownSampling
        sample = self.conv_in(sample)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.layers_per_block):
                sample = self.down[i_level].block[i_block](sample, temb)

            if i_level != self.num_resolutions - 1:
                sample = self.down[i_level].downsample(sample)

        # middle
        sample = self.mid.block_1(sample, temb)
        sample = self.mid.block_2(sample, temb)

        # post-process
        sample = self.norm_out(sample)
        sample = self.act_fn(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder3D(nn.Module):
    r"""
    The `Decoder3D` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        dropout: float = 0.0,
        resolution: int = 256,
        give_pre_end: bool = False,
        pad_mode: str = "first",
        temporal_compress_times: int = 4,
        norm_num_groups=32,
    ):
        super().__init__()

        self.act_fn = get_activation(act_fn)
        self.num_resolutions = len(block_out_channels)
        self.layers_per_block = layers_per_block
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.norm_num_groups = norm_num_groups
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        block_in = block_out_channels[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, in_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_in, kernel_size=3, pad_mode=pad_mode)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CogVideoXResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
            non_linearity=act_fn,
            spatial_norm_dim=in_channels,
            groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        self.mid.block_2 = CogVideoXResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
            non_linearity=act_fn,
            spatial_norm_dim=in_channels,
            groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        # UpSampling

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()

            block_out = block_out_channels[i_level]
            for i_block in range(self.layers_per_block + 1):
                block.append(
                    CogVideoXResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        non_linearity=act_fn,
                        dropout=dropout,
                        spatial_norm_dim=in_channels,
                        groups=norm_num_groups,
                        pad_mode=pad_mode,
                    )
                )
                block_in = block_out

            up = nn.Module()
            up.block = block

            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = CogVideoXUpSample3D(in_channels=block_in, out_channels=block_in, compress_time=False)
                else:
                    up.upsample = CogVideoXUpSample3D(in_channels=block_in, out_channels=block_in, compress_time=True)
                curr_res = curr_res * 2

            self.up.insert(0, up)

        self.norm_out = CogVideoXSpatialNorm3D(f_channels=block_in, zq_channels=in_channels)

        self.conv_out = CogVideoXCausalConv3d(block_in, out_channels, kernel_size=3, pad_mode=pad_mode)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""
        # timestep embedding

        temb = None

        hidden_states = self.conv_in(sample)

        # middle
        hidden_states = self.mid.block_1(hidden_states, temb, sample)

        hidden_states = self.mid.block_2(hidden_states, temb, sample)

        # UpSampling

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.layers_per_block + 1):
                hidden_states = self.up[i_level].block[i_block](hidden_states, temb, sample)

            if i_level != 0:
                hidden_states = self.up[i_level].upsample(hidden_states)

        # end
        if self.give_pre_end:
            return hidden_states

        hidden_states = self.norm_out(hidden_states, sample)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class AutoencoderKL3D(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encodfing images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        mid_block_add_attention (`bool`, *optional*, default to `True`):
            If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
            mid_block will only have resnet blocks
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        sample_size: int = 256,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            resolution=sample_size,
        )
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            resolution=sample_size,
        )
        self.quant_conv = CogVideoXSaveConv3d(2 * out_channels, 2 * out_channels, 1) if use_quant_conv else None
        self.post_quant_conv = CogVideoXSaveConv3d(out_channels, out_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder3D, Decoder3D)):
            module.gradient_checkpointing = value

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if not return_dict:
            return (dec,)
        return dec
