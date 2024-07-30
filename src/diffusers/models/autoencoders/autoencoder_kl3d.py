import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Callable, Optional, Tuple, Union
from einops import rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


def normalize3d(in_channels, z_ch, add_conv):
    return SpatialNorm3D(
        in_channels,
        z_ch,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True
    )


class SafeConv3d(torch.nn.Conv3d):
    def forward(self, input):
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024 ** 3
        if memory_count > 2:  # Set to 2GB
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input, part_num, dim=2)  # NCTHW
            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1:], input_chunks[i]), dim=2) for i in
                    range(1, len(input_chunks))]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super(SafeConv3d, self).forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super(SafeConv3d, self).forward(input)


class OriginCausalConv3d(nn.Module):
    @beartype
    def __init__(
            self,
            chan_in,
            chan_out,
            kernel_size: Union[int, Tuple[int, int, int]],
            pad_mode='constant',
            **kwargs
    ):
        super().__init__()

        def cast_tuple(t, length=1):
            return t if isinstance(t, tuple) else ((t,) * length)

        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)

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
        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        if self.pad_mode == 'constant':
            causal_padding_3d = (self.time_pad, 0, self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_3d, mode='constant', value=0)
        elif self.pad_mode == 'first':
            pad_x = torch.cat([x[:, :, :1]] * self.time_pad, dim=2)
            x = torch.cat([pad_x, x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode='constant', value=0)
        elif self.pad_mode == 'reflect':
            # reflect padding
            reflect_x = x[:, :, 1:self.time_pad + 1, :, :].flip(dims=[2])
            if reflect_x.shape[2] < self.time_pad:
                reflect_x = torch.cat(
                    [torch.zeros_like(x[:, :, :1, :, :])] * (self.time_pad - reflect_x.shape[2]) + [reflect_x], dim=2)
            x = torch.cat([reflect_x, x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode='constant', value=0)
        else:
            raise ValueError("Invalid pad mode")
        return self.conv(x)


class CausalConv3d(OriginCausalConv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_cache = None

    def forward(self, x):
        if self.time_pad == 0:
            return super().forward(x)
        if self.conv_cache is None:
            self.conv_cache = x[:, :, -self.time_pad:].detach().clone().cpu()
            return super().forward(x)
        else:
            # print(self.conv_cache.shape, x.shape)
            x = torch.cat([self.conv_cache.to(x.device), x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode='constant', value=0)
            self.conv_cache = None
            return self.conv(x)


class SpatialNorm3D(nn.Module):
    def __init__(self, f_channels, z_channels, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=False,
                 pad_mode='constant', **norm_layer_params):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = CausalConv3d(z_channels, z_channels, kernel_size=3, pad_mode=pad_mode)

        self.conv_y = CausalConv3d(z_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
        self.conv_b = CausalConv3d(z_channels, f_channels, kernel_size=1, pad_mode=pad_mode)

    def forward(self, f, z):
        if z.shape[2] > 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = z[:, :, :1], z[:, :, 1:]
            z_first = torch.nn.functional.interpolate(z_first, size=f_first_size, mode="nearest")
            z_rest = torch.nn.functional.interpolate(z_rest, size=f_rest_size, mode="nearest")
            z = torch.cat([z_first, z_rest], dim=2)
        else:
            z = torch.nn.functional.interpolate(z, size=f.shape[-3:], mode="nearest")
        if self.add_conv:
            z = self.conv(z)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(z) + self.conv_b(z)
        return new_f


class UpSample3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            with_conv: bool,
            compress_time: bool = False
    ):
        super(UpSample3D, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            if x.shape[2] > 1 and x.shape[2] % 2 == 1:
                # split first frame
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0, mode="nearest")
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0, mode="nearest")
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            elif x.shape[2] > 1:
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            else:
                x = x.squeeze(2)
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
                x = x[:, :, None, :, :]
        else:
            # only interpolate 2D
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.with_conv:
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.conv(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class DownSample3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            with_conv: bool = False,
            compress_time: bool = False,
            out_channels: Optional[int] = None
    ):
        super(DownSample3D, self).__init__()
        self.with_conv = with_conv
        if out_channels is None:
            out_channels = in_channels
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=0
            )
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            h, w = x.shape[-2:]
            x = rearrange(x, 'b c t h w -> (b h w) c t')

            if x.shape[-1] % 2 == 1:
                # split first frame
                x_first, x_rest = x[..., 0], x[..., 1:]

                if x_rest.shape[-1] > 0:
                    x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=2, stride=2)
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = rearrange(x, '(b h w) c t -> b c t h w', h=h, w=w)
            else:
                x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
                x = rearrange(x, '(b h w) c t -> b c t h w', h=h, w=w)

        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.conv(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        else:
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
            self,
            *,
            in_channels: int,
            out_channels: int,
            conv_shortcut: bool = False,
            dropout: float,
            act_fn: str = "silu",
            temb_channels: int = 512,
            z_ch: Optional[int] = None,
            add_conv: bool = False,
            pad_mode: str = 'constant',
            norm_num_groups: int = 32,
            normalization: Callable = None
    ):
        super(ResnetBlock3D, self).__init__()
        self.in_channels = in_channels
        self.act_fn = get_activation(act_fn)
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if normalization is None:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=norm_num_groups, eps=1e-6)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6)
        else:
            self.norm1 = normalization(
                in_channels,
                z_ch=z_ch,
                add_conv=add_conv,
            )
            self.norm2 = normalization(
                out_channels,
                z_ch=z_ch,
                add_conv=add_conv
            )

        self.conv1 = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            pad_mode=pad_mode
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            else:
                self.nin_shortcut = SafeConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )

    def forward(self, x, temb, z=None):
        h = x
        if z is not None:
            h = self.norm1(h, z)
        else:
            h = self.norm1(h)
        h = self.act_fn(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.act_fn(temb))[:, :, None, None, None]

        if z is not None:
            h = self.norm2(h, z)
        else:
            h = self.norm2(h)
        h = self.act_fn(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock2D(nn.Module):
    def __init__(self, in_channels, norm_num_groups):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=norm_num_groups, eps=1e-6)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

        t = h_.shape[2]
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw

        # # original version, nan in fp16
        # w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c)**(-0.5))
        # # implement c**-0.5 on q

        q = q * (int(c) ** (-0.5))
        w_ = torch.bmm(q, k)
        # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)

        return x + h_


class Encoder3D(nn.Module):
    def __init__(
            self,
            *,
            ch: int,
            in_channels: int = 3,
            out_channels: int = 16,
            ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
            num_res_blocks: int,
            act_fn: str = "silu",
            norm_num_groups: int = 32,
            attn_resolutions=None,
            dropout: float = 0.0,
            resamp_with_conv: bool = True,
            resolution: int,
            z_channels: int,
            double_z: bool = True,
            pad_mode: str = 'first',
            temporal_compress_times: int = 4,
    ):
        super(Encoder3D, self).__init__()
        if attn_resolutions is None:
            attn_resolutions = []
        self.act_fn = get_activation(act_fn)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.attn_resolutions = attn_resolutions

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        self.conv_in = CausalConv3d(in_channels, self.ch, kernel_size=3, pad_mode=pad_mode)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        act_fn=act_fn,
                        dropout=dropout,
                        norm_num_groups=norm_num_groups,
                        pad_mode=pad_mode
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        AttnBlock2D(block_in)
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=True)
                else:
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=False)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            act_fn=act_fn,
            temb_channels=0,
            norm_num_groups=norm_num_groups,
            dropout=dropout, pad_mode=pad_mode
        )
        if len(attn_resolutions) > 0:
            self.mid.attn_1 = AttnBlock2D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            act_fn=act_fn,
            temb_channels=0,
            norm_num_groups=norm_num_groups,
            dropout=dropout, pad_mode=pad_mode
        )

        self.norm_out = nn.GroupNorm(num_channels=block_in, num_groups=norm_num_groups, eps=1e-6)
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(
            block_in, conv_out_channels if double_z else z_channels,
            kernel_size=3,
            pad_mode=pad_mode
        )

    def forward(self, x):

        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)

        if len(self.attn_resolutions):
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = self.act_fn(h)
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    def __init__(
            self, *,
            ch: int,
            in_channels: int = 16,
            out_channels: int = 3,
            ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
            num_res_blocks: int,
            attn_resolutions=None,
            act_fn: str = "silu",
            dropout: float = 0.0,
            resamp_with_conv: bool = True,
            resolution: int,
            z_channels: int,
            give_pre_end: bool = False,
            z_ch: Optional[int] = None,
            add_conv: bool = False,
            pad_mode: str = 'first',
            temporal_compress_times: int = 4,
            norm_num_groups=32,
    ):
        super(Decoder3D, self).__init__()
        if attn_resolutions is None:
            attn_resolutions = []
        self.ch = ch
        self.act_fn = get_activation(act_fn)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.attn_resolutions = attn_resolutions
        self.norm_num_groups = norm_num_groups

        # log2 of temporal_compress_times

        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        if z_ch is None:
            z_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, pad_mode=pad_mode)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
            act_fn=act_fn,
            z_ch=z_ch,
            add_conv=add_conv,
            normalization=normalize3d,
            norm_num_groups=norm_num_groups,
            pad_mode=pad_mode
        )
        if len(attn_resolutions) > 0:
            self.mid.attn_1 = AttnBlock2D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
            act_fn=act_fn,
            z_ch=z_ch,
            add_conv=add_conv,
            normalization=normalize3d,
            norm_num_groups=norm_num_groups,
            pad_mode=pad_mode
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        act_fn=act_fn,
                        dropout=dropout,
                        z_ch=z_ch,
                        add_conv=add_conv,
                        normalization=normalize3d,
                        norm_num_groups=norm_num_groups,
                        pad_mode=pad_mode
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        AttnBlock2D
                        (block_in=block_in, norm_num_groups=norm_num_groups)
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = UpSample3D(block_in, resamp_with_conv, compress_time=False)
                else:
                    up.upsample = UpSample3D(block_in, resamp_with_conv, compress_time=True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = normalize3d(block_in, z_ch, add_conv=add_conv)

        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3, pad_mode=pad_mode)

    def forward(self, z):

        # timestep embedding
        temb = None

        # z to block_in

        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, z)
        if len(self.attn_resolutions) > 0:
            h = self.mid.attn_1(h, z)
        h = self.mid.block_2(h, temb, z)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, z)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, z)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, z)
        h = self.act_fn(h)
        h = self.conv_out(h)

        return h


class AutoencoderKL3D(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
       A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

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
           latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
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
    _no_split_modules = ["ResnetBlock3D"]

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = ("DownEncoderBlock3D",),
            up_block_types: Tuple[str] = ("UpDecoderBlock3D",),
            ch: int = 128,
            block_out_channels: Tuple[int] = (1, 2, 2, 4),
            layers_per_block: int = 3,
            act_fn: str = "silu",
            latent_channels: int = 16,
            norm_num_groups: int = 32,
            sample_size: int = 256,

            # Do Not Know how to use
            scaling_factor: float = 0.13025,
            shift_factor: Optional[float] = None,
            latents_mean: Optional[Tuple[float]] = None,
            latents_std: Optional[Tuple[float]] = None,
            force_upcast: float = True,
            use_quant_conv: bool = False,
            use_post_quant_conv: bool = False,
            mid_block_add_attention: bool = True
    ):
        super().__init__()

        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            ch_mult=block_out_channels,
            ch=ch,
            num_res_blocks=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            resolution=sample_size,
            z_channels=latent_channels,
        )
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            ch=ch,
            ch_mult=block_out_channels,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            num_res_blocks=layers_per_block,
            resolution=sample_size,
            z_channels=latent_channels,
        )
        self.quant_conv = SafeConv3d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = SafeConv3d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        # self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
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
            generator: Optional[torch.Generator] = None
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
