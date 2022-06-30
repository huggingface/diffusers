import string
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def conv_transpose_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.ConvTranspose1d(*args, **kwargs)
    elif dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def Normalize(in_channels, num_groups=32, eps=1e-6):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)


def nonlinearity(x, swish=1.0):
    # swish
    if swish == 1.0:
        return F.silu(x)
    else:
        return x * F.sigmoid(x * float(swish))


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.conv = conv_transpose_nd(dims, channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)

        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, dims=2, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.padding = padding
        stride = 2 if dims != 3 else (1, 2, 2)
        self.name = name

        if use_conv:
            conv = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = avg_pool_nd(dims, kernel_size=stride, stride=stride)

        if name == "conv":
            self.conv = conv
        else:
            self.op = conv

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv and self.padding == 0 and self.dims == 2:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)

        if self.name == "conv":
            return self.conv(x)
        else:
            return self.op(x)


# TODO (patil-suraj): needs test
# class Upsample1d(nn.Module):
#    def __init__(self, dim):
#        super().__init__()
#        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
#
#    def forward(self, x):
#        return self.conv(x)


# RESNETS

# unet_glide.py
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels. :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout. :param out_channels: if specified, the number of out channels. :param
    use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D. :param use_checkpoint: if True, use gradient checkpointing
    on this module. :param up: if True, use this block for upsampling. :param down: if True, use this block for
    downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        overwrite=False,  # TODO(Patrick) - use for glide at later stage
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0),
            nn.Identity(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, use_conv=False, dims=dims)
            self.x_upd = Upsample(channels, use_conv=False, dims=dims)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False, dims=dims, padding=1, name="op")
            self.x_upd = Downsample(channels, use_conv=False, dims=dims, padding=1, name="op")
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.overwrite = overwrite
        self.is_overwritten = False
        if self.overwrite:
            in_channels = channels
            out_channels = self.out_channels
            conv_shortcut = False
            dropout = 0.0
            temb_channels = emb_channels
            groups = 32
            pre_norm = True
            eps = 1e-5
            non_linearity = "silu"
            self.pre_norm = pre_norm
            self.in_channels = in_channels
            out_channels = in_channels if out_channels is None else out_channels
            self.out_channels = out_channels
            self.use_conv_shortcut = conv_shortcut

            if self.pre_norm:
                self.norm1 = Normalize(in_channels, num_groups=groups, eps=eps)
            else:
                self.norm1 = Normalize(out_channels, num_groups=groups, eps=eps)

            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
            self.norm2 = Normalize(out_channels, num_groups=groups, eps=eps)
            self.dropout = torch.nn.Dropout(dropout)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if non_linearity == "swish":
                self.nonlinearity = nonlinearity
            elif non_linearity == "mish":
                self.nonlinearity = Mish()
            elif non_linearity == "silu":
                self.nonlinearity = nn.SiLU()

            if self.in_channels != self.out_channels:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def set_weights(self):
        # TODO(Patrick): use for glide at later stage
        self.norm1.weight.data = self.in_layers[0].weight.data
        self.norm1.bias.data = self.in_layers[0].bias.data

        self.conv1.weight.data = self.in_layers[-1].weight.data
        self.conv1.bias.data = self.in_layers[-1].bias.data

        self.temb_proj.weight.data = self.emb_layers[-1].weight.data
        self.temb_proj.bias.data = self.emb_layers[-1].bias.data

        self.norm2.weight.data = self.out_layers[0].weight.data
        self.norm2.bias.data = self.out_layers[0].bias.data

        self.conv2.weight.data = self.out_layers[-1].weight.data
        self.conv2.bias.data = self.out_layers[-1].bias.data

        if self.in_channels != self.out_channels:
            self.nin_shortcut.weight.data = self.skip_connection.weight.data
            self.nin_shortcut.bias.data = self.skip_connection.bias.data

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features. :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.overwrite:
            # TODO(Patrick): use for glide at later stage
            self.set_weights()

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        result = self.skip_connection(x) + h

        # TODO(Patrick) Use for glide at later stage
        #        result = self.forward_2(x, emb)

        return result

    def forward_2(self, x, temb, mask=1.0):
        if self.overwrite and not self.is_overwritten:
            self.set_weights()
            self.is_overwritten = True

        h = x
        if self.pre_norm:
            h = self.norm1(h)
            h = self.nonlinearity(h)

        h = self.conv1(h)

        if not self.pre_norm:
            h = self.norm1(h)
            h = self.nonlinearity(h)

        h = h + self.temb_proj(self.nonlinearity(temb))[:, :, None, None]

        if self.pre_norm:
            h = self.norm2(h)
            h = self.nonlinearity(h)

        h = self.dropout(h)
        h = self.conv2(h)

        if not self.pre_norm:
            h = self.norm2(h)
            h = self.nonlinearity(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


# unet.py and unet_grad_tts.py
class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        overwrite_for_grad_tts=False,
        overwrite_for_ldm=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if self.pre_norm:
            self.norm1 = Normalize(in_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = Normalize(out_channels, num_groups=groups, eps=eps)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, num_groups=groups, eps=eps)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if non_linearity == "swish":
            self.nonlinearity = nonlinearity
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # TODO(Patrick) - this branch is never used I think => can be deleted!
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.is_overwritten = False
        self.overwrite_for_grad_tts = overwrite_for_grad_tts
        self.overwrite_for_ldm = overwrite_for_ldm
        if self.overwrite_for_grad_tts:
            dim = in_channels
            dim_out = out_channels
            time_emb_dim = temb_channels
            self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))
            self.pre_norm = pre_norm

            self.block1 = Block(dim, dim_out, groups=groups)
            self.block2 = Block(dim_out, dim_out, groups=groups)
            if dim != dim_out:
                self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
            else:
                self.res_conv = torch.nn.Identity()
        elif self.overwrite_for_ldm:
            dims = 2
            #            eps = 1e-5
            #            non_linearity = "silu"
            #            overwrite_for_ldm
            channels = in_channels
            emb_channels = temb_channels
            use_scale_shift_norm = False

            self.in_layers = nn.Sequential(
                normalization(channels, swish=1.0),
                nn.Identity(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
                nn.SiLU() if use_scale_shift_norm else nn.Identity(),
                nn.Dropout(p=dropout),
                zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
            )
            if self.out_channels == in_channels:
                self.skip_connection = nn.Identity()
            #            elif use_conv:
            #                self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def set_weights_grad_tts(self):
        self.conv1.weight.data = self.block1.block[0].weight.data
        self.conv1.bias.data = self.block1.block[0].bias.data
        self.norm1.weight.data = self.block1.block[1].weight.data
        self.norm1.bias.data = self.block1.block[1].bias.data

        self.conv2.weight.data = self.block2.block[0].weight.data
        self.conv2.bias.data = self.block2.block[0].bias.data
        self.norm2.weight.data = self.block2.block[1].weight.data
        self.norm2.bias.data = self.block2.block[1].bias.data

        self.temb_proj.weight.data = self.mlp[1].weight.data
        self.temb_proj.bias.data = self.mlp[1].bias.data

        if self.in_channels != self.out_channels:
            self.nin_shortcut.weight.data = self.res_conv.weight.data
            self.nin_shortcut.bias.data = self.res_conv.bias.data

    def set_weights_ldm(self):
        self.norm1.weight.data = self.in_layers[0].weight.data
        self.norm1.bias.data = self.in_layers[0].bias.data

        self.conv1.weight.data = self.in_layers[-1].weight.data
        self.conv1.bias.data = self.in_layers[-1].bias.data

        self.temb_proj.weight.data = self.emb_layers[-1].weight.data
        self.temb_proj.bias.data = self.emb_layers[-1].bias.data

        self.norm2.weight.data = self.out_layers[0].weight.data
        self.norm2.bias.data = self.out_layers[0].bias.data

        self.conv2.weight.data = self.out_layers[-1].weight.data
        self.conv2.bias.data = self.out_layers[-1].bias.data

        if self.in_channels != self.out_channels:
            self.nin_shortcut.weight.data = self.skip_connection.weight.data
            self.nin_shortcut.bias.data = self.skip_connection.bias.data

    def forward(self, x, temb, mask=1.0):
        if self.overwrite_for_grad_tts and not self.is_overwritten:
            self.set_weights_grad_tts()
            self.is_overwritten = True
        elif self.overwrite_for_ldm and not self.is_overwritten:
            self.set_weights_ldm()
            self.is_overwritten = True

        h = x
        h = h * mask
        if self.pre_norm:
            h = self.norm1(h)
            h = self.nonlinearity(h)

        h = self.conv1(h)

        if not self.pre_norm:
            h = self.norm1(h)
            h = self.nonlinearity(h)
        h = h * mask

        h = h + self.temb_proj(self.nonlinearity(temb))[:, :, None, None]

        h = h * mask
        if self.pre_norm:
            h = self.norm2(h)
            h = self.nonlinearity(h)

        h = self.dropout(h)
        h = self.conv2(h)

        if not self.pre_norm:
            h = self.norm2(h)
            h = self.nonlinearity(h)
        h = h * mask

        x = x * mask
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


# TODO(Patrick) - just there to convert the weights; can delete afterward
class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1), torch.nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


# unet_score_estimation.py
class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
        skip_rescale=True,
        init_scale=0.0,
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            if self.fir:
                h = upsample_2d(h, self.fir_kernel, factor=2)
                x = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_2d(h, self.fir_kernel, factor=2)
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_downsample_2d(h, factor=2)
                x = naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


# unet_score_estimation.py
class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        skip_rescale=False,
        init_scale=0.0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


# unet_rl.py
class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            RearrangeDim(),
            #            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ] t : [ batch_size x embed_dim ] returns: out : [ batch_size x
        out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# HELPER Modules


def normalization(channels, swish=0.0):
    """
    Make a standard normalization layer, with an optional swish activation.

    :param channels: number of input channels. :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=32, swish=swish)


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.swish = swish

    def forward(self, x):
        y = super().forward(x.float()).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            RearrangeDim(),
            #            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            RearrangeDim(),
            #            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class RearrangeDim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        if len(tensor.shape) == 2:
            return tensor[:, :, None]
        if len(tensor.shape) == 3:
            return tensor[:, :, None, :]
        elif len(tensor.shape) == 4:
            return tensor[:, :, 0, :]
        else:
            raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"):
    """Ported from JAX."""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError("invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


def upsample_2d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 2D images with the given filter.

    Args:
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is a:
    multiple of the upsampling factor.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 2D images with the given filter.

    Args:
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2))


def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)
