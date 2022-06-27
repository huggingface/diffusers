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


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x, swish=1.0):
    # swish
    if swish == 1.0:
        return F.silu(x)
    else:
        return x * F.sigmoid(x * float(swish))


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
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

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
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
class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


# class ResnetBlock(nn.Module):
#     def __init__(
#         self,
#         *,
#         in_channels,
#         out_channels=None,
#         conv_shortcut=False,
#         dropout,
#         temb_channels=512,
#         use_scale_shift_norm=False,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut
#         self.use_scale_shift_norm = use_scale_shift_norm

#         self.norm1 = Normalize(in_channels)
#         self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

#         temp_out_channles = 2 * out_channels if use_scale_shift_norm else out_channels
#         self.temb_proj = torch.nn.Linear(temb_channels, temp_out_channles)

#         self.norm2 = Normalize(out_channels)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#             else:
#                 self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, x, temb):
#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)

#         # TODO: check if this broadcasting works correctly for 1D and 3D
#         temb = self.temb_proj(nonlinearity(temb))[:, :, None, None]

#         if self.use_scale_shift_norm:
#             out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
#             scale, shift = torch.chunk(temb, 2, dim=1)
#             h = self.norm2(h) * (1 + scale) + shift
#             h = out_rest(h)
#         else:
#             h = h + temb
#             h = self.norm2(h)
#             h = nonlinearity(h)
#             h = self.dropout(h)
#             h = self.conv2(h)

#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x)

#         return x + h
