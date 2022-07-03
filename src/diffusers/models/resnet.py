from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample2D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                x = self.conv(x)
            else:
                x = self.Conv2d_0(x)

        return x


class Downsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        elif name == "Conv2d_0":
            self.Conv2d_0 = conv
        else:
            self.op = conv

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.name == "conv":
            return self.conv(x)
        elif self.name == "Conv2d_0":
            return self.Conv2d_0(x)
        else:
            return self.op(x)


class Upsample1D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample1D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.conv(x)


class FirUpsample2D(nn.Module):
    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_conv:
            h = _upsample_conv_2d(x, self.Conv2d_0.weight, k=self.fir_kernel)
            h = h + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            h = upsample_2d(x, self.fir_kernel, factor=2)

        return h


class FirDownsample2D(nn.Module):
    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = self.Conv2d_0 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_conv:
            x = _conv_downsample_2d(x, self.Conv2d_0.weight, k=self.fir_kernel)
            x = x + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            x = downsample_2d(x, self.fir_kernel, factor=2)

        return x


def _conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    """Fused `Conv2d()` followed by `downsample_2d()`.

    Args:
    Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
    efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of arbitrary
    order.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w: Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and same datatype
        as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def _upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_2d()` followed by `Conv2d()`.

    Args:
    Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
    efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of arbitrary
    order.
      x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
        C]`.
      w: Weight tensor of the shape `[filterH, filterW, inChannels,
        outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
      k: FIR filter of the shape `[firH, firW]` or `[firN]`
        (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
      factor: Integer upsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
      Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same datatype as
      `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]

    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor)

    # Determine data dimensions.
    stride = [1, 1, factor, factor]
    output_shape = ((x.shape[2] - 1) * factor + convH, (x.shape[3] - 1) * factor + convW)
    output_padding = (
        output_shape[0] - (x.shape[2] - 1) * stride[0] - convH,
        output_shape[1] - (x.shape[3] - 1) * stride[1] - convW,
    )
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = x.shape[1] // inC

    # Transpose weights.
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)

    return upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


# TODO (patil-suraj): needs test
# class Upsample2D1d(nn.Module):
#    def __init__(self, dim):
#        super().__init__()
#        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
#
#    def forward(self, x):
#        return self.conv(x)


# unet.py, unet_grad_tts.py, unet_ldm.py, unet_glide.py, unet_score_vde.py
# => All 2D-Resnets are included here now!
class ResnetBlock2D(nn.Module):
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
        kernel=None,
        output_scale_factor=1.0,
        use_nin_shortcut=None,
        up=False,
        down=False,
        overwrite_for_grad_tts=False,
        overwrite_for_ldm=False,
        overwrite_for_glide=False,
        overwrite_for_score_vde=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        if self.pre_norm:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if time_embedding_norm == "default" and temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        elif time_embedding_norm == "scale_shift" and temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, k=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, k=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_nin_shortcut = self.in_channels != self.out_channels if use_nin_shortcut is None else use_nin_shortcut

        self.nin_shortcut = None
        if self.use_nin_shortcut:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # TODO(SURAJ, PATRICK): ALL OF THE FOLLOWING OF THE INIT METHOD CAN BE DELETED ONCE WEIGHTS ARE CONVERTED
        self.is_overwritten = False
        self.overwrite_for_glide = overwrite_for_glide
        self.overwrite_for_grad_tts = overwrite_for_grad_tts
        self.overwrite_for_ldm = overwrite_for_ldm or overwrite_for_glide
        self.overwrite_for_score_vde = overwrite_for_score_vde
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
            channels = in_channels
            emb_channels = temb_channels
            use_scale_shift_norm = False
            non_linearity = "silu"

            self.in_layers = nn.Sequential(
                normalization(channels, swish=1.0),
                nn.Identity(),
                nn.Conv2d(channels, self.out_channels, 3, padding=1),
            )
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if self.time_embedding_norm == "scale_shift" else self.out_channels,
                ),
            )
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
                nn.SiLU() if use_scale_shift_norm else nn.Identity(),
                nn.Dropout(p=dropout),
                zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
            )
            if self.out_channels == in_channels:
                self.skip_connection = nn.Identity()
            else:
                self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
        elif self.overwrite_for_score_vde:
            in_ch = in_channels
            out_ch = out_channels

            eps = 1e-6
            num_groups = min(in_ch // 4, 32)
            num_groups_out = min(out_ch // 4, 32)
            temb_dim = temb_channels

            self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=in_ch, eps=eps)
            self.up = up
            self.down = down
            self.Conv_0 = conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            if temb_dim is not None:
                self.Dense_0 = nn.Linear(temb_dim, out_ch)
                self.Dense_0.weight.data = variance_scaling()(self.Dense_0.weight.shape)
                nn.init.zeros_(self.Dense_0.bias)

            self.GroupNorm_1 = nn.GroupNorm(num_groups=num_groups_out, num_channels=out_ch, eps=eps)
            self.Dropout_0 = nn.Dropout(dropout)
            self.Conv_1 = conv2d(out_ch, out_ch, init_scale=0.0, kernel_size=3, padding=1)
            if in_ch != out_ch or up or down:
                # 1x1 convolution with DDPM initialization.
                self.Conv_2 = conv2d(in_ch, out_ch, kernel_size=1, padding=0)

            self.in_ch = in_ch
            self.out_ch = out_ch

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

    def set_weights_score_vde(self):
        self.conv1.weight.data = self.Conv_0.weight.data
        self.conv1.bias.data = self.Conv_0.bias.data
        self.norm1.weight.data = self.GroupNorm_0.weight.data
        self.norm1.bias.data = self.GroupNorm_0.bias.data

        self.conv2.weight.data = self.Conv_1.weight.data
        self.conv2.bias.data = self.Conv_1.bias.data
        self.norm2.weight.data = self.GroupNorm_1.weight.data
        self.norm2.bias.data = self.GroupNorm_1.bias.data

        self.temb_proj.weight.data = self.Dense_0.weight.data
        self.temb_proj.bias.data = self.Dense_0.bias.data

        if self.in_channels != self.out_channels or self.up or self.down:
            self.nin_shortcut.weight.data = self.Conv_2.weight.data
            self.nin_shortcut.bias.data = self.Conv_2.bias.data

    def forward(self, x, temb, mask=1.0):
        # TODO(Patrick) eventually this class should be split into multiple classes
        # too many if else statements
        if self.overwrite_for_grad_tts and not self.is_overwritten:
            self.set_weights_grad_tts()
            self.is_overwritten = True
        elif self.overwrite_for_ldm and not self.is_overwritten:
            self.set_weights_ldm()
            self.is_overwritten = True
        elif self.overwrite_for_score_vde and not self.is_overwritten:
            self.set_weights_score_vde()
            self.is_overwritten = True

        h = x
        h = h * mask
        if self.pre_norm:
            h = self.norm1(h)
            h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if not self.pre_norm:
            h = self.norm1(h)
            h = self.nonlinearity(h)
        h = h * mask

        if temb is not None:
            temb = self.temb_proj(self.nonlinearity(temb))[:, :, None, None]
        else:
            temb = 0

        if self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)

            h = self.norm2(h)
            h = h + h * scale + shift
            h = self.nonlinearity(h)
        elif self.time_embedding_norm == "default":
            h = h + temb
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
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)

        return (x + h) / self.output_scale_factor


# TODO(Patrick) - just there to convert the weights; can delete afterward
class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1), torch.nn.GroupNorm(groups, dim_out), Mish()
        )


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


def conv2d(in_planes, out_planes, kernel_size=3, stride=1, bias=True, init_scale=1.0, padding=1):
    """nXn convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    conv.weight.data = variance_scaling(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def variance_scaling(scale=1.0, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"):
    """Ported from JAX."""
    scale = 1e-10 if scale == 0 else scale

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        denominator = (fan_in + fan_out) / 2
        variance = scale / denominator
        return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(3 * variance)

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
    r"""Upsample2D a batch of 2D images with the given filter.

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
    r"""Downsample2D a batch of 2D images with the given filter.

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


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k
