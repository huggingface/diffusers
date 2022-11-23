#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from aitemplate.compiler import ops
from aitemplate.frontend import nn


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


class Upsample2D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels,
        use_conv=False,
        use_conv_transpose=False,
        out_channels=None,
        name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2dBias(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2dBias(self.channels, self.out_channels, 3, 1, 1)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, x):
        assert get_shape(x)[-1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)

        x = nn.Upsampling2d(scale_factor=2.0, mode="nearest")(x)

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

    def __init__(
        self, channels, use_conv=False, out_channels=None, padding=1, name="conv"
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2dBias(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, x):
        assert get_shape(x)[-1] == self.channels
        x = self.conv(x)

        return x


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
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
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

        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            use_swish=True,
        )

        self.conv1 = nn.Conv2dBias(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            use_swish=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2dBias(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.upsample = self.downsample = None

        self.use_nin_shortcut = (
            self.in_channels != self.out_channels
            if use_nin_shortcut is None
            else use_nin_shortcut
        )

        if self.use_nin_shortcut:
            self.conv_shortcut = nn.Conv2dBias(
                in_channels, out_channels, 1, 1, 0
            )  # kernel_size=1, stride=1, padding=0) # conv_bias_add
        else:
            self.conv_shortcut = None

    def forward(self, x, temb=None):
        hidden_states = x

        # make sure hidden states is in float32
        # when running in half-precision
        hidden_states = self.norm1(
            hidden_states
        )  # .float()).type(hidden_states.dtype) # fused swish
        # hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            x = self.upsample(x)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            x = self.downsample(x)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(ops.silu(temb))
            bs, dim = get_shape(temb)
            temb = ops.reshape()(temb, [bs, 1, 1, dim])
            hidden_states = hidden_states + temb

        # make sure hidden states is in float32
        # when running in half-precision
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = hidden_states + x

        return out
