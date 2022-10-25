# Copyright 2022 The HuggingFace Team. All rights reserved.
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


import torch
import torch.nn.functional as F
from torch import nn

from .resnet import Downsample1D, ResidualTemporalBlock1D, Upsample1D, rearrange_dims


class DownResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        conv_shortcut=False,
        temb_channels=32,
        groups=32,
        groups_out=None,
        non_linearity=None,
        time_embedding_norm="default",
        output_scale_factor=1.0,
        add_downsample=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.add_downsample = add_downsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.ModuleList(resnets)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = None

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)

    def forward(self, hidden_states, temb=None):
        output_states = ()

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        output_states += (hidden_states,)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states, output_states


class UpResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        temb_channels=32,
        groups=32,
        groups_out=None,
        non_linearity=None,
        time_embedding_norm="default",
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.add_upsample = add_upsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.ModuleList(resnets)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = None

        self.upsample = None
        if add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

    def forward(self, hidden_states, res_hidden_states=None, temb=None):
        if res_hidden_states is not None:
            hidden_states = torch.cat((hidden_states, res_hidden_states), dim=1)

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class DownBlock1D(nn.Module):
    pass


class AttnDownBlock1D(nn.Module):
    pass


class DownBlock1DNoSkip(nn.Module):
    pass


class UpBlock1D(nn.Module):
    pass


class AttnUpBlock1D(nn.Module):
    pass


class UpBlock1DNoSkip(nn.Module):
    pass


class ValueFunctionMidBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.res1 = ResidualTemporalBlock1D(in_channels, in_channels // 2, embed_dim=embed_dim)
        self.down1 = Downsample1D(out_channels // 2, use_conv=True)
        self.res2 = ResidualTemporalBlock1D(in_channels // 2, in_channels // 4, embed_dim=embed_dim)
        self.down2 = Downsample1D(out_channels // 4, use_conv=True)

    def forward(self, x, temb=None):
        x = self.res1(x, temb)
        x = self.down1(x)
        x = self.res2(x, temb)
        x = self.down2(x)
        return x


class MidResTemporalBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        num_layers: int = 1,
        add_downsample: bool = False,
        add_upsample: bool = False,
        non_linearity=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=embed_dim)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=embed_dim))

        self.resnets = nn.ModuleList(resnets)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = None

        self.upsample = None
        if add_upsample:
            self.upsample = Downsample1D(out_channels, use_conv=True)

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True)

        if self.upsample and self.downsample:
            raise ValueError("Block cannot downsample and upsample")

    def forward(self, hidden_states, temb):
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)
        if self.downsample:
            self.downsample = self.downsample(hidden_states)

        return hidden_states


class OutConv1DBlock(nn.Module):
    def __init__(self, num_groups_out, out_channels, embed_dim, act_fn):
        super().__init__()
        self.final_conv1d_1 = nn.Conv1d(embed_dim, embed_dim, 5, padding=2)
        self.final_conv1d_gn = nn.GroupNorm(num_groups_out, embed_dim)
        if act_fn == "silu":
            self.final_conv1d_act = nn.SiLU()
        if act_fn == "mish":
            self.final_conv1d_act = nn.Mish()
        self.final_conv1d_2 = nn.Conv1d(embed_dim, out_channels, 1)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.final_conv1d_1(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_gn(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_act(hidden_states)
        hidden_states = self.final_conv1d_2(hidden_states)
        return hidden_states


class OutValueFunctionBlock(nn.Module):
    def __init__(self, fc_dim, embed_dim):
        super().__init__()
        self.final_block = nn.ModuleList(
            [
                nn.Linear(fc_dim + embed_dim, fc_dim // 2),
                nn.Mish(),
                nn.Linear(fc_dim // 2, 1),
            ]
        )

    def forward(self, hidden_states, temb):
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        hidden_states = torch.cat((hidden_states, temb), dim=-1)
        for layer in self.final_block:
            hidden_states = layer(hidden_states)

        return hidden_states


def get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample):
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )

    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(up_block_type, num_layers, in_channels, out_channels, temb_channels, add_upsample):
    if up_block_type == "UpResnetBlock1D":
        return UpResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
        )
    elif up_block_type == "Identity":
        return nn.Identity()
    raise ValueError(f"{up_block_type} does not exist.")


def get_mid_block(mid_block_type, num_layers, in_channels, mid_channels, out_channels, embed_dim, add_downsample):
    if mid_block_type == "MidResTemporalBlock1D":
        return MidResTemporalBlock1D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            add_downsample=add_downsample,
        )
    elif mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim)
    raise ValueError(f"{mid_block_type} does not exist.")


def get_out_block(*, out_block_type, num_groups_out, embed_dim, out_channels, act_fn, fc_dim):
    if out_block_type == "OutConv1DBlock":
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == "ValueFunction":
        return OutValueFunctionBlock(fc_dim, embed_dim)
    else:
        return None
