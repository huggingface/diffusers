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
import math
from torch import nn


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


class Downsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv1d(x, weight, stride=2)


class Upsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)


class SelfAttention1d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.0):
        super().__init__()
        self.channels = c_in
        self.group_norm = nn.GroupNorm(1, num_channels=c_in)
        self.num_heads = n_head

        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels, 1)

        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))

        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output


class ResConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = c_in != c_out

        if self.has_conv_skip:
            self.conv_skip = nn.Conv1d(c_in, c_out, 1, bias=False)

        self.conv_1 = nn.Conv1d(c_in, c_mid, 5, padding=2)
        self.group_norm_1 = nn.GroupNorm(1, c_mid)
        self.gelu_1 = nn.GELU()
        self.conv_2 = nn.Conv1d(c_mid, c_out, 5, padding=2)

        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, c_out)
            self.gelu_2 = nn.GELU()

    def forward(self, hidden_states):
        residual = self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output


def get_down_block(down_block_type, c, c_prev):
    if down_block_type == "DownBlock1D":
        return DownBlock1D(c, c_prev)
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(c, c_prev)
    elif down_block_type == "DownBlock1DNoSkip":
        return DownBlock1DNoSkip(c, c_prev)
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(up_block_type, c, c_prev):
    if up_block_type == "UpBlock1D":
        return UpBlock1D(c, c_prev)
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(c, c_prev)
    elif up_block_type == "UpBlock1DNoSkip":
        return UpBlock1DNoSkip(c, c_prev)
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock1D(nn.Module):
    def __init__(self, c, c_prev):
        super().__init__()

        # there is always at least one resnet
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(c_prev, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c_prev),
        ]
        attentions = [
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c_prev, c_prev // 32),
        ]
        self.up = Upsample1d(kernel="cubic")

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states):
        hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class AttnDownBlock1D(nn.Module):
    def __init__(self, c, c_prev):
        super().__init__()
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(c_prev, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
        ]
        attentions = [
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c, c // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1D(nn.Module):
    def __init__(self, c, c_prev):
        super().__init__()
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(c_prev, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
        ]

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1DNoSkip(nn.Module):
    def __init__(self, c, c_prev):
        super().__init__()
        resnets = [
            ResConvBlock(c_prev, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
        ]

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = torch.cat([hidden_states, temb], dim=1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class AttnUpBlock1D(nn.Module):
    def __init__(self, c, c_prev):
        super().__init__()
        resnets = [
            ResConvBlock(2 * c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c_prev),
        ]
        attentions = [
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c, c // 32),
            SelfAttention1d(c_prev, c_prev // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, res_hidden_states_tuple):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1D(nn.Module):
    def __init__(self, c, c_prev):
        super().__init__()
        resnets = [
            ResConvBlock(2 * c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c_prev),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, res_hidden_states_tuple):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1DNoSkip(nn.Module):
    def __init__(self, c, c_prev):
        super().__init__()
        resnets = [
            ResConvBlock(2 * c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c_prev, is_last=True),
        ]

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, res_hidden_states_tuple):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states
