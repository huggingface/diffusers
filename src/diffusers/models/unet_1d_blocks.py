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
from torch import nn
import torch.nn.functional as F


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
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv1d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv1d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, s = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, s]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, s])
        return input + self.dropout(self.out_proj(y))


class SkipBlock(nn.Module):
    def __init__(self, *main):
        super().__init__()
        self.main = nn.Sequential(*main)

    def forward(self, input):
        return torch.cat([self.main(input), input], dim=1)


# Noise level (and other) conditioning
class ResConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        super().__init__()
        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        layers = [
            nn.Conv1d(c_in, c_mid, 5, padding=2),
            nn.GroupNorm(1, c_mid),
            nn.GELU(),
            nn.Conv1d(c_mid, c_out, 5, padding=2),
            nn.GroupNorm(1, c_out) if not is_last else nn.Identity(),
            nn.GELU() if not is_last else nn.Identity(),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input) + self.skip(input)


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

    def forward(self, hidden_states):
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

    def forward(self, hidden_states):
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

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (None,)


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
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states
