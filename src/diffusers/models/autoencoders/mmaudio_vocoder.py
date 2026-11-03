# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

# Adapted from the BigVGAN-v2 vocoder MMAudio bundles at
# https://github.com/hkchengrex/MMAudio/tree/main/mmaudio/ext/bigvgan_v2, itself adapted from
# https://github.com/NVIDIA/BigVGAN (MIT license), with the anti-aliased Snake activations of
# https://github.com/junjun3518/alias-free-torch (Apache License 2.0).

"""BigVGAN vocoder that turns the mel spectrograms decoded by [`MMAudioVAE`] into waveforms."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Kaiser-windowed sinc low-pass filter of shape `(1, 1, kernel_size)` normalized to unit sum."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size
    filter = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter = filter / filter.sum()
    return filter.view(1, 1, kernel_size)


class MMAudioSnakeBeta(nn.Module):
    """`x + 1/b * sin^2(a * x)` with per-channel log-scale frequency `a` and magnitude `b`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha)[None, :, None]
        beta = torch.exp(self.beta)[None, :, None]
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class MMAudioLowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int) -> None:
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class MMAudioUpSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class MMAudioDownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.lowpass = MMAudioLowPassFilter1d(0.5 / ratio, 0.6 / ratio, stride=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class MMAudioActivation1d(nn.Module):
    """Anti-aliased activation: 2x upsample, Snake-beta, 2x downsample."""

    def __init__(self, channels: int, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.act = MMAudioSnakeBeta(channels)
        self.upsample = MMAudioUpSample1d(ratio, kernel_size)
        self.downsample = MMAudioDownSample1d(ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class MMAudioAMPBlock(nn.Module):
    """Anti-aliased multi-periodicity block: dilated convolutions each followed by a dilation-1 convolution."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    padding=(kernel_size * dilation - dilation) // 2,
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilations]
        )
        self.activations = nn.ModuleList([MMAudioActivation1d(channels) for _ in range(2 * len(dilations))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations_1, activations_2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, activations_1, activations_2):
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class MMAudioVocoder(ModelMixin, ConfigMixin):
    r"""
    BigVGAN-v2 vocoder (https://github.com/NVIDIA/BigVGAN, MIT license) with the anti-aliased Snake activations of
    https://github.com/junjun3518/alias-free-torch (Apache 2.0), turning the mel spectrograms [`MMAudioVAE`] decodes
    into waveforms for [`Kandinsky6TI2VAPipeline`].

    Args:
        num_mels (`int`, defaults to `128`):
            Number of mel bins of the input spectrogram. Must match the paired [`MMAudioVAE`]'s `mel_bins`.
        upsample_initial_channel (`int`, defaults to `1536`):
            Width of the first layer.
        upsample_rates (`tuple[int, ...]`, defaults to `(8, 4, 2, 2, 2, 2)`):
            Upsampling factors of the vocoder stages. Their product is the total upsampling factor and must match the
            paired [`MMAudioVAE`]'s `hop_length`.
        upsample_kernel_sizes (`tuple[int, ...]`, defaults to `(16, 8, 4, 4, 4, 4)`):
            Transposed-convolution kernel sizes of the vocoder stages.
        resblock_kernel_sizes (`tuple[int, ...]`, defaults to `(3, 7, 11)`):
            Kernel sizes of the residual blocks.
        resblock_dilation_sizes (`tuple[tuple[int, ...], ...]`, defaults to `((1, 3, 5), (1, 3, 5), (1, 3, 5))`):
            Dilations of the residual blocks.
    """

    _no_split_modules = ["MMAudioAMPBlock"]

    @register_to_config
    def __init__(
        self,
        num_mels: int = 128,
        upsample_initial_channel: int = 1536,
        upsample_rates: tuple[int, ...] = (8, 4, 2, 2, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (16, 8, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ) -> None:
        super().__init__()
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError("`upsample_rates` and `upsample_kernel_sizes` must have the same length")
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError("`resblock_kernel_sizes` and `resblock_dilation_sizes` must have the same length")

        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = upsample_initial_channel
        for rate, kernel_size in zip(upsample_rates, upsample_kernel_sizes):
            self.ups.append(
                nn.ModuleList(
                    [nn.ConvTranspose1d(channels, channels // 2, kernel_size, rate, padding=(kernel_size - rate) // 2)]
                )
            )
            channels //= 2
            for block_kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(MMAudioAMPBlock(channels, block_kernel_size, tuple(dilations)))

        self.activation_post = MMAudioActivation1d(channels)
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3, bias=False)

    def forward(self, mel: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        r"""
        Args:
            mel (`torch.Tensor` of shape `(batch_size, num_mels, num_mel_frames)`):
                Mel spectrogram, as decoded by [`MMAudioVAE`].
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.DecoderOutput`] instead of a plain tuple.

        Returns:
            The waveform of shape `(batch_size, 1, num_samples)` in `[-1, 1]`.
        """
        hidden_states = self.conv_pre(mel)
        for stage, up in enumerate(self.ups):
            hidden_states = up[0](hidden_states)
            blocks = self.resblocks[stage * self.num_kernels : (stage + 1) * self.num_kernels]
            hidden_states = sum(block(hidden_states) for block in blocks) / self.num_kernels
        hidden_states = self.conv_post(self.activation_post(hidden_states))
        waveform = torch.clamp(hidden_states, min=-1.0, max=1.0)
        if not return_dict:
            return (waveform,)
        return DecoderOutput(sample=waveform)
