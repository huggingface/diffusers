# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""MMAudio mel-spectrogram VAE and BigVGAN vocoder used by the Kandinsky 6 TI2VA pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..attention_dispatch import AttentionBackendName, dispatch_attention_fn
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


# Activations of the magnitude-preserving blocks are clipped to this range.
ACTIVATION_CLIP = 256.0


def mel_filterbank(sample_rate: int, n_fft: int, num_mels: int, f_min: float, f_max: float) -> torch.Tensor:
    """Slaney-scale mel filterbank of shape `(num_mels, n_fft // 2 + 1)` with Slaney area normalization, the
    `librosa.filters.mel` default that the MMAudio front end was trained with."""

    def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
        linear_step = 200.0 / 3
        min_log_hz = 1000.0
        log_step = math.log(6.4) / 27.0
        return torch.where(
            freq >= min_log_hz, min_log_hz / linear_step + torch.log(freq / min_log_hz) / log_step, freq / linear_step
        )

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        linear_step = 200.0 / 3
        min_log_hz = 1000.0
        log_step = math.log(6.4) / 27.0
        min_log_mel = min_log_hz / linear_step
        return torch.where(
            mel >= min_log_mel, min_log_hz * torch.exp(log_step * (mel - min_log_mel)), linear_step * mel
        )

    fft_freqs = torch.linspace(0, sample_rate / 2, 1 + n_fft // 2, dtype=torch.float64)
    mel_limits = hz_to_mel(torch.tensor([f_min, f_max], dtype=torch.float64))
    mel_freqs = mel_to_hz(torch.linspace(mel_limits[0], mel_limits[1], num_mels + 2, dtype=torch.float64))
    freq_diff = torch.diff(mel_freqs)
    ramps = mel_freqs[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / freq_diff[:-1, None]
    upper = ramps[2:] / freq_diff[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0)
    weights = weights * (2.0 / (mel_freqs[2 : num_mels + 2] - mel_freqs[:num_mels]))[:, None]
    return weights.float()


# The magnitude-preserving building blocks below follow Karras et al., "Analyzing and Improving the Training
# Dynamics of Diffusion Models" (https://arxiv.org/abs/2312.02696): each layer keeps a unit-variance input
# unit-variance, which MMAudio's VAE relies on instead of normalization layers.


def normalize(x: torch.Tensor, dim: list[int] | None = None, eps: float = 1e-4) -> torch.Tensor:
    """Rescale `x` to unit L2 norm over `dim` (default: every dimension but the first)."""
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = eps + norm * math.sqrt(norm.numel() / x.numel())
    return x / norm.to(x.dtype)


def mp_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU rescaled so that a unit-variance input stays unit-variance."""
    return F.silu(x) / 0.596


def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """Interpolate `a` and `b` and rescale so that the result stays unit-variance."""
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t**2)


class MMAudioMPConv1d(nn.Conv1d):
    """Magnitude-preserving 1D convolution with an optional per-call gain. The released weights are already
    normalized to unit norm per output channel and scaled by `1 / sqrt(fan_in)`, so the weight is applied directly."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor, gain: float | torch.Tensor = 1.0) -> torch.Tensor:
        return F.conv1d(x, (self.weight * gain).to(x.dtype), padding=self.padding)


class MMAudioResnetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = MMAudioMPConv1d(in_channels, out_channels, kernel_size)
        self.conv2 = MMAudioMPConv1d(out_channels, out_channels, kernel_size)
        if in_channels != out_channels:
            self.nin_shortcut = MMAudioMPConv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize(x, dim=1)
        hidden_states = self.conv1(mp_silu(x))
        hidden_states = self.conv2(mp_silu(hidden_states))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return mp_sum(x, hidden_states, t=0.3)


class MMAudioAttnBlock1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = MMAudioMPConv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = MMAudioMPConv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        qkv = self.qkv(x).reshape(batch_size, self.num_heads, -1, 3, length)
        query, key, value = normalize(qkv, dim=2).unbind(3)
        # `(B, heads, D, T)` -> `(B, T, heads, D)` for the attention dispatcher. `D` is not contiguous after the
        # permute (q/k/v are interleaved along the channel dim), which some attention backends (e.g. FlashAttention-3)
        # require, so make the tensors contiguous here.
        query, key, value = (t.permute(0, 3, 1, 2).contiguous() for t in (query, key, value))
        # This block always attends with a single head over the full channel width (like the SD-VAE `AttnBlock`),
        # so `head_dim` can be in the thousands. FlashAttention-family backends cap `head_dim` at 256, so force
        # the native SDPA backend here regardless of whatever backend is globally active for the rest of the
        # pipeline (e.g. via `transformer.set_attention_backend(...)`).
        hidden_states = dispatch_attention_fn(query, key, value, backend=AttentionBackendName.NATIVE)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, channels, length)
        return mp_sum(x, self.proj_out(hidden_states), t=0.3)


class MMAudioUpsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = MMAudioMPConv1d(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest-exact"))


class MMAudioDownsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = MMAudioMPConv1d(channels, channels, kernel_size=1)
        self.conv2 = MMAudioMPConv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.avg_pool1d(self.conv1(x), kernel_size=2, stride=2))


class MMAudioEncoder1D(nn.Module):
    """Mel-spectrogram encoder: residual blocks over `channel_multipliers` levels, a single 2x temporal downsample
    after the first level, and an attention block in the middle."""

    def __init__(
        self,
        mel_bins: int,
        latent_channels: int,
        hidden_channels: int,
        channel_multipliers: tuple[int, ...],
        layers_per_block: int,
    ) -> None:
        super().__init__()
        self.conv_in = MMAudioMPConv1d(mel_bins, hidden_channels, kernel_size=3)

        self.down = nn.ModuleList()
        block_in = hidden_channels
        for level, multiplier in enumerate(channel_multipliers):
            block_out = hidden_channels * multiplier
            blocks = nn.ModuleList()
            for _ in range(layers_per_block):
                blocks.append(MMAudioResnetBlock1D(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = blocks
            if level == 0:
                down.downsample = MMAudioDownsample1D(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = MMAudioResnetBlock1D(block_in, block_in)
        self.mid.attn_1 = MMAudioAttnBlock1D(block_in)
        self.mid.block_2 = MMAudioResnetBlock1D(block_in, block_in)

        self.conv_out = MMAudioMPConv1d(block_in, 2 * latent_channels, kernel_size=3)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                hidden_states = block(hidden_states).clamp(-ACTIVATION_CLIP, ACTIVATION_CLIP)
            if hasattr(down, "downsample"):
                hidden_states = down.downsample(hidden_states)

        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states).clamp(-ACTIVATION_CLIP, ACTIVATION_CLIP)
        return self.conv_out(mp_silu(hidden_states), gain=self.learnable_gain + 1)


class MMAudioDecoder1D(nn.Module):
    """Mirror of [`MMAudioEncoder1D`]: the 2x temporal upsample sits after the second-to-last level."""

    def __init__(
        self,
        mel_bins: int,
        latent_channels: int,
        hidden_channels: int,
        channel_multipliers: tuple[int, ...],
        layers_per_block: int,
    ) -> None:
        super().__init__()
        block_in = hidden_channels * channel_multipliers[-1]
        self.conv_in = MMAudioMPConv1d(latent_channels, block_in, kernel_size=3)

        self.mid = nn.Module()
        self.mid.block_1 = MMAudioResnetBlock1D(block_in, block_in)
        self.mid.attn_1 = MMAudioAttnBlock1D(block_in)
        self.mid.block_2 = MMAudioResnetBlock1D(block_in, block_in)

        self.up = nn.ModuleList()
        for level in reversed(range(len(channel_multipliers))):
            block_out = hidden_channels * channel_multipliers[level]
            blocks = nn.ModuleList()
            for _ in range(layers_per_block + 1):
                blocks.append(MMAudioResnetBlock1D(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = blocks
            if level == 1:
                up.upsample = MMAudioUpsample1D(block_in)
            self.up.insert(0, up)

        self.conv_out = MMAudioMPConv1d(block_in, mel_bins, kernel_size=3)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(z)
        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states).clamp(-ACTIVATION_CLIP, ACTIVATION_CLIP)

        for level in reversed(range(len(self.up))):
            up = self.up[level]
            for block in up.block:
                hidden_states = block(hidden_states).clamp(-ACTIVATION_CLIP, ACTIVATION_CLIP)
            if hasattr(up, "upsample"):
                hidden_states = up.upsample(hidden_states)
        return self.conv_out(mp_silu(hidden_states), gain=self.learnable_gain + 1)


class MMAudioAutoencoder(nn.Module):
    """Encoder/decoder pair over standardized log-mel spectrograms. `data_mean` and `data_std` hold the per-bin
    statistics the checkpoint was trained with."""

    def __init__(
        self,
        mel_bins: int,
        latent_channels: int,
        hidden_channels: int,
        channel_multipliers: tuple[int, ...],
        layers_per_block: int,
    ) -> None:
        super().__init__()
        self.register_buffer("data_mean", torch.zeros(1, mel_bins, 1))
        self.register_buffer("data_std", torch.ones(1, mel_bins, 1))
        self.encoder = MMAudioEncoder1D(
            mel_bins, latent_channels, hidden_channels, channel_multipliers, layers_per_block
        )
        self.decoder = MMAudioDecoder1D(
            mel_bins, latent_channels, hidden_channels, channel_multipliers, layers_per_block
        )

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder((mel - self.data_mean) / self.data_std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) * self.data_std + self.data_mean


# BigVGAN vocoder (https://github.com/NVIDIA/BigVGAN, MIT license) with the anti-aliased Snake activations of
# https://github.com/junjun3518/alias-free-torch (Apache 2.0).


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
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ = filter_ / filter_.sum()
    return filter_.view(1, 1, kernel_size)


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


class MMAudioBigVGAN(nn.Module):
    """BigVGAN-v2 vocoder turning a mel spectrogram into a waveform."""

    def __init__(
        self,
        num_mels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ) -> None:
        super().__init__()
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

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_pre(mel)
        for stage, up in enumerate(self.ups):
            hidden_states = up[0](hidden_states)
            blocks = self.resblocks[stage * self.num_kernels : (stage + 1) * self.num_kernels]
            hidden_states = sum(block(hidden_states) for block in blocks) / self.num_kernels
        hidden_states = self.conv_post(self.activation_post(hidden_states))
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


class MMAudioMelSpectrogram(nn.Module):
    """Log-mel front end of the encoder. The filterbank and window are buffers so they follow the model's device."""

    def __init__(self, sample_rate: int, n_fft: int, num_mels: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("mel_basis", mel_filterbank(sample_rate, n_fft, num_mels, 0.0, sample_rate / 2))
        self.register_buffer("hann_window", torch.hann_window(n_fft))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.clamp(min=-1.0, max=1.0)
        padding = (self.n_fft - self.hop_length) // 2
        waveform = F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spectrum = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9).float()
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, magnitude), min=1e-5))


class MMAudioVAE(ModelMixin, ConfigMixin):
    r"""
    Audio VAE of [`Kandinsky6TI2VAPipeline`]: a magnitude-preserving autoencoder over log-mel spectrograms (MMAudio,
    https://arxiv.org/abs/2412.15322) followed by a BigVGAN-v2 vocoder.

    `encode` turns a waveform into a latent distribution; `decode` turns latents into a waveform through the mel
    decoder and the vocoder. One latent frame covers `hop_length * 2` samples.

    Args:
        mel_bins (`int`, defaults to `128`):
            Number of mel bins.
        latent_channels (`int`, defaults to `40`):
            Number of latent channels.
        hidden_channels (`int`, defaults to `512`):
            Base width of the autoencoder.
        channel_multipliers (`tuple[int, ...]`, defaults to `(1, 2, 4)`):
            Width multipliers of the autoencoder levels.
        layers_per_block (`int`, defaults to `2`):
            Residual blocks per encoder level; the decoder uses one more per level.
        sample_rate (`int`, defaults to `44100`):
            Waveform sample rate.
        n_fft (`int`, defaults to `2048`):
            FFT size of the mel front end.
        hop_length (`int`, defaults to `512`):
            Hop length of the mel front end, also the vocoder's total upsampling factor.
        vocoder_upsample_initial_channel (`int`, defaults to `1536`):
            Width of the vocoder's first layer.
        vocoder_upsample_rates (`tuple[int, ...]`, defaults to `(8, 4, 2, 2, 2, 2)`):
            Upsampling factors of the vocoder stages.
        vocoder_upsample_kernel_sizes (`tuple[int, ...]`, defaults to `(16, 8, 4, 4, 4, 4)`):
            Transposed-convolution kernel sizes of the vocoder stages.
        vocoder_resblock_kernel_sizes (`tuple[int, ...]`, defaults to `(3, 7, 11)`):
            Kernel sizes of the vocoder's residual blocks.
        vocoder_resblock_dilation_sizes (`tuple[tuple[int, ...], ...]`, defaults to `((1, 3, 5), (1, 3, 5), (1, 3, 5))`):
            Dilations of the vocoder's residual blocks.
        scaling_factor (`float`, defaults to `0.417`):
            Scale applied to the latents before they enter the diffusion transformer.
    """

    _no_split_modules = ["MMAudioResnetBlock1D", "MMAudioAttnBlock1D", "MMAudioAMPBlock"]

    @register_to_config
    def __init__(
        self,
        mel_bins: int = 128,
        latent_channels: int = 40,
        hidden_channels: int = 512,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        layers_per_block: int = 2,
        sample_rate: int = 44_100,
        n_fft: int = 2048,
        hop_length: int = 512,
        vocoder_upsample_initial_channel: int = 1536,
        vocoder_upsample_rates: tuple[int, ...] = (8, 4, 2, 2, 2, 2),
        vocoder_upsample_kernel_sizes: tuple[int, ...] = (16, 8, 4, 4, 4, 4),
        vocoder_resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        vocoder_resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        scaling_factor: float = 0.417,
    ) -> None:
        super().__init__()
        if math.prod(vocoder_upsample_rates) != hop_length:
            raise ValueError("the vocoder upsampling rates must multiply to `hop_length`")

        self.mel_converter = MMAudioMelSpectrogram(sample_rate, n_fft, mel_bins, hop_length)
        self.vae = MMAudioAutoencoder(
            mel_bins, latent_channels, hidden_channels, channel_multipliers, layers_per_block
        )
        self.vocoder = MMAudioBigVGAN(
            num_mels=mel_bins,
            upsample_initial_channel=vocoder_upsample_initial_channel,
            upsample_rates=vocoder_upsample_rates,
            upsample_kernel_sizes=vocoder_upsample_kernel_sizes,
            resblock_kernel_sizes=vocoder_resblock_kernel_sizes,
            resblock_dilation_sizes=vocoder_resblock_dilation_sizes,
        )
        # The encoder downsamples the mel frames once by 2.
        self.latent_hop_length = hop_length * 2

    @apply_forward_hook
    def encode(self, audio: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        r"""
        Encode a waveform into its latent distribution.

        Args:
            audio (`torch.Tensor` of shape `(batch_size, num_samples)`):
                Mono waveform in `[-1, 1]` at `sample_rate`.
            return_dict (`bool`, defaults to `True`):
                Whether to return an [`~models.modeling_outputs.AutoencoderKLOutput`] instead of a plain tuple.
        """
        mel = self.mel_converter(audio).to(self.dtype)
        posterior = DiagonalGaussianDistribution(self.vae.encode(mel))
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        r"""
        Decode latents into a waveform.

        Args:
            z (`torch.Tensor` of shape `(batch_size, latent_channels, num_latent_frames)`):
                Latents, already divided by `scaling_factor`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.DecoderOutput`] instead of a plain tuple.

        Returns:
            The waveform of shape `(batch_size, 1, num_samples)` in `[-1, 1]`.
        """
        waveform = self.vocoder(self.vae.decode(z))
        if not return_dict:
            return (waveform,)
        return DecoderOutput(sample=waveform)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | tuple:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        waveform = self.decode(z).sample
        if not return_dict:
            return (waveform,)
        return DecoderOutput(sample=waveform)


__all__ = ["MMAudioVAE"]
