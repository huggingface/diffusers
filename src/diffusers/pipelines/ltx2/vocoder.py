import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """
    Creates a Kaiser sinc kernel for low-pass filtering.

    Args:
        cutoff (`float`):
            Normalized frequency cutoff (relative to the sampling rate). Must be between 0 and 0.5 (the Nyquist
            frequency).
        half_width (`float`):
            Used to determine the Kaiser window's beta parameter.
        kernel_size:
            Size of the Kaiser window (and ultimately the Kaiser sinc kernel).

    Returns:
        `torch.Tensor` of shape `(kernel_size,)`:
            The Kaiser sinc kernel.
    """
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size

    if cutoff == 0.0:
        filter = torch.zeros_like(time)
    else:
        time = 2 * cutoff * time
        sinc = torch.where(
            time == 0,
            torch.ones_like(time),
            torch.sin(math.pi * time) / math.pi / time,
        )
        filter = 2 * cutoff * window * sinc
        filter = filter / filter.sum()
    return filter


class DownSample1d(nn.Module):
    """1D low-pass filter for antialias downsampling."""

    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        use_padding: bool = True,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size or int(6 * ratio // 2) * 2
        self.pad_left = self.kernel_size // 2 + (self.kernel_size % 2) - 1
        self.pad_right = self.kernel_size // 2
        self.use_padding = use_padding
        self.padding_mode = padding_mode

        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        low_pass_filter = kaiser_sinc_filter1d(cutoff, half_width, self.kernel_size)
        self.register_buffer("filter", low_pass_filter.view(1, 1, self,kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: [batch_size, num_channels, hidden_dim]
        num_channels = x.shape[1]
        if self.use_padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        x_filtered = F.conv1d(x, self.filter.expand(num_channels, -1, -1), stride=self.ratio, groups=num_channels)
        return x_filtered


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None  = None,
        window_type: str = "kaiser",
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.ratio = ratio
        self.padding_mode = padding_mode

        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio

            time_axis = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            sinc_filter = (torch.sinc(time_axis) * window * rolloff / ratio).view(1, 1, -1)
        else:
            # Kaiser sinc filter is BigVGAN default
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

            sinc_filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

        self.register_buffer("filter", sinc_filter, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: [batch_size, num_channels, hidden_dim]
        num_channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode=self.padding_mode)
        low_pass_filter = self.filter.to(dtype=x.dtype, device=x.device).expand(num_channels, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, low_pass_filter, stride=self.ratio, groups=num_channels)
        return x[..., self.pad_left:-self.pad_right]


class SnakeBeta(nn.Module):
    """
    Implements the Snake and SnakeBeta activations, which help with learning periodic patterns.
    """
    def __init__(
        self,
        channels: int,
        alpha: float = 1.0,
        eps: float = 1e-9,
        trainable_params: bool = True,
        logscale: bool = True,
        use_beta: bool = True,
    ):
        self.eps = eps
        self.logscale = logscale
        self.use_beta = use_beta

        self.alpha = nn.Parameter(torch.zeros(channels) if self.logscale else torch.ones(channels) * alpha)
        self.alpha.requires_grad = trainable_params
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(channels) if self.logscale else torch.ones(channels) * alpha)
            self.beta.requires_grad = trainable_params

    def forward(self, hidden_states: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        broadcast_shape = [1] * hidden_states.ndim
        broadcast_shape[channel_dim] = -1
        alpha = self.alpha.view(broadcast_shape)
        if self.use_beta:
            beta = self.beta.view(broadcast_shape)

        if self.logscale:
            alpha = torch.exp(alpha)
            if self.use_beta:
                beta = torch.exp(beta)

        amplitude = beta if self.use_beta else alpha
        hidden_states = hidden_states + (1.0 / (amplitude + self.eps)) * torch.sin(hidden_states * alpha).pow(2)
        return hidden_states


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: tuple[int, ...] = (1, 3, 5),
        act_fn: str = "leaky_relu",
        leaky_relu_negative_slope: float = 0.1,
        antialias: bool = False,
        antialias_ratio: int = 2,
        antialias_kernel_size: int = 12,
        padding_mode: str = "same",
    ):
        super().__init__()
        self.dilations = dilations

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, stride=stride, dilation=dilation, padding=padding_mode)
                for dilation in dilations
            ]
        )
        self.acts1 = nn.ModuleList()
        for _ in range(len(self.convs1)):
            if act_fn == "snakebeta":
                act = SnakeBeta(channels, use_beta=True)
            elif act_fn == "snake":
                act = SnakeBeta(channels, use_beta=False)
            else:
                act_fn = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

            if antialias:
                act = nn.Sequential(
                    UpSample1d(ratio=antialias_ratio, kernel_size=antialias_kernel_size),
                    act,
                    DownSample1d(ratio=antialias_ratio, kernel_size=antialias_kernel_size),
                )
            self.acts1.append(act)

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, stride=stride, dilation=1, padding=padding_mode)
                for _ in range(len(dilations))
            ]
        )
        self.acts2 = nn.ModuleList()
        for _ in range(len(self.convs2)):
            if act_fn == "snakebeta":
                act = SnakeBeta(channels, use_beta=True)
            elif act_fn == "snake":
                act = SnakeBeta(channels, use_beta=False)
            else:
                act_fn = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

            if antialias:
                act = nn.Sequential(
                    UpSample1d(ratio=antialias_ratio, kernel_size=antialias_kernel_size),
                    act,
                    DownSample1d(ratio=antialias_ratio, kernel_size=antialias_kernel_size),
                )
            self.acts2.append(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act1, conv1, act2, conv2 in zip(self.acts1, self.convs1, self.acts2, self.convs2):
            xt = act1(xt)
            xt = conv1(xt)
            xt = act2(xt)
            xt = conv2(xt)
            x = x + xt
        return x


class LTX2Vocoder(ModelMixin, ConfigMixin):
    r"""
    LTX 2.0 vocoder for converting generated mel spectrograms back to audio waveforms.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 1024,
        out_channels: int = 2,
        upsample_kernel_sizes: list[int] = [16, 15, 8, 4, 4],
        upsample_factors: list[int] = [6, 5, 2, 2, 2],
        resnet_kernel_sizes: list[int] = [3, 7, 11],
        resnet_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        act_fn: str = "leaky_relu",
        leaky_relu_negative_slope: float = 0.1,
        antialias: bool = False,
        antialias_ratio: int = 2,
        antialias_kernel_size: int = 12,
        final_act_fn: str | None = "tanh",
        final_bias: bool = True,
        output_sampling_rate: int = 24000,
    ):
        super().__init__()
        self.num_upsample_layers = len(upsample_kernel_sizes)
        self.resnets_per_upsample = len(resnet_kernel_sizes)
        self.out_channels = out_channels
        self.total_upsample_factor = math.prod(upsample_factors)
        self.act_fn = act_fn
        self.negative_slope = leaky_relu_negative_slope
        self.final_act_fn = final_act_fn

        if self.num_upsample_layers != len(upsample_factors):
            raise ValueError(
                f"`upsample_kernel_sizes` and `upsample_factors` should be lists of the same length but are length"
                f" {self.num_upsample_layers} and {len(upsample_factors)}, respectively."
            )

        if self.resnets_per_upsample != len(resnet_dilations):
            raise ValueError(
                f"`resnet_kernel_sizes` and `resnet_dilations` should be lists of the same length but are length"
                f" {len(self.resnets_per_upsample)} and {len(resnet_dilations)}, respectively."
            )

        supported_act_fns = ["snakebeta", "snake", "leaky_relu"]
        if self.act_fn not in supported_act_fns:
            raise ValueError(
                f"Unsupported activation function: {self.act_fn}. Currently supported values of `act_fn` are "
                f"{supported_act_fns}."
            )

        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=3)

        self.upsamplers = nn.ModuleList()
        self.resnets = nn.ModuleList()
        input_channels = hidden_channels
        for i, (stride, kernel_size) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            output_channels = input_channels // 2
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    input_channels,  # hidden_channels // (2 ** i)
                    output_channels,  # hidden_channels // (2 ** (i + 1))
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

            for kernel_size, dilations in zip(resnet_kernel_sizes, resnet_dilations):
                self.resnets.append(
                    ResBlock(
                        channels=output_channels,
                        kernel_size=kernel_size,
                        dilations=dilations,
                        act_fn=act_fn,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                        antialias=antialias,
                        antialias_ratio=antialias_ratio,
                        antialias_kernel_size=antialias_kernel_size,
                    )
                )
            input_channels = output_channels

        if act_fn == "snakebeta" or act_fn == "snake":
            # Always use antialiasing
            self.act_out = nn.Sequential(
                UpSample1d(ratio=antialias_ratio, kernel_size=antialias_kernel_size),
                SnakeBeta(channels=out_channels, use_beta=True),
                DownSample1d(ratio=antialias_ratio, kernel_size=antialias_kernel_size),
            )
        elif act_fn == "leaky_relu":
            # NOTE: does NOT use self.negative_slope, following the original code
            self.act_out = nn.LeakyReLU()

        self.conv_out = nn.Conv1d(output_channels, out_channels, 7, stride=1, padding=3, bias=final_bias)

    def forward(self, hidden_states: torch.Tensor, time_last: bool = False) -> torch.Tensor:
        r"""
        Forward pass of the vocoder.

        Args:
            hidden_states (`torch.Tensor`):
                Input Mel spectrogram tensor of shape `(batch_size, num_channels, time, num_mel_bins)` if `time_last`
                is `False` (the default) or shape `(batch_size, num_channels, num_mel_bins, time)` if `time_last` is
                `True`.
            time_last (`bool`, *optional*, defaults to `False`):
                Whether the last dimension of the input is the time/frame dimension or the Mel bins dimension.

        Returns:
            `torch.Tensor`:
                Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """

        # Ensure that the time/frame dimension is last
        if not time_last:
            hidden_states = hidden_states.transpose(2, 3)
        # Combine channels and frequency (mel bins) dimensions
        hidden_states = hidden_states.flatten(1, 2)

        hidden_states = self.conv_in(hidden_states)

        for i in range(self.num_upsample_layers):
            if self.act_fn == "leaky_relu":
                # Other activations are inside each upsampling block
                hidden_states = F.leaky_relu(hidden_states, negative_slope=self.negative_slope)
            hidden_states = self.upsamplers[i](hidden_states)

            # Run all resnets in parallel on hidden_states
            start = i * self.resnets_per_upsample
            end = (i + 1) * self.resnets_per_upsample
            resnet_outputs = torch.stack([self.resnets[j](hidden_states) for j in range(start, end)], dim=0)

            hidden_states = torch.mean(resnet_outputs, dim=0)

        hidden_states = self.act_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        if self.final_act_fn == "tanh":
            hidden_states = torch.tanh(hidden_states)
        else:
            hidden_states = torch.clamp(hidden_states, -1, 1)

        return hidden_states
