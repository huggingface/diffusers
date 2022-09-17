# All the vocoders used in diffusions pipelines will be implemented here.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin


# DiffSound Uses MelGAN
class MelGAN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        return


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = (
            self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
        )

    def forward(self, x, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )[..., : -self.causal_padding]


class SoundStreamResNet(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.causal_conv = nn.CausalConv1d(in_channels, out_channels, kernel_size=7, dilation=dilation)
        self.conv_1d = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act = nn.ELU()

    def forward(self, hidden_states):
        residuals = hidden_states
        hidden_states = self.causal_conv(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.conv_1d(hidden_states)
        return residuals + hidden_states


class SoundStreamDecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()
        self.project_in = CausalConvTranspose1d(
            in_channels=2 * out_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride
        )
        self.act = nn.ELU()

        self.resnet_blocks = nn.ModuleList(
            [SoundStreamResNet(out_channels, out_channels, 512, dilation=3 ^ rate) for rate in range(3)]
        )

    def forward(self, hidden_states):
        hidden_states = self.project_in(hidden_states)
        hidden_states = self.act(hidden_states)
        for resnet in self.resnet_blocks:
            hidden_states = resnet(hidden_states)
        return hidden_states


# notes2audio uses SoundStream
class SoundStreamVocoder(ModelMixin, ConfigMixin):
    """Residual VQ VAE model from `SoundStream: An End-to-End Neural Audio Codec`

    Args:
        in_channels (`int`): number of input channels. It corresponds to the number of spectrogram features
            that are passed to the decoder to compute the raw audio.
        ConfigMixin (_type_): _description_
    """

    def __init__(self, in_channels=8, out_channels=1, strides=[8, 5, 4, 2], channel_factors=[8, 4, 2, 1]):
        super().__init__()
        self.act = nn.ELU()
        self.bottleneck = CausalConv1d(in_channels=in_channels, out_channels=16 * out_channels, kernel_size=7)
        self.decoder_blocks = nn.ModuleList(
            SoundStreamDecoderBlock(out_channels=out_channels * channel_factors[i], stride=strides[i])
            for i in range(4)
        )
        self.last_layer_conv = CausalConv1d(in_channels=out_channels, out_channels=1, kernel_size=7)
        return

    def decode(self, features):
        """Decodes features to audio.
        Args:
        features: Mel spectrograms, shape [batch, n_frames, n_dims].
        Returns:
        audio: Shape [batch, n_frames * hop_size]
        """
        if self._decode_dither_amount > 0:
            features += torch.random.normal(size=features.shape) * self._decode_dither_amount

        hidden_states = self.bottleneck(features)
        hidden_states = self.act(hidden_states)
        for layer in self.decoder_blocks:
            hidden_states = layer(hidden_states)
            hidden_states = self.act(hidden_states)

        audio = self.last_layer_conv(hidden_states)

        return audio


# TODO @Arthur DiffSinger uses this as vocoder
class HiFiGAN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        return
