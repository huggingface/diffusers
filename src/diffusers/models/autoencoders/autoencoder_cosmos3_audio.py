# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos3 AVAE Audio Tokenizer.

The decoder reuses the Oobleck architecture (Snake1d activations + weight-norm convs + residual units), inlined here
instead of imported so the audio module is self-contained. The encoder is the Cosmos3 SpecConvNeXt audio encoder used
by AVAE checkpoints; it is intentionally separate from Oobleck's waveform encoder because the tensor layouts and
bottleneck semantics are different.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import FP32LayerNorm
from .autoencoder_oobleck import OobleckDiagonalGaussianDistribution


# Copied from diffusers.models.autoencoders.autoencoder_oobleck.Snake1d
class Snake1d(nn.Module):
    """
    A 1-dimensional Snake activation function module.
    """

    def __init__(self, hidden_dim, logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))

        self.alpha.requires_grad = True
        self.beta.requires_grad = True
        self.logscale = logscale

    def forward(self, hidden_states):
        shape = hidden_states.shape

        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)

        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        hidden_states = hidden_states.reshape(shape)
        return hidden_states


class Cosmos3AudioConvNeXtBlock(nn.Module):
    """1D ConvNeXt block used by the Cosmos3 SpecConvNeXt encoder."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        identity_init: bool = False,
        use_snake: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        if causal:
            self.dwconv = nn.Sequential(
                nn.ConstantPad1d((6, 0), 0),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, groups=hidden_dim),
            )
        else:
            self.dwconv = nn.Sequential(
                nn.ConstantPad1d((3, 3), 0),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, groups=hidden_dim),
            )

        self.norm = FP32LayerNorm(hidden_dim, eps=1e-5, bias=False)
        self.pwconv1 = nn.Conv1d(hidden_dim, intermediate_dim, kernel_size=1)
        self.act = Snake1d(intermediate_dim) if use_snake else nn.GELU()
        self.pwconv2 = nn.Conv1d(intermediate_dim, hidden_dim, kernel_size=1)
        if identity_init:
            nn.init.zeros_(self.pwconv2.weight)
            if self.pwconv2.bias is not None:
                nn.init.zeros_(self.pwconv2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = self.norm(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        return residual + hidden_states


class Cosmos3AudioSpectrogramConvNeXtEncoder(nn.Module):
    """Cosmos3 waveform-to-latent encoder using STFT features and ConvNeXt blocks."""

    def __init__(
        self,
        input_channels: int,
        stereo: bool,
        channels: int,
        latent_dim: int,
        channel_multiples: tuple[int, ...],
        strides: tuple[int, ...],
        num_blocks: int,
        n_fft: int,
        hop_length: int,
        identity_init: bool,
        use_snake: bool,
        causal: bool,
        padding_mode: str,
    ):
        super().__init__()

        if causal:
            raise NotImplementedError("Cosmos3 AVAE causal audio encoder is not supported yet.")
        if len(channel_multiples) != len(strides):
            raise ValueError(
                "`enc_c_mults` and `enc_strides` must have the same length, got "
                f"{len(channel_multiples)} and {len(strides)}."
            )

        self.input_channels = input_channels * (2 if stereo else 1)
        self.channels = channels
        self.latent_dim = latent_dim
        self.channel_multiples = tuple(channel_multiples)
        self.strides = tuple(strides)
        self.num_blocks = num_blocks
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.causal = causal

        layers: list[nn.Module] = [
            weight_norm(
                nn.Conv1d(
                    (n_fft + 2) * self.input_channels,
                    self.channel_multiples[0] * channels,
                    kernel_size=1,
                    bias=False,
                )
            )
        ]

        for index, stride in enumerate(self.strides):
            input_dim = self.channel_multiples[index] * channels
            output_dim = (
                self.channel_multiples[index + 1] * channels
                if index < len(self.channel_multiples) - 1
                else self.channel_multiples[-1] * channels
            )

            for _ in range(num_blocks):
                layers.append(
                    Cosmos3AudioConvNeXtBlock(
                        hidden_dim=input_dim,
                        intermediate_dim=input_dim * 4,
                        identity_init=identity_init,
                        use_snake=use_snake,
                        causal=causal,
                    )
                )

            layers.append(
                weight_norm(
                    nn.Conv1d(
                        input_dim,
                        output_dim,
                        kernel_size=2 * stride,
                        stride=stride,
                        padding=math.ceil(stride / 2),
                        padding_mode=padding_mode,
                    )
                )
            )

        layers.append(
            weight_norm(nn.Conv1d(self.channel_multiples[-1] * channels, latent_dim, kernel_size=1, bias=False))
        )
        self.layers = nn.Sequential(*layers)

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        pad_left = (self.n_fft - self.hop_length) // 2
        pad_right = (self.n_fft - self.hop_length) - pad_left
        waveform = F.pad(waveform, (pad_left, pad_right)).float()
        window = torch.hann_window(self.n_fft, device=waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_samples = audio.shape
        if num_channels != self.input_channels:
            raise ValueError(
                f"Cosmos3 AVAE encoder expected {self.input_channels} audio channels, got {num_channels}."
            )

        if num_channels > 1:
            audio = audio.reshape(batch_size * num_channels, 1, num_samples)

        spectrogram = self._spectrogram(audio.squeeze(1))
        real, imaginary = torch.view_as_real(spectrogram).chunk(2, dim=-1)
        spectrogram = torch.cat([real, imaginary], dim=1).squeeze(-1)

        spectrogram = spectrogram.to(audio.dtype)
        if num_channels > 1:
            spectrogram = spectrogram.reshape(batch_size, num_channels * spectrogram.shape[1], spectrogram.shape[2])

        hidden_states = self.layers(spectrogram)
        return hidden_states.transpose(1, 2)


# Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckResidualUnit with Oobleck->Cosmos3Audio
class Cosmos3AudioResidualUnit(nn.Module):
    """
    A residual unit composed of Snake1d and weight-normalized Conv1d layers with dilations.
    """

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state):
        """
        Forward pass through the residual unit.

        Args:
            hidden_state (`torch.Tensor` of shape `(batch_size, channels, time_steps)`):
                Input tensor .

        Returns:
            output_tensor (`torch.Tensor` of shape `(batch_size, channels, time_steps)`)
                Input tensor after passing through the residual unit.
        """
        output_tensor = hidden_state
        output_tensor = self.conv1(self.snake1(output_tensor))
        output_tensor = self.conv2(self.snake2(output_tensor))

        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        output_tensor = hidden_state + output_tensor
        return output_tensor


"""
Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckDecoderBlock with Oobleck->Cosmos3Audio with
output_padding enabled.
"""


class Cosmos3AudioDecoderBlock(nn.Module):
    """Decoder block used in Cosmos3Audio decoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1, output_padding: int = 0):
        super().__init__()

        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = Cosmos3AudioResidualUnit(output_dim, dilation=1)
        self.res_unit2 = Cosmos3AudioResidualUnit(output_dim, dilation=3)
        self.res_unit3 = Cosmos3AudioResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state):
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)

        return hidden_state


"""
Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckDecoder with Oobleck->Cosmos3Audio and one change
of adding "output_padding=stride % 2,"
"""


class Cosmos3AudioDecoder(nn.Module):
    """Cosmos3Audio Decoder"""

    def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
        super().__init__()

        strides = upsampling_ratios
        channel_multiples = [1] + channel_multiples

        # Add first conv layer
        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))

        # Add upsampling + MRF blocks
        block = []
        for stride_index, stride in enumerate(strides):
            block += [
                Cosmos3AudioDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            ]

        self.block = nn.ModuleList(block)
        output_dim = channels
        self.snake1 = Snake1d(output_dim)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)

        for layer in self.block:
            hidden_state = layer(hidden_state)

        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state


@dataclass
class Cosmos3AudioEncoderOutput(BaseOutput):
    """Output of `Cosmos3AVAEAudioTokenizer.encode`."""

    latent_dist: OobleckDiagonalGaussianDistribution


@dataclass
class Cosmos3AudioDecoderOutput(BaseOutput):
    """Output of `Cosmos3AVAEAudioTokenizer.forward`."""

    sample: torch.Tensor


class Cosmos3AVAEAudioTokenizer(ModelMixin, ConfigMixin):
    """Audio tokenizer for Cosmos3 sound generation.

    Wraps the Cosmos3 AVAE SpecConvNeXt encoder and Oobleck-style decoder used by the Cosmos3 omni model. The decoder
    API stays tensor-returning because ``Cosmos3OmniPipeline`` calls it directly when ``enable_sound=True``.

    Only the shipped AVAE configuration (``model_type="autoencoder_v2"``, waveform input, ``spec_convnext`` encoder,
    ``vae`` bottleneck, ``oobleck`` decoder, log-scale SnakeBeta, no latent normalization) is supported; any other
    value raises ``NotImplementedError``.

    Parameters:
        model_type (`str`, defaults to `"autoencoder_v2"`): AVAE model variant; only `"autoencoder_v2"` is supported.
        sampling_rate (`int`, defaults to `48000`): Audio sample rate in Hz.
        vocoder_input_dim (`int`, defaults to `64`): Latent channel count fed into the decoder
            (``== transformer sound_dim``).
        dec_dim (`int`, defaults to `320`): Base decoder channel count.
        dec_c_mults (`tuple[int, ...]`, defaults to `(1, 2, 4, 8, 16)`): Decoder channel multipliers.
        dec_strides (`tuple[int, ...]`, defaults to `(2, 4, 5, 6, 8)`): Decoder upsampling strides.
        dec_out_channels (`int`, defaults to `2`): Output audio channels (2 = stereo).
        stereo (`bool`, defaults to `True`):
            Whether the audio is stereo; doubles the encoder's effective channel count.
        use_wav_as_input (`bool`, defaults to `True`): Whether the encoder consumes raw waveforms; only `True` is
            supported.
        normalize_volume (`bool`, defaults to `True`): Whether `encode` peak-normalizes the waveform before encoding.
        hop_size (`int`, *optional*): Waveform→latent temporal compression factor used for `encode` padding. Defaults
            to `prod(dec_strides)` when `None`.
        input_channels (`int`, defaults to `1`): Per-channel encoder input count before the `stereo` doubling.
        enc_type (`str`, defaults to `"spec_convnext"`): Encoder type; only `"spec_convnext"` is supported.
        enc_dim (`int`, defaults to `192`): Base encoder channel count.
        enc_intermediate_dim (`int`, defaults to `768`): Unused; kept for config fidelity (ConvNeXt blocks use
            ``input_dim * 4``).
        enc_num_layers (`int`, defaults to `12`):
            Unused; kept for config fidelity (depth derives from `enc_num_blocks`).
        enc_num_blocks (`int`, defaults to `2`): ConvNeXt blocks per encoder downsampling stage.
        enc_n_fft (`int`, defaults to `64`): STFT FFT size for the encoder spectrogram front-end.
        enc_hop_length (`int`, defaults to `16`): STFT hop length for the encoder spectrogram front-end.
        enc_latent_dim (`int`, defaults to `128`):
            Encoder output channels; split into mean/scale by the VAE bottleneck (so ``enc_latent_dim == 2 *
            vocoder_input_dim``).
        enc_c_mults (`tuple[int, ...]`, defaults to `(1, 2, 4)`): Encoder channel multipliers per stage.
        enc_strides (`tuple[int, ...]`, defaults to `(4, 5, 6)`): Encoder downsampling strides per stage.
        enc_identity_init (`bool`, defaults to `False`): Whether to zero-init the ConvNeXt residual 1x1 convs.
        enc_use_snake (`bool`, defaults to `True`): Whether ConvNeXt blocks use SnakeBeta (else GELU).
        dec_type (`str`, defaults to `"oobleck"`): Decoder type; only `"oobleck"` is supported.
        dec_use_snake (`bool`, defaults to `True`): Whether the decoder uses SnakeBeta; only `True` is supported.
        dec_final_tanh (`bool`, defaults to `False`): Vestigial decoder tanh flag; only `False` is supported.
        dec_anti_aliasing (`bool`, defaults to `False`): Decoder anti-aliasing flag; only `False` is supported.
        dec_use_nearest_upsample (`bool`, defaults to `False`): Decoder upsample mode flag; only `False` is supported.
        dec_use_tanh_at_final (`bool`, defaults to `False`): Decoder final-tanh flag; only `False` is supported.
        bottleneck_type (`str`, defaults to `"vae"`): Bottleneck type; only `"vae"` is supported.
        bottleneck (`dict`, *optional*): Bottleneck config; if given, its `"type"` must be `"vae"`.
        activation (`str`, defaults to `"snakebeta"`): Activation family; only `"snakebeta"` is supported.
        snake_logscale (`bool`, defaults to `True`): Whether SnakeBeta parameters are log-scaled; only `True` is
            supported.
        anti_aliasing (`bool`, defaults to `False`): Global anti-aliasing flag; only `False` is supported.
        use_cuda_kernel (`bool`, defaults to `False`): Whether to use fused CUDA kernels; only `False` is supported.
        causal (`bool`, defaults to `False`):
            Whether convolutions are causal; only `False` is supported by the encoder.
        padding_mode (`str`, defaults to `"zeros"`): Convolution padding mode.
        latent_mean (`float` or `list[float]`, *optional*): Latent normalization mean; latent normalization is not
            implemented, so a non-`None` value raises ``NotImplementedError``.
        latent_std (`float` or `list[float]`, *optional*): Latent normalization std; latent normalization is not
            implemented, so a non-`None` value raises ``NotImplementedError``.
        encoder_enabled (`bool`, defaults to `True`): Whether to instantiate the encoder. Set to `False` (or
            auto-disabled on load) for decoder-only checkpoints, which cannot `encode`.
    """

    _supports_gradient_checkpointing = False
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        model_type: str = "autoencoder_v2",
        sampling_rate: int = 48000,
        vocoder_input_dim: int = 64,
        dec_dim: int = 320,
        dec_c_mults: tuple = (1, 2, 4, 8, 16),
        dec_strides: tuple = (2, 4, 5, 6, 8),
        dec_out_channels: int = 2,
        stereo: bool = True,
        use_wav_as_input: bool = True,
        normalize_volume: bool = True,
        hop_size: int | None = None,
        input_channels: int = 1,
        enc_type: str = "spec_convnext",
        enc_dim: int = 192,
        enc_intermediate_dim: int = 768,
        enc_num_layers: int = 12,
        enc_num_blocks: int = 2,
        enc_n_fft: int = 64,
        enc_hop_length: int = 16,
        enc_latent_dim: int = 128,
        enc_c_mults: tuple = (1, 2, 4),
        enc_strides: tuple = (4, 5, 6),
        enc_identity_init: bool = False,
        enc_use_snake: bool = True,
        dec_type: str = "oobleck",
        dec_use_snake: bool = True,
        dec_final_tanh: bool = False,
        dec_anti_aliasing: bool = False,
        dec_use_nearest_upsample: bool = False,
        dec_use_tanh_at_final: bool = False,
        bottleneck_type: str = "vae",
        bottleneck: dict | None = None,
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        anti_aliasing: bool = False,
        use_cuda_kernel: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
        latent_mean: float | list[float] | None = None,
        latent_std: float | list[float] | None = None,
        encoder_enabled: bool = True,
    ):
        super().__init__()

        if model_type != "autoencoder_v2":
            raise NotImplementedError(f"Cosmos3 AVAE model type {model_type!r} is not supported.")
        if not use_wav_as_input:
            raise NotImplementedError("Cosmos3 AVAE tokenizer only supports waveform input.")
        if enc_type != "spec_convnext":
            raise NotImplementedError(f"Cosmos3 AVAE encoder type {enc_type!r} is not supported.")
        if bottleneck is not None and bottleneck.get("type", bottleneck_type) != "vae":
            raise NotImplementedError("Cosmos3 AVAE tokenizer only supports the VAE bottleneck.")
        if bottleneck_type != "vae":
            raise NotImplementedError("Cosmos3 AVAE tokenizer only supports the VAE bottleneck.")
        if dec_type != "oobleck":
            raise NotImplementedError(f"Cosmos3 AVAE decoder type {dec_type!r} is not supported.")
        if (
            not dec_use_snake
            or dec_final_tanh
            or dec_anti_aliasing
            or dec_use_nearest_upsample
            or dec_use_tanh_at_final
        ):
            raise NotImplementedError("Cosmos3 AVAE decoder only supports the shipped Oobleck decoder configuration.")
        if activation != "snakebeta" or not snake_logscale or anti_aliasing or use_cuda_kernel:
            raise NotImplementedError("Cosmos3 AVAE tokenizer only supports the shipped SnakeBeta configuration.")
        if latent_mean is not None or latent_std is not None:
            raise NotImplementedError(
                "Cosmos3 AVAE tokenizer does not apply latent normalization; `latent_mean`/`latent_std` must be None."
            )

        self.encoder = None
        self._encoder_available = False
        if encoder_enabled:
            self.encoder = Cosmos3AudioSpectrogramConvNeXtEncoder(
                input_channels=input_channels,
                stereo=stereo,
                channels=enc_dim,
                latent_dim=enc_latent_dim,
                channel_multiples=tuple(enc_c_mults),
                strides=tuple(enc_strides),
                num_blocks=enc_num_blocks,
                n_fft=enc_n_fft,
                hop_length=enc_hop_length,
                identity_init=enc_identity_init,
                use_snake=enc_use_snake,
                causal=causal,
                padding_mode=padding_mode,
            )
            self._encoder_available = True

        self.decoder = Cosmos3AudioDecoder(
            channels=dec_dim,
            input_channels=vocoder_input_dim,
            audio_channels=dec_out_channels,
            upsampling_ratios=list(reversed(dec_strides)),
            channel_multiples=list(dec_c_mults),
        )

        self._hop_size: int = int(hop_size) if hop_size is not None else math.prod(dec_strides)

    def _disable_encoder(self):
        self.encoder = None
        self._encoder_available = False
        self.register_to_config(encoder_enabled=False)

    def _fix_state_dict_keys_on_load(self, state_dict: OrderedDict) -> None:
        super()._fix_state_dict_keys_on_load(state_dict)
        if self.encoder is not None and not any(key.startswith("encoder.") for key in state_dict):
            self._disable_encoder()

    def _encode(self, sample: torch.Tensor) -> torch.Tensor:
        return self.encoder(sample).transpose(1, 2)

    @apply_forward_hook
    def encode(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
        force_pad: bool = False,
    ) -> Cosmos3AudioEncoderOutput | tuple[OobleckDiagonalGaussianDistribution]:
        """Encode a waveform into a VAE latent distribution.

        Args:
            sample: Audio waveform tensor with shape ``[B, C, T]``.
            return_dict: Whether to return a ``Cosmos3AudioEncoderOutput``.
            force_pad: Whether to right-pad to ``hop_size`` even when the model is in training mode.
        """
        if sample.ndim != 3:
            raise ValueError(f"`sample` must have shape [B, C, T], got {tuple(sample.shape)}.")

        if self.encoder is None or not self._encoder_available:
            raise ValueError(
                "This Cosmos3 AVAE sound tokenizer was loaded from decoder-only weights and cannot encode audio. "
                "Re-convert the AVAE checkpoint with encoder weights to use `encode()`."
            )

        hidden_states = sample
        if self.config.normalize_volume:
            hidden_states = hidden_states / (hidden_states.abs().max() + 1e-5) * 0.95

        if force_pad or not self.training:
            sample_length = hidden_states.shape[-1]
            padding = (self._hop_size - (sample_length % self._hop_size)) % self._hop_size
            if padding > 0:
                hidden_states = F.pad(hidden_states, (0, padding), mode="constant", value=0)

        encoder_dtype = get_parameter_dtype(self.encoder)
        moments = self._encode(hidden_states.to(dtype=encoder_dtype))
        posterior = OobleckDiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return Cosmos3AudioEncoderOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sound latents into an audio waveform.

        Args:
            latents: ``[B, C, T]`` or ``[C, T]`` tensor of diffusion-model latents.

        Returns:
            Waveform tensor ``[B, audio_channels, N]`` or ``[audio_channels, N]``.
        """
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        audio = self.decoder(latents).clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
        force_pad: bool = False,
    ) -> Cosmos3AudioDecoderOutput | tuple[torch.Tensor]:
        """Encode then decode a waveform; ``sample_posterior=False`` (default) decodes the distribution mode (mean),
        whereas the upstream Cosmos3 AVAE always samples — pass ``sample_posterior=True`` for reference-equivalent
        behavior."""
        posterior = self.encode(sample, force_pad=force_pad).latent_dist
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        decoded = self.decode(latents)

        if not return_dict:
            return (decoded,)

        return Cosmos3AudioDecoderOutput(sample=decoded)
