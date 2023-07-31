# Copyright 2023 Ollin Boer Bohan and The HuggingFace Team. All rights reserved.
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


from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, apply_forward_hook, is_torch_version
from .activations import get_activation
from .modeling_utils import ModelMixin
from .vae import DecoderOutput


@dataclass
class TinyAutoencoderOutput(BaseOutput):
    """
    Output of TinyAutoencoder encoding method.

    Args:
        latents (`torch.Tensor`): Encoded outputs of the `Encoder`.

    """

    latents: torch.Tensor


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class TinyAutoencoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn: Callable):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class TinyEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        block_out_channels: int,
        act_fn: Callable,
    ):
        super().__init__()

        layers = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]

            if i == 0:
                layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=2, bias=False))

            for _ in range(num_block):
                layers.append(TinyAutoencoderBlock(num_channels, num_channels, act_fn))

        layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            x = self.layers(x)

        return x


class TinyDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        block_out_channels: int,
        upsampling_scaling_factor: int,
        act_fn: Callable,
    ):
        super().__init__()

        layers = [Clamp(), nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1), act_fn]

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]

            for _ in range(num_block):
                layers.append(TinyAutoencoderBlock(num_channels, num_channels, act_fn))

            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor))

            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(nn.Conv2d(num_channels, conv_out_channel, kernel_size=3, padding=1, bias=is_final_block))

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            x = self.layers(x)

        return x


class TinyAutoencoder(ModelMixin, ConfigMixin):
    r"""
    A tiny VAE model for encoding images into latents and decoding latent representations into images. It was distilled
    by Ollin Boer Bohan as detailed in [https://github.com/madebyollin/taesd](https://github.com/madebyollin/taesd).

    [`TinyAutoencoder`] is just wrapper around the original implementation of `TAESD` found in the above-mentioned
    repository.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        TODO(sayakpaul): Fill rest of the docstrings.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        encoder_block_out_channels: Tuple[int] = (64, 64, 64, 64),
        decoder_block_out_channels: Tuple[int] = (64, 64, 64, 64),
        act_fn: str = "relu",
        latent_channels: int = 4,
        upsampling_scaling_factor: int = 2,
        num_encoder_blocks: Tuple[int] = (1, 3, 3, 3),
        num_decoder_blocks: Tuple[int] = (3, 3, 3, 1),
        latent_magnitude: int = 3,
        latent_shift: float = 0.5,
        force_upcast: float = True,
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        act_fn = get_activation(act_fn)

        self.encoder = TinyEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )

        self.decoder = TinyDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
        )

        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (TinyEncoder, TinyDecoder)):
            module.gradient_checkpointing = value

    def scale_latents(self, x):
        """raw latents -> [0, 1]"""
        return x.div(2 * self.latent_magnitude).add(self.latent_shift).clamp(0, 1)

    def unscale_latents(self, x):
        """[0, 1] -> raw latents"""
        return x.sub(self.latent_shift).mul(2 * self.latent_magnitude)

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[TinyAutoencoderOutput, Tuple[torch.FloatTensor]]:
        output = self.encoder(x)

        if not return_dict:
            return (output,)

        return TinyAutoencoderOutput(latents=output)

    apply_forward_hook

    def decode(self, x: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        output = self.decoder(x)

        if not return_dict:
            return (output,)

        return DecoderOutput(sample=output)

    def forward(
        self,
        x: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        enc = self.encode(x)
        scaled_enc = self.scale_latents(enc).mul_(255).round_().byte()
        unscaled_enc = self.unscale_latents(scaled_enc)
        dec = self.decode(unscaled_enc)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
