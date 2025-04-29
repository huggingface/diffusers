# Copyright (c) 2022 Dominic Rampas MIT License
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.autoencoders.vae import DecoderOutput, VectorQuantizer
from ...models.modeling_utils import ModelMixin
from ...models.vq_model import VQEncoderOutput
from ...utils.accelerate_utils import apply_forward_hook


class MixingResidualBlock(nn.Module):
    """
    Residual block with mixing used by Paella's VQ-VAE.
    """

    def __init__(self, inp_channels, embed_dim):
        super().__init__()
        # depthwise
        self.norm1 = nn.LayerNorm(inp_channels, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Sequential(
            nn.ReplicationPad2d(1), nn.Conv2d(inp_channels, inp_channels, kernel_size=3, groups=inp_channels)
        )

        # channelwise
        self.norm2 = nn.LayerNorm(inp_channels, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(inp_channels, embed_dim), nn.GELU(), nn.Linear(embed_dim, inp_channels)
        )

        self.gammas = nn.Parameter(torch.zeros(6), requires_grad=True)

    def forward(self, x):
        mods = self.gammas
        x_temp = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * (1 + mods[0]) + mods[1]
        x = x + self.depthwise(x_temp) * mods[2]
        x_temp = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * (1 + mods[3]) + mods[4]
        x = x + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]
        return x


class PaellaVQModel(ModelMixin, ConfigMixin):
    r"""VQ-VAE model from Paella model.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        up_down_scale_factor (int, *optional*, defaults to 2): Up and Downscale factor of the input image.
        levels  (int, *optional*, defaults to 2): Number of levels in the model.
        bottleneck_blocks (int, *optional*, defaults to 12): Number of bottleneck blocks in the model.
        embed_dim (int, *optional*, defaults to 384): Number of hidden channels in the model.
        latent_channels (int, *optional*, defaults to 4): Number of latent channels in the VQ-VAE model.
        num_vq_embeddings (int, *optional*, defaults to 8192): Number of codebook vectors in the VQ-VAE.
        scale_factor (float, *optional*, defaults to 0.3764): Scaling factor of the latent space.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_down_scale_factor: int = 2,
        levels: int = 2,
        bottleneck_blocks: int = 12,
        embed_dim: int = 384,
        latent_channels: int = 4,
        num_vq_embeddings: int = 8192,
        scale_factor: float = 0.3764,
    ):
        super().__init__()

        c_levels = [embed_dim // (2**i) for i in reversed(range(levels))]
        # Encoder blocks
        self.in_block = nn.Sequential(
            nn.PixelUnshuffle(up_down_scale_factor),
            nn.Conv2d(in_channels * up_down_scale_factor**2, c_levels[0], kernel_size=1),
        )
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = MixingResidualBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(
            nn.Sequential(
                nn.Conv2d(c_levels[-1], latent_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(latent_channels),  # then normalize them to have mean 0 and std 1
            )
        )
        self.down_blocks = nn.Sequential(*down_blocks)

        # Vector Quantizer
        self.vquantizer = VectorQuantizer(num_vq_embeddings, vq_embed_dim=latent_channels, legacy=False, beta=0.25)

        # Decoder blocks
        up_blocks = [nn.Sequential(nn.Conv2d(latent_channels, c_levels[-1], kernel_size=1))]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = MixingResidualBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(
                    nn.ConvTranspose2d(
                        c_levels[levels - 1 - i], c_levels[levels - 2 - i], kernel_size=4, stride=2, padding=1
                    )
                )
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            nn.Conv2d(c_levels[0], out_channels * up_down_scale_factor**2, kernel_size=1),
            nn.PixelShuffle(up_down_scale_factor),
        )

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
        h = self.in_block(x)
        h = self.down_blocks(h)

        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self, h: torch.Tensor, force_not_quantize: bool = True, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        if not force_not_quantize:
            quant, _, _ = self.vquantizer(h)
        else:
            quant = h

        x = self.up_blocks(quant)
        dec = self.out_block(x)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        h = self.encode(x).latents
        dec = self.decode(h).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
