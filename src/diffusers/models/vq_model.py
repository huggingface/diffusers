# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.accelerate_utils import apply_forward_hook
from .modeling_utils import ModelMixin
from .vae import Decoder, DecoderOutput, Encoder, VectorQuantizer


@dataclass
class VQEncoderOutput(BaseOutput):
    """
    Output of VQModel encoding method.

    Args:
        latents (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The encoded output sample from the last layer of the model.
    """

    latents: torch.FloatTensor


class VQModel(ModelMixin, ConfigMixin):
    r"""
    A VQ-VAE model for decoding latent representations.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `1`): Number of layers per block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
        norm_num_groups (`int`, *optional*, defaults to `32`): Number of groups for normalization layers.
        vq_embed_dim (`int`, *optional*): Hidden dim of codebook vectors in the VQ-VAE.
        scaling_factor (`float`, *optional*, defaults to `0.18215`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        norm_type (`str`, *optional*, defaults to `"group"`):
            Type of normalization layer to use. Can be one of `"group"` or `"spatial"`.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: Optional[int] = None,
        scaling_factor: float = 0.18215,
        norm_type: str = "group",  # group, spatial
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_type=norm_type,
        )

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> VQEncoderOutput:
        h = self.encoder(x)
        h = self.quant_conv(h)

        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self, h: torch.FloatTensor, force_not_quantize: bool = False, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        # also go through quantization layer
        if not force_not_quantize:
            quant, _, _ = self.quantize(h)
        else:
            quant = h
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant if self.config.norm_type == "spatial" else None)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(self, sample: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        The [`VQModel`] forward method.

        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vq_model.VQEncoderOutput`] is returned, otherwise a plain `tuple`
                is returned.
        """
        x = sample
        h = self.encode(x).latents
        dec = self.decode(h).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
