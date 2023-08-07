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
from typing import Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, apply_forward_hook
from .modeling_utils import ModelMixin
from .vae import DecoderOutput, DecoderTiny, EncoderTiny


@dataclass
class AutoencoderTinyOutput(BaseOutput):
    """
    Output of AutoencoderTiny encoding method.

    Args:
        latents (`torch.Tensor`): Encoded outputs of the `Encoder`.

    """

    latents: torch.Tensor


class AutoencoderTiny(ModelMixin, ConfigMixin):
    r"""
    A tiny distilled VAE model for encoding images into latents and decoding latent representations into images.

    [`AutoencoderTiny`] is a wrapper around the original implementation of `TAESD`.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented for
    all models (such as downloading or saving).

    Parameters:
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`,  *optional*, defaults to 3): Number of channels in the output.
        encoder_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64, 64, 64, 64)`):
            Tuple of integers representing the number of output channels for each encoder block. The length of the
            tuple should be equal to the number of encoder blocks.
        decoder_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64, 64, 64, 64)`):
            Tuple of integers representing the number of output channels for each decoder block. The length of the
            tuple should be equal to the number of decoder blocks.
        act_fn (`str`, *optional*, defaults to `"relu"`):
            Activation function to be used throughout the model.
        latent_channels (`int`, *optional*, defaults to 4):
            Number of channels in the latent representation. The latent space acts as a compressed representation of
            the input image.
        upsampling_scaling_factor (`int`, *optional*, defaults to 2):
            Scaling factor for upsampling in the decoder. It determines the size of the output image during the
            upsampling process.
        num_encoder_blocks (`Tuple[int]`, *optional*, defaults to `(1, 3, 3, 3)`):
            Tuple of integers representing the number of encoder blocks at each stage of the encoding process. The
            length of the tuple should be equal to the number of stages in the encoder. Each stage has a different
            number of encoder blocks.
        num_decoder_blocks (`Tuple[int]`, *optional*, defaults to `(3, 3, 3, 1)`):
            Tuple of integers representing the number of decoder blocks at each stage of the decoding process. The
            length of the tuple should be equal to the number of stages in the decoder. Each stage has a different
            number of decoder blocks.
        latent_magnitude (`float`, *optional*, defaults to 3.0):
            Magnitude of the latent representation. This parameter scales the latent representation values to control
            the extent of information preservation.
        latent_shift (float, *optional*, defaults to 0.5):
            Shift applied to the latent representation. This parameter controls the center of the latent space.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper. For this Autoencoder,
            however, no such scaling factor was used, hence the value of 1.0 as the default.
        force_upcast (`bool`, *optional*, default to `False`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without losing too much precision, in which case
            `force_upcast` can be set to `False` (see this fp16-friendly
            [AutoEncoder](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).
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
        force_upcast: float = False,
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        self.encoder = EncoderTiny(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )

        self.decoder = DecoderTiny(
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
        if isinstance(module, (EncoderTiny, DecoderTiny)):
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
    ) -> Union[AutoencoderTinyOutput, Tuple[torch.FloatTensor]]:
        output = self.encoder(x)

        if not return_dict:
            return (output,)

        return AutoencoderTinyOutput(latents=output)

    @apply_forward_hook
    def decode(self, x: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        output = self.decoder(x)
        # Refer to the following discussion to know why this is needed.
        # https://github.com/huggingface/diffusers/pull/4384#discussion_r1279401854
        output = output.mul_(2).sub_(1)

        if not return_dict:
            return (output,)

        return DecoderOutput(sample=output)

    def forward(
        self,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        enc = self.encode(sample).latents
        scaled_enc = self.scale_latents(enc).mul_(255).round_().byte()
        unscaled_enc = self.unscale_latents(scaled_enc)
        dec = self.decode(unscaled_enc)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
