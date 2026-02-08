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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import deprecate
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


def get_norm_layer_2d(
        in_channels: int,
        num_groups: int = 32,
        **kwargs
    ) -> nn.GroupNorm:
    """
    Creates a 2D GroupNorm normalization layer.
    """

    return nn.GroupNorm(num_channels=in_channels, num_groups=num_groups, eps=1e-6, affine=True)


class KVAEResnetBlock2D(nn.Module):
    r"""
    A Resnet block with optional guidance.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        conv_shortcut (`bool`, *optional*, default to `False`):
            If `True` and `in_channels` not equal to `out_channels`, add a 3x3 nn.conv2d layer for skip-connection.
        temb_channels (`int`, *optional*, default to `512`): The number of channels in timestep embedding.
        zq_ch (`int`, *optional*, default to `None`): Guidance channels for normalization.
        add_conv (`bool`, *optional*, default to `False`):
            If `True` add conv2d layer for normalization.
        normalization (`nn.Module`, *optional*, default to `None`): The normalization layer.
        act_fn (`str`, *optional*, default to `"swish"`): The activation function to use.
        
    """
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        temb_channels: int = 512,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
        normalization: nn.Module = get_norm_layer_2d,
        act_fn: str = 'swish'
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.nonlinearity = get_activation(act_fn)

        self.norm1 = normalization(in_channels, zq_channels=zq_ch, add_conv=add_conv)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(out_channels, zq_channels=zq_ch, add_conv=add_conv)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=(1, 1),
                    padding_mode="replicate",
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor, zq: torch.Tensor = None) -> torch.Tensor:
        h = x

        if zq is None:
            h = self.norm1(h)
        else:
            h = self.norm1(h, zq)

        h = self.nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is None:
            h = self.norm2(h)
        else:
            h = self.norm2(h, zq)

        h = self.nonlinearity(h)

        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class KVAEPXSDownsample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            factor: int = 2
        ):
        r"""
        A Downsampling module.

        Args:
            in_channels (`int`): The number of channels in the input.
            factor (`int`, *optional*, default to `2`): The downsampling factor.
        """
        super().__init__()
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.spatial_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="reflect"
        )
        self.linear = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bchw)
        pxs_interm = self.unshuffle(x)
        b, c, h, w = pxs_interm.shape
        pxs_interm_view = pxs_interm.view(b, c // self.factor**2, self.factor**2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)

        conv_out = self.spatial_conv(x)

        # adding it all together
        out = conv_out + pxs_out
        return self.linear(out)


class KVAEPXSUpsample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            factor: int = 2
        ):
        r"""
        An Upsampling module.

        Args:
            in_channels (`int`): The number of channels in the input.
            factor (`int`, *optional*, default to `2`): The upsampling factor.
        """
        super().__init__()
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"
        )

        self.linear = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        repeated = x.repeat_interleave(self.factor**2, dim=1)
        pxs_interm = self.shuffle(repeated)

        image_like_ups = F.interpolate(x, scale_factor=2, mode="nearest")
        conv_out = self.spatial_conv(image_like_ups)

        # adding it all together
        out = conv_out + pxs_interm
        return self.linear(out)


class KVAEDecoderSpacialNorm2D(nn.Module):
    r"""
    A 2D normalization module for decoder.

    Args:
        in_channels (`int`): The number of channels in the input.
        zq_channels (`int`): The number of channels in the guidance.
        add_conv (`bool`, *optional*, default to `false`): If `True` add conv2d 3x3 layer for guidance in the beginning.
    """
    def __init__(
        self,
        in_channels: int,
        zq_channels: int,
        add_conv: bool = False,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = get_norm_layer_2d(in_channels, **norm_layer_params)

        self.add_conv = add_conv
        if add_conv:
            self.conv = nn.Conv2d(
                in_channels=zq_channels,
                out_channels=zq_channels,
                kernel_size=3,
                padding=(1, 1),
                padding_mode="replicate",
            )

        self.conv_y = nn.Conv2d(
            in_channels=zq_channels,
            out_channels=in_channels,
            kernel_size=1,
        )
        self.conv_b = nn.Conv2d(
            in_channels=zq_channels,
            out_channels=in_channels,
            kernel_size=1,
        )

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_first = f
        f_first_size = f_first.shape[2:]
        zq = F.interpolate(zq, size=f_first_size, mode="nearest")

        if self.add_conv:
            zq = self.conv(zq)

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class KVAEEncoder2D(nn.Module):
    r"""
    A 2D encoder module.

    Args:
        ch (`int`): The base number of channels in multiresolution blocks.
        ch_mult (`Tuple[int, ...]`, *optional*, default to `(1, 2, 4, 8)`): 
            The channel multipliers in multiresolution blocks.
        num_res_blocks (`int`): The number of Resnet blocks.
        in_channels (`int`): The number of channels in the input.
        z_channels (`int`): The number of output channels.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
        act_fn (`str`, *optional*, default to `"swish"`): The activation function to use.
    """
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
        double_z: bool = True,
        act_fn: str = 'swish'
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * self.num_resolutions
        else:
            self.num_res_blocks = num_res_blocks
        self.nonlinearity = get_activation(act_fn)

        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.ch,
            kernel_size=3,
            padding=(1, 1),
        )

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    KVAEResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_resolutions - 1:
                down.downsample = KVAEPXSDownsample(in_channels=block_in)  # mb: bad out channels
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = KVAEResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
        )

        self.mid.block_2 = KVAEResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
        )

        # end
        self.norm_out = get_norm_layer_2d(block_in)

        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)

        return h


class KVAEDecoder2D(nn.Module):
    r"""
    A 2D decoder module.

    Args:
        ch (`int`): The base number of channels in multiresolution blocks.
        out_ch (`int`): The number of output channels.
        ch_mult (`Tuple[int, ...]`, *optional*, default to `(1, 2, 4, 8)`): 
            The channel multipliers in multiresolution blocks.
        num_res_blocks (`int`): The number of Resnet blocks.
        in_channels (`int`): The number of channels in the input.
        z_channels (`int`): The number of input channels.
        give_pre_end (`bool`, *optional*, default to `false`):
            If `True` exit the forward pass early and return the penultimate feature map.
        zq_ch (`bool`, *optional*, default to `None`): The number of channels in the guidance.
        add_conv (`bool`, *optional*, default to `false`): If `True` add conv2d layer for Resnet normalization layer.
        act_fn (`str`, *optional*, default to `"swish"`): The activation function to use.
    """
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
        give_pre_end: bool = False,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
        act_fn: str = 'swish'
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.nonlinearity = get_activation(act_fn)

        if zq_ch is None:
            zq_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = nn.Conv2d(
            in_channels=z_channels, out_channels=block_in, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = KVAEResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=KVAEDecoderSpacialNorm2D,
        )

        self.mid.block_2 = KVAEResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=KVAEDecoderSpacialNorm2D,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    KVAEResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        normalization=KVAEDecoderSpacialNorm2D,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = KVAEPXSUpsample(in_channels=block_in)
            self.up.insert(0, up)

        self.norm_out =KVAEDecoderSpacialNorm2D(block_in, zq_ch, add_conv=add_conv)  # , gather=gather_norm)

        self.conv_out = nn.Conv2d(
            in_channels=block_in, out_channels=out_ch, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        t = z.shape[2]
        # z to block_in

        zq = z
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.block_2(h, temb, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)

                # h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = self.nonlinearity(h)
        h = self.conv_out(h)

        return h


class AutoencoderKLKVAE(
    ModelMixin, AutoencoderMixin, ConfigMixin
):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        channels (int,  *optional*, defaults to 128): The base number of channels in multiresolution blocks.
        num_enc_blocks (int, *optional*, defaults to 2): The number of Resnet blocks in encoder multiresolution layers.
        num_dec_blocks (int, *optional*, defaults to 2): The number of Resnet blocks in decoder multiresolution layers.
        z_channels (int, *optional*, defaults to 16): Number of channels in the latent space.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels of encoder.
        ch_mult (`Tuple[int, ...]`, *optional*, default to `(1, 2, 4, 8)`): 
            The channel multipliers in multiresolution blocks.
        bottleneck (nn.Module, *optional*, defaults to `None`): Bottleneck module of VAE.
        sample_size (`int`, *optional*, defaults to `1024`): Sample input size.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 128,
        num_enc_blocks: int = 2,
        num_dec_blocks: int = 2,
        z_channels: int = 16,
        double_z: bool = True,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        bottleneck: Optional[nn.Module] = None,
        sample_size: int = 1024,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = KVAEEncoder2D(
            in_channels=in_channels,
            ch=channels,
            ch_mult=ch_mult,
            num_res_blocks=num_enc_blocks,
            z_channels=z_channels,
            double_z=double_z,
        )

        # pass init params to Decoder
        self.decoder = KVAEDecoder2D(
            out_ch=in_channels,
            ch=channels,
            ch_mult=ch_mult,
            num_res_blocks=num_dec_blocks,
            in_channels=None,
            z_channels=z_channels,
        )

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.ch_mult) - 1)))
        self.tile_overlap_factor = 0.25


    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self._tiled_encode(x)

        enc = self.encoder(x)

        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        enc = torch.cat(result_rows, dim=2)
        return enc

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        deprecation_message = (
            "The tiled_encode implementation supporting the `return_dict` parameter is deprecated. In the future, the "
            "implementation of this method will be replaced with that of `_tiled_encode` and you will no longer be able "
            "to pass `return_dict`. You will also have to create a `DiagonalGaussianDistribution()` from the returned value."
        )
        deprecate("tiled_encode", "1.0.0", deprecation_message, standard_warn=False)

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
