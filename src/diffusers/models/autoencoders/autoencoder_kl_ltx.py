# Copyright 2024 The Mochi team and The HuggingFace Team.
# All rights reserved.
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

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .autoencoder_kl_cogvideox import CogVideoXCausalConv3d, CogVideoXMidBlock3D, CogVideoXResnetBlock3D, _get_norm
from .vae import DecoderOutput, DiagonalGaussianDistribution


# {'_class_name': 'CausalVideoAutoencoder', 'dims': 3, 'in_channels': 3, 'out_channels': 3, 'latent_channels': 128,
# 'blocks': [
#   ['res_x', 4], ['compress_all', 1], ['res_x_y', 1],
#   ['res_x', 3], ['compress_all', 1], ['res_x_y', 1],
#   ['res_x', 3], ['compress_all', 1],
#   ['res_x', 3],
#   ['res_x', 4]],
# 'scaling_factor': 1.0, 'norm_layer': 'pixel_norm', 'patch_size': 4, 'latent_log_var': 'uniform', 'use_quant_conv': False, 'causal_decoder': False}


class LTXDownsampler3D(CogVideoXCausalConv3d):
    pass


class LTXUpsampler3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)

        out_channels = in_channels * stride[0] * stride[1] * stride[2]

        self.conv = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states, _ = self.conv(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)

        hidden_states = hidden_states[:, :, self.stride[0] - 1 :]
        return hidden_states


class LTXDownBlock3D(nn.Module):
    r"""
    Down block used in the LTX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        add_downsample (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_norm_type: str = "rms_norm",
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatio_temporal_scale: bool = True,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    norm_type=resnet_norm_type,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList(
                [LTXDownsampler3D(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=(2, 2, 2))]
            )

        self.conv_out = None
        if in_channels != out_channels:
            self.conv_out = CogVideoXResnetBlock3D(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                temb_channels=temb_channels,
                norm_type=resnet_norm_type,
                final_norm_type="layer_norm",
                groups=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward method of the `LTXDownBlock3D` class."""

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, _ = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states)
            else:
                hidden_states, _ = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states, _ = downsampler(hidden_states)

        if self.conv_out is not None:
            hidden_states, _ = self.conv_out(hidden_states)

        return hidden_states


class LTXUpBlock3D(nn.Module):
    r"""
    Up block used in the LTX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        add_downsample (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_norm_type: str = "rms_norm",
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatio_temporal_scale: bool = True,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = CogVideoXResnetBlock3D(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                temb_channels=temb_channels,
                norm_type=resnet_norm_type,
                final_norm_type="layer_norm",
                groups=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList([LTXUpsampler3D(out_channels, stride=(2, 2, 2))])

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    norm_type=resnet_norm_type,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward method of the `LTXDownBlock3D` class."""

        if self.conv_in is not None:
            hidden_states, _ = self.conv_in(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, _ = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states)
            else:
                hidden_states, _ = resnet(hidden_states)

        return hidden_states


class LTXEncoder3D(nn.Module):
    r"""
    The `LTXEncoder3D` layer of a variational autoencoder that encodes input video samples to its latent
    representation.

    Args:
        TODO(aryan)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_type: str = "rms_norm",
        resnet_groups: int = 32,
        resnet_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size**2

        output_channel = block_out_channels[0]

        self.conv_in = CogVideoXCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
        )

        # down blocks
        num_block_out_channels = len(block_out_channels)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i + 1] if i + 1 < num_block_out_channels else block_out_channels[i]

            down_block = LTXDownBlock3D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                num_layers=layers_per_block[i],
                resnet_norm_type=resnet_norm_type,
                resnet_eps=resnet_norm_eps,
                resnet_groups=resnet_groups,
                spatio_temporal_scale=spatio_temporal_scaling[i],
            )

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=output_channel,
            temb_channels=None,
            num_layers=layers_per_block[-1],
            norm_type=resnet_norm_type,
            resnet_eps=resnet_norm_eps,
            resnet_groups=resnet_groups,
        )

        # out
        self.norm_out = _get_norm(resnet_norm_type, output_channel, eps=1e-6, elementwise_affine=False, channel_dim=1)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            in_channels=output_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            stride=1,
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `LTXEncoder3D` class."""

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = hidden_states.reshape(
            batch_size, -1, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
        )
        hidden_states, _ = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def create_forward(*inputs):
                    return module(*inputs)

                return create_forward

            for down_block in self.down_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block),
                    hidden_states,
                )
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states, _ = self.conv_out(hidden_states)

        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)

        return hidden_states


class LTXDecoder3D(nn.Module):
    r"""
    The `LTXDecoder3D` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        TODO(aryan)
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_type: str = "rms_norm",
        resnet_groups: int = 32,
        resnet_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size**2

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))

        output_channel = block_out_channels[0]

        self.conv_in = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
        )

        self.mid_block = CogVideoXMidBlock3D(
            in_channels=output_channel,
            temb_channels=None,
            num_layers=layers_per_block[0],
            norm_type=resnet_norm_type,
            resnet_eps=resnet_norm_eps,
            resnet_groups=resnet_groups,
        )

        # up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            up_block = LTXUpBlock3D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                num_layers=layers_per_block[i + 1],
                resnet_norm_type=resnet_norm_type,
                resnet_eps=resnet_norm_eps,
                resnet_groups=resnet_groups,
                spatio_temporal_scale=spatio_temporal_scaling[i],
            )

            self.up_blocks.append(up_block)

        # out
        self.norm_out = _get_norm(resnet_norm_type, output_channel, eps=1e-6, elementwise_affine=False, channel_dim=1)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            in_channels=output_channel,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def create_forward(*inputs):
                    return module(*inputs)

                return create_forward

            for up_block in self.up_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    hidden_states,
                )
        else:
            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states, _ = self.conv_out(hidden_states, causal=self.causal)

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 2)

        return hidden_states


class AutoencoderKLLTX(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. Used in
    [LTX](https://huggingface.co/Lightricks/LTX-Video).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Args:
        TODO(aryan)
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_type: str = "rms_norm",
        resnet_groups: int = 32,
        resnet_norm_eps: float = 1e-6,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = LTXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_type=resnet_norm_type,
            resnet_groups=resnet_groups,
            resnet_norm_eps=resnet_norm_eps,
        )
        self.decoder = LTXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_type=resnet_norm_type,
            resnet_groups=resnet_groups,
            resnet_norm_eps=resnet_norm_eps,
        )

        self.spatial_compression_ratio = patch_size * patch_size
        self.temporal_compression_ratio = patch_size_t

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.num_latent_frames_batch_sizes`), the memory requirement can be lowered.
        self.use_framewise_encoding = True
        self.use_framewise_decoding = True

        # This can be configured based on the amount of GPU memory available.
        # `12` for sample frames and `2` for latent frames are sensible defaults for consumer GPUs.
        # Setting it to higher values results in higher memory usage.
        self.num_sample_frames_batch_size = 8
        self.num_latent_frames_batch_size = 2

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (LTXEncoder3D, LTXDecoder3D)):
            module.gradient_checkpointing = value

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        if self.use_framewise_encoding:
            enc = []
            for i in range(0, num_frames, self.num_sample_frames_batch_size):
                x_intermediate = x[:, :, i : i + self.num_sample_frames_batch_size]
                x_intermediate = self.encoder(x_intermediate)
                enc.append(x_intermediate)
            enc = torch.cat(enc, dim=2)
        else:
            enc, _ = self.encoder(x)

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
                The latent representations of the encoded videos. If `return_dict` is True, a
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
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.use_framewise_decoding:
            dec = []
            for i in range(0, num_frames, self.num_latent_frames_batch_size):
                z_intermediate = z[:, :, i : i + self.num_latent_frames_batch_size]
                z_intermediate = self.decoder(z_intermediate)
                dec.append(z_intermediate)
            dec = torch.cat(dec, dim=2)
        else:
            dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
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
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                if self.use_framewise_encoding:
                    time = []
                    for k in range(0, num_frames, self.num_sample_frames_batch_size):
                        tile = x[
                            :,
                            :,
                            k : k + self.num_sample_frames_batch_size,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                        tile = self.encoder(tile)
                        time.append(tile)
                else:
                    time = self.encoder(
                        x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    )

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

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

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                if self.use_framewise_decoding:
                    time = []
                    for k in range(0, num_frames, self.num_latent_frames_batch_size):
                        tile = z[
                            :,
                            :,
                            k : k + self.num_latent_frames_batch_size,
                            i : i + tile_latent_min_height,
                            j : j + tile_latent_min_width,
                        ]
                        tile = self.decoder(tile)
                        time.append(tile)
                    time = torch.cat(time, dim=2)
                else:
                    time = self.decoder(z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width])

                if self.drop_last_temporal_frames and time.size(2) >= self.temporal_compression_ratio:
                    time = time[:, :, self.temporal_compression_ratio - 1 :]

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if not return_dict:
            return (dec,)
        return dec
