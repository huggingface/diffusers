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

import functools
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..attention_processor import Attention, MochiVaeAttnProcessor2_0
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .autoencoder_kl_cogvideox import CogVideoXCausalConv3d
from .vae import DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MochiChunkedGroupNorm3D(nn.Module):
    r"""
    Applies per-frame group normalization for 5D video inputs. It also supports memory-efficient chunked group
    normalization.

    Args:
        num_channels (int): Number of channels expected in input
        num_groups (int, optional): Number of groups to separate the channels into. Default: 32
        affine (bool, optional): If True, this module has learnable affine parameters. Default: True
        chunk_size (int, optional): Size of each chunk for processing. Default: 8

    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        affine: bool = True,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, affine=affine)
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.size(0)

        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        output = torch.cat([self.norm_layer(chunk) for chunk in x.split(self.chunk_size, dim=0)], dim=0)
        output = output.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        return output


class MochiResnetBlock3D(nn.Module):
    r"""
    A 3D ResNet block used in the Mochi model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        non_linearity (`str`, defaults to `"swish"`):
            Activation function to use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        act_fn: str = "swish",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(act_fn)

        self.norm1 = MochiChunkedGroupNorm3D(num_channels=in_channels)
        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, pad_mode="replicate"
        )
        self.norm2 = MochiChunkedGroupNorm3D(num_channels=out_channels)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, pad_mode="replicate"
        )

    def forward(
        self,
        inputs: torch.Tensor,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(hidden_states, conv_cache=conv_cache.get("conv1"))

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(hidden_states, conv_cache=conv_cache.get("conv2"))

        hidden_states = hidden_states + inputs
        return hidden_states, new_conv_cache


class MochiDownBlock3D(nn.Module):
    r"""
    An downsampling block used in the Mochi model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet blocks in the block.
        temporal_expansion (`int`, defaults to `2`):
            Temporal expansion factor.
        spatial_expansion (`int`, defaults to `2`):
            Spatial expansion factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        add_attention: bool = True,
    ):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

        self.conv_in = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(temporal_expansion, spatial_expansion, spatial_expansion),
            stride=(temporal_expansion, spatial_expansion, spatial_expansion),
            pad_mode="replicate",
        )

        resnets = []
        norms = []
        attentions = []
        for _ in range(num_layers):
            resnets.append(MochiResnetBlock3D(in_channels=out_channels))
            if add_attention:
                norms.append(MochiChunkedGroupNorm3D(num_channels=out_channels))
                attentions.append(
                    Attention(
                        query_dim=out_channels,
                        heads=out_channels // 32,
                        dim_head=32,
                        qk_norm="l2",
                        is_causal=True,
                        processor=MochiVaeAttnProcessor2_0(),
                    )
                )
            else:
                norms.append(None)
                attentions.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.norms = nn.ModuleList(norms)
        self.attentions = nn.ModuleList(attentions)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
        chunk_size: int = 2**15,
    ) -> torch.Tensor:
        r"""Forward method of the `MochiUpBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(hidden_states)

        for i, (resnet, norm, attn) in enumerate(zip(self.resnets, self.norms, self.attentions)):
            conv_cache_key = f"resnet_{i}"

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    conv_cache=conv_cache.get(conv_cache_key),
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )

            if attn is not None:
                residual = hidden_states
                hidden_states = norm(hidden_states)

                batch_size, num_channels, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.permute(0, 3, 4, 2, 1).flatten(0, 2).contiguous()

                # Perform attention in chunks to avoid following error:
                # RuntimeError: CUDA error: invalid configuration argument
                if hidden_states.size(0) <= chunk_size:
                    hidden_states = attn(hidden_states)
                else:
                    hidden_states_chunks = []
                    for i in range(0, hidden_states.size(0), chunk_size):
                        hidden_states_chunk = hidden_states[i : i + chunk_size]
                        hidden_states_chunk = attn(hidden_states_chunk)
                        hidden_states_chunks.append(hidden_states_chunk)
                    hidden_states = torch.cat(hidden_states_chunks)

                hidden_states = hidden_states.unflatten(0, (batch_size, height, width)).permute(0, 4, 3, 1, 2)

                hidden_states = residual + hidden_states

        return hidden_states, new_conv_cache


class MochiMidBlock3D(nn.Module):
    r"""
    A middle block used in the Mochi model.

    Args:
        in_channels (`int`):
            Number of input channels.
        num_layers (`int`, defaults to `3`):
            Number of resnet blocks in the block.
    """

    def __init__(
        self,
        in_channels: int,  # 768
        num_layers: int = 3,
        add_attention: bool = True,
    ):
        super().__init__()

        resnets = []
        norms = []
        attentions = []

        for _ in range(num_layers):
            resnets.append(MochiResnetBlock3D(in_channels=in_channels))

            if add_attention:
                norms.append(MochiChunkedGroupNorm3D(num_channels=in_channels))
                attentions.append(
                    Attention(
                        query_dim=in_channels,
                        heads=in_channels // 32,
                        dim_head=32,
                        qk_norm="l2",
                        is_causal=True,
                        processor=MochiVaeAttnProcessor2_0(),
                    )
                )
            else:
                norms.append(None)
                attentions.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.norms = nn.ModuleList(norms)
        self.attentions = nn.ModuleList(attentions)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `MochiMidBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, (resnet, norm, attn) in enumerate(zip(self.resnets, self.norms, self.attentions)):
            conv_cache_key = f"resnet_{i}"

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )

            if attn is not None:
                residual = hidden_states
                hidden_states = norm(hidden_states)

                batch_size, num_channels, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.permute(0, 3, 4, 2, 1).flatten(0, 2).contiguous()
                hidden_states = attn(hidden_states)
                hidden_states = hidden_states.unflatten(0, (batch_size, height, width)).permute(0, 4, 3, 1, 2)

                hidden_states = residual + hidden_states

        return hidden_states, new_conv_cache


class MochiUpBlock3D(nn.Module):
    r"""
    An upsampling block used in the Mochi model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet blocks in the block.
        temporal_expansion (`int`, defaults to `2`):
            Temporal expansion factor.
        spatial_expansion (`int`, defaults to `2`):
            Spatial expansion factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
    ):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

        resnets = []
        for _ in range(num_layers):
            resnets.append(MochiResnetBlock3D(in_channels=in_channels))
        self.resnets = nn.ModuleList(resnets)

        self.proj = nn.Linear(in_channels, out_channels * temporal_expansion * spatial_expansion**2)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `MochiUpBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    conv_cache=conv_cache.get(conv_cache_key),
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        st = self.temporal_expansion
        sh = self.spatial_expansion
        sw = self.spatial_expansion

        # Reshape and unpatchify
        hidden_states = hidden_states.view(batch_size, -1, st, sh, sw, num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        hidden_states = hidden_states.view(batch_size, -1, num_frames * st, height * sh, width * sw)

        return hidden_states, new_conv_cache


class FourierFeatures(nn.Module):
    def __init__(self, start: int = 6, stop: int = 8, step: int = 1):
        super().__init__()

        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward method of the `FourierFeatures` class."""
        original_dtype = inputs.dtype
        inputs = inputs.to(torch.float32)
        num_channels = inputs.shape[1]
        num_freqs = (self.stop - self.start) // self.step

        freqs = torch.arange(self.start, self.stop, self.step, dtype=inputs.dtype, device=inputs.device)
        w = torch.pow(2.0, freqs) * (2 * torch.pi)  # [num_freqs]
        w = w.repeat(num_channels)[None, :, None, None, None]  # [1, num_channels * num_freqs, 1, 1, 1]

        # Interleaved repeat of input channels to match w
        h = inputs.repeat_interleave(num_freqs, dim=1)  # [B, C * num_freqs, T, H, W]
        # Scale channels by frequency.
        h = w * h

        return torch.cat([inputs, torch.sin(h), torch.cos(h)], dim=1).to(original_dtype)


class MochiEncoder3D(nn.Module):
    r"""
    The `MochiEncoder3D` layer of a variational autoencoder that encodes input video samples to its latent
    representation.

    Args:
        in_channels (`int`, *optional*):
            The number of input channels.
        out_channels (`int`, *optional*):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(128, 256, 512, 768)`):
            The number of output channels for each block.
        layers_per_block (`Tuple[int, ...]`, *optional*, defaults to `(3, 3, 4, 6, 3)`):
            The number of resnet blocks for each block.
        temporal_expansions (`Tuple[int, ...]`, *optional*, defaults to `(1, 2, 3)`):
            The temporal expansion factor for each of the up blocks.
        spatial_expansions (`Tuple[int, ...]`, *optional*, defaults to `(2, 2, 2)`):
            The spatial expansion factor for each of the up blocks.
        non_linearity (`str`, *optional*, defaults to `"swish"`):
            The non-linearity to use in the decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 768),
        layers_per_block: Tuple[int, ...] = (3, 3, 4, 6, 3),
        temporal_expansions: Tuple[int, ...] = (1, 2, 3),
        spatial_expansions: Tuple[int, ...] = (2, 2, 2),
        add_attention_block: Tuple[bool, ...] = (False, True, True, True, True),
        act_fn: str = "swish",
    ):
        super().__init__()

        self.nonlinearity = get_activation(act_fn)

        self.fourier_features = FourierFeatures()
        self.proj_in = nn.Linear(in_channels, block_out_channels[0])
        self.block_in = MochiMidBlock3D(
            in_channels=block_out_channels[0], num_layers=layers_per_block[0], add_attention=add_attention_block[0]
        )

        down_blocks = []
        for i in range(len(block_out_channels) - 1):
            down_block = MochiDownBlock3D(
                in_channels=block_out_channels[i],
                out_channels=block_out_channels[i + 1],
                num_layers=layers_per_block[i + 1],
                temporal_expansion=temporal_expansions[i],
                spatial_expansion=spatial_expansions[i],
                add_attention=add_attention_block[i + 1],
            )
            down_blocks.append(down_block)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.block_out = MochiMidBlock3D(
            in_channels=block_out_channels[-1], num_layers=layers_per_block[-1], add_attention=add_attention_block[-1]
        )
        self.norm_out = MochiChunkedGroupNorm3D(block_out_channels[-1])
        self.proj_out = nn.Linear(block_out_channels[-1], 2 * out_channels, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        r"""Forward method of the `MochiEncoder3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = self.fourier_features(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3)

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def create_forward(*inputs):
                    return module(*inputs)

                return create_forward

            hidden_states, new_conv_cache["block_in"] = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.block_in), hidden_states, conv_cache=conv_cache.get("block_in")
            )

            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )
        else:
            hidden_states, new_conv_cache["block_in"] = self.block_in(
                hidden_states, conv_cache=conv_cache.get("block_in")
            )

            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = down_block(
                    hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )

        hidden_states, new_conv_cache["block_out"] = self.block_out(
            hidden_states, conv_cache=conv_cache.get("block_out")
        )

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3)

        return hidden_states, new_conv_cache


class MochiDecoder3D(nn.Module):
    r"""
    The `MochiDecoder3D` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, *optional*):
            The number of input channels.
        out_channels (`int`, *optional*):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(128, 256, 512, 768)`):
            The number of output channels for each block.
        layers_per_block (`Tuple[int, ...]`, *optional*, defaults to `(3, 3, 4, 6, 3)`):
            The number of resnet blocks for each block.
        temporal_expansions (`Tuple[int, ...]`, *optional*, defaults to `(1, 2, 3)`):
            The temporal expansion factor for each of the up blocks.
        spatial_expansions (`Tuple[int, ...]`, *optional*, defaults to `(2, 2, 2)`):
            The spatial expansion factor for each of the up blocks.
        non_linearity (`str`, *optional*, defaults to `"swish"`):
            The non-linearity to use in the decoder.
    """

    def __init__(
        self,
        in_channels: int,  # 12
        out_channels: int,  # 3
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 768),
        layers_per_block: Tuple[int, ...] = (3, 3, 4, 6, 3),
        temporal_expansions: Tuple[int, ...] = (1, 2, 3),
        spatial_expansions: Tuple[int, ...] = (2, 2, 2),
        act_fn: str = "swish",
    ):
        super().__init__()

        self.nonlinearity = get_activation(act_fn)

        self.conv_in = nn.Conv3d(in_channels, block_out_channels[-1], kernel_size=(1, 1, 1))
        self.block_in = MochiMidBlock3D(
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block[-1],
            add_attention=False,
        )

        up_blocks = []
        for i in range(len(block_out_channels) - 1):
            up_block = MochiUpBlock3D(
                in_channels=block_out_channels[-i - 1],
                out_channels=block_out_channels[-i - 2],
                num_layers=layers_per_block[-i - 2],
                temporal_expansion=temporal_expansions[-i - 1],
                spatial_expansion=spatial_expansions[-i - 1],
            )
            up_blocks.append(up_block)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.block_out = MochiMidBlock3D(
            in_channels=block_out_channels[0],
            num_layers=layers_per_block[0],
            add_attention=False,
        )
        self.proj_out = nn.Linear(block_out_channels[0], out_channels)

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        r"""Forward method of the `MochiDecoder3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = self.conv_in(hidden_states)

        # 1. Mid
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def create_forward(*inputs):
                    return module(*inputs)

                return create_forward

            hidden_states, new_conv_cache["block_in"] = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.block_in), hidden_states, conv_cache=conv_cache.get("block_in")
            )

            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block), hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )
        else:
            hidden_states, new_conv_cache["block_in"] = self.block_in(
                hidden_states, conv_cache=conv_cache.get("block_in")
            )

            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = up_block(
                    hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                )

        hidden_states, new_conv_cache["block_out"] = self.block_out(
            hidden_states, conv_cache=conv_cache.get("block_out")
        )

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3)

        return hidden_states, new_conv_cache


class AutoencoderKLMochi(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. Used in
    [Mochi 1 preview](https://github.com/genmoai/models).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        scaling_factor (`float`, *optional*, defaults to `1.15258426`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MochiResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int] = (64, 128, 256, 384),
        decoder_block_out_channels: Tuple[int] = (128, 256, 512, 768),
        latent_channels: int = 12,
        layers_per_block: Tuple[int, ...] = (3, 3, 4, 6, 3),
        act_fn: str = "silu",
        temporal_expansions: Tuple[int, ...] = (1, 2, 3),
        spatial_expansions: Tuple[int, ...] = (2, 2, 2),
        add_attention_block: Tuple[bool, ...] = (False, True, True, True, True),
        latents_mean: Tuple[float, ...] = (
            -0.06730895953510081,
            -0.038011381506090416,
            -0.07477820912866141,
            -0.05565264470995561,
            0.012767231469026969,
            -0.04703542746246419,
            0.043896967884726704,
            -0.09346305707025976,
            -0.09918314763016893,
            -0.008729793427399178,
            -0.011931556316503654,
            -0.0321993391887285,
        ),
        latents_std: Tuple[float, ...] = (
            0.9263795028493863,
            0.9248894543193766,
            0.9393059390890617,
            0.959253732819592,
            0.8244560132752793,
            0.917259975397747,
            0.9294154431013696,
            1.3720942357788521,
            0.881393668867029,
            0.9168315692124348,
            0.9185249279345552,
            0.9274757570805041,
        ),
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        self.encoder = MochiEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=layers_per_block,
            temporal_expansions=temporal_expansions,
            spatial_expansions=spatial_expansions,
            add_attention_block=add_attention_block,
            act_fn=act_fn,
        )
        self.decoder = MochiDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            temporal_expansions=temporal_expansions,
            spatial_expansions=spatial_expansions,
            act_fn=act_fn,
        )

        self.spatial_compression_ratio = functools.reduce(lambda x, y: x * y, spatial_expansions, 1)
        self.temporal_compression_ratio = functools.reduce(lambda x, y: x * y, temporal_expansions, 1)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.num_latent_frames_batch_sizes`), the memory requirement can be lowered.
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # This can be used to determine how the number of output frames in the final decoded video. To maintain consistency with
        # the original implementation, this defaults to `True`.
        #   - Original implementation (drop_last_temporal_frames=True):
        #       Output frames = (latent_frames - 1) * temporal_compression_ratio + 1
        #   - Without dropping additional temporal upscaled frames (drop_last_temporal_frames=False):
        #       Output frames = latent_frames * temporal_compression_ratio
        # The latter case is useful for frame packing and some training/finetuning scenarios where the additional.
        self.drop_last_temporal_frames = True

        # This can be configured based on the amount of GPU memory available.
        # `12` for sample frames and `2` for latent frames are sensible defaults for consumer GPUs.
        # Setting it to higher values results in higher memory usage.
        self.num_sample_frames_batch_size = 12
        self.num_latent_frames_batch_size = 2

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (MochiEncoder3D, MochiDecoder3D)):
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

    def _enable_framewise_encoding(self):
        r"""
        Enables the framewise VAE encoding implementation with past latent padding. By default, Diffusers uses the
        oneshot encoding implementation without current latent replicate padding.

        Warning: Framewise encoding may not work as expected due to the causal attention layers. If you enable
        framewise encoding, encode a video, and try to decode it, there will be noticeable jittering effect.
        """
        self.use_framewise_encoding = True
        for name, module in self.named_modules():
            if isinstance(module, CogVideoXCausalConv3d):
                module.pad_mode = "constant"

    def _enable_framewise_decoding(self):
        r"""
        Enables the framewise VAE decoding implementation with past latent padding. By default, Diffusers uses the
        oneshot decoding implementation without current latent replicate padding.
        """
        self.use_framewise_decoding = True
        for name, module in self.named_modules():
            if isinstance(module, CogVideoXCausalConv3d):
                module.pad_mode = "constant"

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        if self.use_framewise_encoding:
            raise NotImplementedError(
                "Frame-wise encoding does not work with the Mochi VAE Encoder due to the presence of attention layers. "
                "As intermediate frames are not independent from each other, they cannot be encoded frame-wise."
            )
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
            conv_cache = None
            dec = []

            for i in range(0, num_frames, self.num_latent_frames_batch_size):
                z_intermediate = z[:, :, i : i + self.num_latent_frames_batch_size]
                z_intermediate, conv_cache = self.decoder(z_intermediate, conv_cache=conv_cache)
                dec.append(z_intermediate)

            dec = torch.cat(dec, dim=2)
        else:
            dec, _ = self.decoder(z)

        if self.drop_last_temporal_frames and dec.size(2) >= self.temporal_compression_ratio:
            dec = dec[:, :, self.temporal_compression_ratio - 1 :]

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
                    raise NotImplementedError(
                        "Frame-wise encoding does not work with the Mochi VAE Encoder due to the presence of attention layers. "
                        "As intermediate frames are not independent from each other, they cannot be encoded frame-wise."
                    )
                else:
                    time, _ = self.encoder(
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
                    conv_cache = None

                    for k in range(0, num_frames, self.num_latent_frames_batch_size):
                        tile = z[
                            :,
                            :,
                            k : k + self.num_latent_frames_batch_size,
                            i : i + tile_latent_min_height,
                            j : j + tile_latent_min_width,
                        ]
                        tile, conv_cache = self.decoder(tile, conv_cache=conv_cache)
                        time.append(tile)

                    time = torch.cat(time, dim=2)
                else:
                    time, _ = self.decoder(z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width])

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
