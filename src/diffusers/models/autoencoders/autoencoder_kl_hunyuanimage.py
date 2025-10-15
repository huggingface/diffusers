# Copyright 2025 Tencent Hunyuan Team and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_outputs import AutoencoderKLOutput
from ...models.modeling_utils import ModelMixin
from ...utils import BaseOutput
from ...utils.torch_utils import randn_tensor


@dataclass
class DecoderOutput(BaseOutput):
    """Output of the decoder with sample and optional posterior distribution."""

    sample: torch.FloatTensor
    posterior: Optional["DiagonalGaussianDistribution"] = None


class DiagonalGaussianDistribution:
    """
    Gaussian Distribution with diagonal covariance matrix.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H, W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            zero_tensor = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)
            self.var = zero_tensor
            self.std = zero_tensor

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device, dtype=self.parameters.dtype)
        reduce_dim = list(range(1, self.mean.ndim))
        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=reduce_dim,
            )
        else:
            return 0.5 * torch.sum(
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=reduce_dim,
            )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = (1, 2, 3)) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device, dtype=self.parameters.dtype)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    """
    Forward pass with optional gradient checkpointing for memory efficiency.
    """

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


class AttnBlock(nn.Module):
    """Self-attention block for 3D tensors."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b (h w) c").contiguous()

        x = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    Residual block with two convolutions and optional channel change.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """
    Downsampling block for spatial reduction.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        factor = 4
        assert out_channels % factor == 0

        self.conv = nn.Conv2d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = rearrange(h, "b c (h r1) (w r2) -> b (r1 r2 c) h w", r1=2, r2=2)
        shortcut = rearrange(x, "b c (h r1) (w r2) -> b (r1 r2 c) h w", r1=2, r2=2)

        B, C, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, H, W).mean(dim=2)
        return h + shortcut


class Upsample(nn.Module):
    """
    Upsampling block for spatial expansion.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        factor = 4
        self.conv = nn.Conv2d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = rearrange(h, "b (r1 r2 c) h w -> b c (h r1) (w r2)", r1=2, r2=2)
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 c) h w -> b c (h r1) (w r2)", r1=2, r2=2)
        return h + shortcut


class Encoder(nn.Module):
    """
    Encoder network that compresses input to latent representation.
    """

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]

        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))

            if add_spatial_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = Downsample(block_in, block_out)
                block_in = block_out

            self.down.append(down)

        # Middle blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # Downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
            if hasattr(self.down[i_level], "downsample"):
                h = forward_with_checkpointing(self.down[i_level].downsample, h, use_checkpointing=use_checkpointing)

        # Middle processing
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # Output with shortcut connection
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) h w -> b c r h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    """
    Decoder network that reconstructs output from latent representation.
    """

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        block_in = block_out_channels[0]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch

            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            up = nn.Module()
            up.block = block

            # Determine upsampling strategy
            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))

            if add_spatial_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = Upsample(block_in, block_out)
                block_in = block_out

            self.up.append(up)

        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        repeats = self.block_out_channels[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoencoderKLHunyuanImage(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.
    Adapted from HunyuanImage 2.1's custom VAE with 32x spatial compression.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int, *optional*, defaults to 3): Number of channels in the output.
        latent_channels (int, *optional*, defaults to 64): Number of channels in the latent space.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(512, 1024, 2048, 4096)`):
            Tuple of block output channels.
        layers_per_block (int, *optional*, defaults to 2): Number of layers per block.
        ffactor_spatial (int, *optional*, defaults to 32): Spatial downsampling/upsampling factor.
        sample_size (int, *optional*, defaults to 512): Sample size of the model.
        sample_tsize (int, *optional*, defaults to 1): Temporal sample size.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model.
        shift_factor (`float`, *optional*): Shift factor for the latent space.
        downsample_match_channel (bool, *optional*, defaults to True): Whether to match channels during downsampling.
        upsample_match_channel (bool, *optional*, defaults to True): Whether to match channels during upsampling.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 64,
        block_out_channels: Tuple[int] = (512, 1024, 2048, 4096),
        layers_per_block: int = 2,
        ffactor_spatial: int = 32,
        sample_size: int = 512,
        sample_tsize: int = 1,
        scaling_factor: float = 1.0,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            downsample_match_channel=downsample_match_channel,
        )

        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            upsample_match_channel=upsample_match_channel,
        )

        # Tiling and slicing configuration
        self.use_slicing = False
        self.use_spatial_tiling = False

        # Tiling parameters
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Enable or disable gradient checkpointing for memory efficiency.
        """
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_slicing(self):
        """Enable slicing for batch processing."""
        self.use_slicing = True

    def disable_slicing(self):
        """Disable slicing for batch processing."""
        self.use_slicing = False

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """
        Encode input tensor to latent representation.

        Args:
            x (`torch.Tensor`): Input tensor.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a dict.
        """
        original_ndim = x.ndim
        if original_ndim == 5:
            x = x.squeeze(2)

        # Process with or without slicing
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        if original_ndim == 5:
            h = h.unsqueeze(2)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None):
        """
        Decode latent representation to output tensor.

        Args:
            z (`torch.Tensor`): Latent tensor.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a dict.
            generator: unused, for compatibility.
        """
        original_ndim = z.ndim
        if original_ndim == 5:
            z = z.squeeze(2)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self.decoder(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self.decoder(z)

        if original_ndim == 5:
            decoded = decoded.unsqueeze(2)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_posterior: bool = True,
        return_dict: bool = True,
    ):
        """
        Forward pass through the VAE (Encode and Decode).

        Args:
            sample (`torch.Tensor`): Input tensor.
            sample_posterior (`bool`, *optional*, defaults to `False`): Whether to sample from the posterior.
            return_posterior (`bool`, *optional*, defaults to `True`): Whether to return the posterior.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a dict.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample

        if return_dict:
            return DecoderOutput(sample=dec, posterior=posterior)
        else:
            return (dec, posterior)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict, handling possible 5D weight tensors.
        """
        converted_state_dict = {}

        for key, value in state_dict.items():
            if "weight" in key:
                if len(value.shape) == 5 and value.shape[2] == 1:
                    converted_state_dict[key] = value.squeeze(2)
                else:
                    converted_state_dict[key] = value
            else:
                converted_state_dict[key] = value

        return super().load_state_dict(converted_state_dict, strict=strict)
