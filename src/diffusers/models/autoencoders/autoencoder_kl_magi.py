# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)


@dataclass
class AutoencoderKLMagiOutput(BaseOutput):
    """
    Output of AutoencoderKLMagi encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling from the encoded latent vector.
    """

    latent_dist: "DiagonalGaussianDistribution"


class DiagonalGaussianDistribution(object):
    """
    Diagonal Gaussian distribution with mean and logvar.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, generator=None):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device, generator=generator)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3, 4],
                )

    def nll(self, sample, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = torch.log(torch.tensor(2.0 * 3.141592653589793))
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class ManualLayerNorm(nn.Module):
    """
    Manual implementation of LayerNorm for better compatibility.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + self.eps)
        return x_normalized


class Mlp(nn.Module):
    """
    MLP module used in the transformer architecture.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head attention module.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with attention and MLP.
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for 3D data.
    """
    def __init__(self, video_size=224, video_length=16, patch_size=16, patch_length=1, in_chans=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.video_length = video_length
        self.patch_size = patch_size
        self.patch_length = patch_length

        self.grid_size = video_size // patch_size
        self.grid_length = video_length // patch_length
        self.num_patches = self.grid_length * self.grid_size * self.grid_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(patch_length, patch_size, patch_size),
            stride=(patch_length, patch_size, patch_size)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert H == self.video_size and W == self.video_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.video_size}*{self.video_size})."
        assert T == self.video_length, \
            f"Input video length ({T}) doesn't match model ({self.video_length})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder for MAGI-1 VAE.
    """
    def __init__(
        self,
        video_size=256,
        video_length=16,
        patch_size=8,
        patch_length=4,
        in_chans=3,
        z_chans=4,
        double_z=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        with_cls_token=True,
    ):
        super().__init__()
        self.video_size = video_size
        self.video_length = video_length
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.z_chans = z_chans
        self.double_z = double_z
        self.with_cls_token = with_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbed(
            video_size=video_size,
            video_length=video_length,
            patch_size=patch_size,
            patch_length=patch_length,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = 1
        else:
            self.num_tokens = 0

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, 0.0, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Projection to latent space
        self.proj = nn.Linear(embed_dim, z_chans * 2 if double_z else z_chans)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize position embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize cls token if used
        if self.with_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Add class token if used
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Use class token for output if available, otherwise use patch tokens
        if self.with_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        # Project to latent space
        x = self.proj(x)

        # Reshape to [B, C, T, H, W]
        B = x.shape[0]
        T = self.video_length // self.patch_length
        H = self.video_size // self.patch_size
        W = self.video_size // self.patch_size
        C = self.z_chans * 2 if self.double_z else self.z_chans

        x = x.view(B, C, T, H, W)

        return x


class ViTDecoder(nn.Module):
    """
    Vision Transformer Decoder for MAGI-1 VAE.
    """
    def __init__(
        self,
        video_size=256,
        video_length=16,
        patch_size=8,
        patch_length=4,
        in_chans=3,
        z_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        with_cls_token=True,
    ):
        super().__init__()
        self.video_size = video_size
        self.video_length = video_length
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.z_chans = z_chans
        self.with_cls_token = with_cls_token

        # Calculate patch dimensions
        self.grid_size = video_size // patch_size
        self.grid_length = video_length // patch_length
        num_patches = self.grid_length * self.grid_size * self.grid_size

        # Input projection from latent space to embedding dimension
        self.proj_in = nn.Linear(z_chans, embed_dim)

        # Class token and position embedding
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = 1
        else:
            self.num_tokens = 0

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, 0.0, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Output projection to image space
        self.proj_out = nn.ConvTranspose3d(
            embed_dim,
            in_chans,
            kernel_size=(patch_length, patch_size, patch_size),
            stride=(patch_length, patch_size, patch_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize cls token if used
        if self.with_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

        # Initialize output projection
        w = self.proj_out.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, z):
        # Get dimensions
        B, C, T, H, W = z.shape

        # Flatten spatial dimensions and transpose to [B, T*H*W, C]
        z = z.flatten(2).transpose(1, 2)

        # Project to embedding dimension
        x = self.proj_in(z)

        # Add class token if used
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Remove class token if used
        if self.with_cls_token:
            x = x[:, 1:]

        # Reshape to [B, T, H, W, C]
        x = x.reshape(B, T, H, W, -1)

        # Transpose to [B, C, T, H, W]
        x = x.permute(0, 4, 1, 2, 3)

        # Project to image space
        x = self.proj_out(x)

        return x


class AutoencoderKLMagi(ModelMixin, ConfigMixin):
    """
    Variational Autoencoder (VAE) model with KL loss for MAGI-1.

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic methods
    implemented for all models (downloading, saving, loading, etc.)

    Parameters:
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock3D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock3D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to 1): Number of layers per block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 8): Number of channels in the latent space.
        norm_num_groups (`int`, *optional*, defaults to 32): Number of groups for the normalization.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula:
            `z = 1 / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution
            Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        temporal_downsample_factor (`Tuple[int]`, *optional*, defaults to (1, 2, 1, 1)):
            Tuple of temporal downsampling factors for each block.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 8,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.18215,
        temporal_downsample_factor: Tuple[int] = (1, 2, 1, 1),
        video_size: int = 256,
        video_length: int = 16,
        patch_size: int = 8,
        patch_length: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        with_cls_token: bool = True,
    ):
        super().__init__()

        # Save important parameters
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.temperal_downsample = temporal_downsample_factor

        # Create encoder and decoder
        self.encoder = ViTEncoder(
            video_size=video_size,
            video_length=video_length,
            patch_size=patch_size,
            patch_length=patch_length,
            in_chans=in_channels,
            z_chans=latent_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=nn.LayerNorm,
            with_cls_token=with_cls_token,
            double_z=True,
        )

        self.decoder = ViTDecoder(
            video_size=video_size,
            video_length=video_length,
            patch_size=patch_size,
            patch_length=patch_length,
            in_chans=out_channels,
            z_chans=latent_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=nn.LayerNorm,
            with_cls_token=with_cls_token,
        )

        # Enable tiling
        self._enable_tiling = False
        self._tile_sample_min_size = None
        self._tile_sample_stride = None

    @property
    def spatial_downsample_factor(self) -> int:
        """
        Returns the spatial downsample factor for the VAE.
        """
        return self.encoder.patch_size  # MAGI-1 uses patch_size as spatial downsampling

    @property
    def temporal_downsample_factor(self) -> int:
        """
        Returns the temporal downsample factor for the VAE.
        """
        return self.encoder.patch_length  # MAGI-1 uses patch_length as temporal downsampling

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLMagiOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of videos.

        Args:
            x (`torch.FloatTensor`): Input batch of videos.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a dictionary or tuple.

        Returns:
            `AutoencoderKLMagiOutput` or `tuple`:
                If return_dict is True, returns an `AutoencoderKLMagiOutput` object, otherwise returns a tuple.
        """
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLMagiOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[torch.FloatTensor, BaseOutput]:
        """
        Decode a batch of latent vectors.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a dictionary or tuple.

        Returns:
            `BaseOutput` or `torch.FloatTensor`:
                If return_dict is True, returns a `BaseOutput` object, otherwise returns the decoded tensor.
        """
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return BaseOutput(sample=dec)

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
    ) -> None:
        """
        Enable tiled processing for large videos.

        Args:
            tile_sample_min_height (`int`, *optional*): Minimum tile height.
            tile_sample_min_width (`int`, *optional*): Minimum tile width.
        """
        self._enable_tiling = True
        self._tile_sample_min_size = (tile_sample_min_height, tile_sample_min_width)

    def disable_tiling(self) -> None:
        """
        Disable tiled processing.
        """
        self._enable_tiling = False
        self._tile_sample_min_size = None
        self._tile_sample_stride = None

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[BaseOutput, Tuple]:
        """
        Forward pass of the model.

        Args:
            sample (`torch.FloatTensor`): Input batch of videos.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior distribution.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a dictionary or tuple.
            generator (`torch.Generator`, *optional*): Generator for random sampling.

        Returns:
            `BaseOutput` or `tuple`:
                If return_dict is True, returns a `BaseOutput` object, otherwise returns a tuple.
        """
        posterior = self.encode(sample, return_dict=True).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # Scale latents by the scaling factor
        z = self.scaling_factor * z

        # Decode the latents
        dec = self.decode(z, return_dict=return_dict)

        if not return_dict:
            return (dec,)

        return BaseOutput(sample=dec.sample)