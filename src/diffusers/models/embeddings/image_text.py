from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...utils import deprecate
from ..activations import FP32SiLU


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with support for SD3 cropping."""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


class ImagePositionalEmbeddings(nn.Module):
    """
    Converts latent image classes into vector embeddings. Sums the vector embeddings with positional embeddings for the
    height and width of the latent space.

    For more details, see figure 10 of the dall-e paper: https://arxiv.org/abs/2102.12092

    For VQ-diffusion:

    Output vector embeddings are used as input for the transformer.

    Note that the vector embeddings for the transformer are different than the vector embeddings from the VQVAE.

    Args:
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        embed_dim (`int`):
            Dimension of the produced vector embeddings. Used for the latent pixel, height, and width embeddings.
    """

    def __init__(
        self,
        num_embed: int,
        height: int,
        width: int,
        embed_dim: int,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        self.height_emb = nn.Embedding(self.height, embed_dim)
        self.width_emb = nn.Embedding(self.width, embed_dim)

    def forward(self, index):
        emb = self.emb(index)

        height_emb = self.height_emb(torch.arange(self.height, device=index.device).view(1, self.height))

        # 1 x H x D -> 1 x H x 1 x D
        height_emb = height_emb.unsqueeze(2)

        width_emb = self.width_emb(torch.arange(self.width, device=index.device).view(1, self.width))

        # 1 x W x D -> 1 x 1 x W x D
        width_emb = width_emb.unsqueeze(1)

        pos_emb = height_emb + width_emb

        # 1 x H x W x D -> 1 x L xD
        pos_emb = pos_emb.view(1, self.height * self.width, -1)

        emb = emb + pos_emb[:, : emb.shape[1], :]

        return emb


class TextImageProjection(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 1024,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 10,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        batch_size = text_embeds.shape[0]

        # image
        image_text_embeds = self.image_embeds(image_embeds)
        image_text_embeds = image_text_embeds.reshape(batch_size, self.num_image_text_embeds, -1)

        # text
        text_embeds = self.text_proj(text_embeds)

        return torch.cat([image_text_embeds, text_embeds], dim=1)


class ImageProjection(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 32,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor):
        batch_size = image_embeds.shape[0]

        # image
        image_embeds = self.image_embeds(image_embeds)
        image_embeds = image_embeds.reshape(batch_size, self.num_image_text_embeds, -1)
        image_embeds = self.norm(image_embeds)
        return image_embeds


class IPAdapterFullImageProjection(nn.Module):
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024):
        super().__init__()
        from ..attention import FeedForward

        self.ff = FeedForward(image_embed_dim, cross_attention_dim, mult=1, activation_fn="gelu")
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor):
        return self.norm(self.ff(image_embeds))


class IPAdapterFaceIDImageProjection(nn.Module):
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024, mult=1, num_tokens=1):
        super().__init__()
        from ..attention import FeedForward

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.ff = FeedForward(image_embed_dim, cross_attention_dim * num_tokens, mult=mult, activation_fn="gelu")
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor):
        x = self.ff(image_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class IPAdapterPlusImageProjectionBlock(nn.Module):
    def __init__(
        self,
        embed_dims: int = 768,
        dim_head: int = 64,
        heads: int = 16,
        ffn_ratio: float = 4,
    ) -> None:
        super().__init__()
        from ..attention import Attention, FeedForward

        self.ln0 = nn.LayerNorm(embed_dims)
        self.ln1 = nn.LayerNorm(embed_dims)
        self.attn = Attention(
            query_dim=embed_dims,
            dim_head=dim_head,
            heads=heads,
            out_bias=False,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dims),
            FeedForward(embed_dims, embed_dims, activation_fn="gelu", mult=ffn_ratio, bias=False),
        )

    def forward(self, x, latents, residual):
        encoder_hidden_states = self.ln0(x)
        latents = self.ln1(latents)
        encoder_hidden_states = torch.cat([encoder_hidden_states, latents], dim=-2)
        latents = self.attn(latents, encoder_hidden_states) + residual
        latents = self.ff(latents) + latents
        return latents


class IPAdapterPlusImageProjection(nn.Module):
    """Resampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768. output_dims (int): The number of output channels,
        that is the same
            number of the channels in the `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int):
            The number of hidden channels. Defaults to 1280. depth (int): The number of blocks. Defaults
        to 8. dim_head (int): The number of head channels. Defaults to 64. heads (int): Parallel attention heads.
        Defaults to 16. num_queries (int):
            The number of queries. Defaults to 8. ffn_ratio (float): The expansion ratio
        of feedforward network hidden
            layer channels. Defaults to 4.
    """

    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 1024,
        hidden_dims: int = 1280,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        ffn_ratio: float = 4,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, hidden_dims) / hidden_dims**0.5)

        self.proj_in = nn.Linear(embed_dims, hidden_dims)

        self.proj_out = nn.Linear(hidden_dims, output_dims)
        self.norm_out = nn.LayerNorm(output_dims)

        self.layers = nn.ModuleList(
            [IPAdapterPlusImageProjectionBlock(hidden_dims, dim_head, heads, ffn_ratio) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input Tensor.
        Returns:
            torch.Tensor: Output Tensor.
        """
        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for block in self.layers:
            residual = latents
            latents = block(x, latents, residual)

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class IPAdapterFaceIDPlusImageProjection(nn.Module):
    """FacePerceiverResampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768. output_dims (int): The number of output channels,
        that is the same
            number of the channels in the `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int):
            The number of hidden channels. Defaults to 1280. depth (int): The number of blocks. Defaults
        to 8. dim_head (int): The number of head channels. Defaults to 64. heads (int): Parallel attention heads.
        Defaults to 16. num_tokens (int): Number of tokens num_queries (int): The number of queries. Defaults to 8.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        ffproj_ratio (float): The expansion ratio of feedforward network hidden
            layer channels (for ID embeddings). Defaults to 4.
    """

    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 768,
        hidden_dims: int = 1280,
        id_embeddings_dim: int = 512,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_tokens: int = 4,
        num_queries: int = 8,
        ffn_ratio: float = 4,
        ffproj_ratio: int = 2,
    ) -> None:
        super().__init__()
        from ..attention import FeedForward

        self.num_tokens = num_tokens
        self.embed_dim = embed_dims
        self.clip_embeds = None
        self.shortcut = False
        self.shortcut_scale = 1.0

        self.proj = FeedForward(id_embeddings_dim, embed_dims * num_tokens, activation_fn="gelu", mult=ffproj_ratio)
        self.norm = nn.LayerNorm(embed_dims)

        self.proj_in = nn.Linear(hidden_dims, embed_dims)

        self.proj_out = nn.Linear(embed_dims, output_dims)
        self.norm_out = nn.LayerNorm(output_dims)

        self.layers = nn.ModuleList(
            [IPAdapterPlusImageProjectionBlock(embed_dims, dim_head, heads, ffn_ratio) for _ in range(depth)]
        )

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            id_embeds (torch.Tensor): Input Tensor (ID embeds).
        Returns:
            torch.Tensor: Output Tensor.
        """
        id_embeds = id_embeds.to(self.clip_embeds.dtype)
        id_embeds = self.proj(id_embeds)
        id_embeds = id_embeds.reshape(-1, self.num_tokens, self.embed_dim)
        id_embeds = self.norm(id_embeds)
        latents = id_embeds

        clip_embeds = self.proj_in(self.clip_embeds)
        x = clip_embeds.reshape(-1, clip_embeds.shape[2], clip_embeds.shape[3])

        for block in self.layers:
            residual = latents
            latents = block(x, latents, residual)

        latents = self.proj_out(latents)
        out = self.norm_out(latents)
        if self.shortcut:
            out = id_embeds + self.shortcut_scale * out
        return out


class MultiIPAdapterImageProjection(nn.Module):
    def __init__(self, IPAdapterImageProjectionLayers: Union[List[nn.Module], Tuple[nn.Module]]):
        super().__init__()
        self.image_projection_layers = nn.ModuleList(IPAdapterImageProjectionLayers)

    def forward(self, image_embeds: List[torch.Tensor]):
        projected_image_embeds = []

        # currently, we accept `image_embeds` as
        #  1. a tensor (deprecated) with shape [batch_size, embed_dim] or [batch_size, sequence_length, embed_dim]
        #  2. list of `n` tensors where `n` is number of ip-adapters, each tensor can hae shape [batch_size, num_images, embed_dim] or [batch_size, num_images, sequence_length, embed_dim]
        if not isinstance(image_embeds, list):
            deprecation_message = (
                "You have passed a tensor as `image_embeds`.This is deprecated and will be removed in a future release."
                " Please make sure to update your script to pass `image_embeds` as a list of tensors to suppress this warning."
            )
            deprecate("image_embeds not a list", "1.0.0", deprecation_message, standard_warn=False)
            image_embeds = [image_embeds.unsqueeze(1)]

        if len(image_embeds) != len(self.image_projection_layers):
            raise ValueError(
                f"image_embeds must have the same length as image_projection_layers, got {len(image_embeds)} and {len(self.image_projection_layers)}"
            )

        for image_embed, image_projection_layer in zip(image_embeds, self.image_projection_layers):
            batch_size, num_images = image_embed.shape[0], image_embed.shape[1]
            image_embed = image_embed.reshape((batch_size * num_images,) + image_embed.shape[2:])
            image_embed = image_projection_layer(image_embed)
            image_embed = image_embed.reshape((batch_size, num_images) + image_embed.shape[1:])

            projected_image_embeds.append(image_embed)

        return projected_image_embeds


class TextTimeEmbedding(nn.Module):
    def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
        super().__init__()
        from .others import AttentionPooling

        self.norm1 = nn.LayerNorm(encoder_dim)
        self.pool = AttentionPooling(num_heads, encoder_dim)
        self.proj = nn.Linear(encoder_dim, time_embed_dim)
        self.norm2 = nn.LayerNorm(time_embed_dim)

    def forward(self, hidden_states):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class TextImageTimeEmbedding(nn.Module):
    def __init__(self, text_embed_dim: int = 768, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)
        self.text_norm = nn.LayerNorm(time_embed_dim)
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        # text
        time_text_embeds = self.text_proj(text_embeds)
        time_text_embeds = self.text_norm(time_text_embeds)

        # image
        time_image_embeds = self.image_proj(image_embeds)

        return time_image_embeds + time_text_embeds


class ImageTimeEmbedding(nn.Module):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = nn.LayerNorm(time_embed_dim)

    def forward(self, image_embeds: torch.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        return time_image_embeds


class ImageHintTimeEmbedding(nn.Module):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = nn.LayerNorm(time_embed_dim)
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, 4, 3, padding=1),
        )

    def forward(self, image_embeds: torch.Tensor, hint: torch.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        hint = self.input_hint_block(hint)
        return time_image_embeds, hint
