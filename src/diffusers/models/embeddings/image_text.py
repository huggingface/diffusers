import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import deprecate
from ..activations import FP32SiLU


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


class AttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x):
        bs, length, width = x.size()

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        x = torch.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = torch.einsum("bts,bcs->bct", weight, v)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)

        return a[:, 0, :]  # cls_token


class TextTimeEmbedding(nn.Module):
    def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
        super().__init__()
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

class HunyuanDiTAttentionPool(nn.Module):
    # Copied from https://github.com/Tencent/HunyuanDiT/blob/cb709308d92e6c7e8d59d0dff41b74d35088db6a/hydit/modules/poolers.py#L6

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class HunyuanCombinedTimestepTextSizeStyleEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        pooled_projection_dim=1024,
        seq_len=256,
        cross_attention_dim=2048,
        use_style_cond_and_image_meta_size=True,
    ):
        super().__init__()
        from .image_text import PixArtAlphaTextProjection
        from .timestep import TimestepEmbedding, Timesteps

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.size_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)

        self.pooler = HunyuanDiTAttentionPool(
            seq_len, cross_attention_dim, num_heads=8, output_dim=pooled_projection_dim
        )

        # Here we use a default learned embedder layer for future extension.
        self.use_style_cond_and_image_meta_size = use_style_cond_and_image_meta_size
        if use_style_cond_and_image_meta_size:
            self.style_embedder = nn.Embedding(1, embedding_dim)
            extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim
        else:
            extra_in_dim = pooled_projection_dim

        self.extra_embedder = PixArtAlphaTextProjection(
            in_features=extra_in_dim,
            hidden_size=embedding_dim * 4,
            out_features=embedding_dim,
            act_fn="silu_fp32",
        )

    def forward(self, timestep, encoder_hidden_states, image_meta_size, style, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, 256)

        # extra condition1: text
        pooled_projections = self.pooler(encoder_hidden_states)  # (N, 1024)

        if self.use_style_cond_and_image_meta_size:
            # extra condition2: image meta size embdding
            image_meta_size = self.size_proj(image_meta_size.view(-1))
            image_meta_size = image_meta_size.to(dtype=hidden_dtype)
            image_meta_size = image_meta_size.view(-1, 6 * 256)  # (N, 1536)

            # extra condition3: style embedding
            style_embedding = self.style_embedder(style)  # (N, embedding_dim)

            # Concatenate all extra vectors
            extra_cond = torch.cat([pooled_projections, image_meta_size, style_embedding], dim=1)
        else:
            extra_cond = torch.cat([pooled_projections], dim=1)

        conditioning = timesteps_emb + self.extra_embedder(extra_cond)  # [B, D]

        return conditioning
