from typing import Optional

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.models.embeddings import DalleMaskImageEmbedding

from .attention import CrossAttention


class VQDiffusionTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        n_heads: int,
        d_head: int,
        depth: int,
        context_dim: int,
        num_embed: int,
        height: int,
        width: int,
        diffusion_steps: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner_dim = n_heads * d_head

        self.latent_image_embedding = DalleMaskImageEmbedding(
            num_embed=num_embed, embed_dim=inner_dim, height=height, width=width
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    diffusion_steps=diffusion_steps,
                    block_idx=block_idx,
                )
                for block_idx in range(depth)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim)
        self.out = nn.Linear(inner_dim, num_embed)

    def forward(self, latent_images, cond_emb, t):
        embedded_latent_images = self.latent_image_embedding(latent_images)
        hidden_states = embedded_latent_images

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, cond_emb, t)

        logits = self.out(self.norm_out(hidden_states))
        out = logits.permute(0, 2, 1)

        return out


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        diffusion_steps: int,
        block_idx,
        dropout=0.0,
    ):
        super().__init__()

        self.block_idx = block_idx

        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=True
        )  # is a self-attention
        self.ff = FeedForward(dim=dim, dropout=dropout)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=True
        )
        self.norm1 = AdaLayerNorm(dim, diffusion_steps)
        self.norm2 = AdaLayerNorm(dim, diffusion_steps)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, hidden_states, context, timestep):
        hidden_states = self.attn1(self.norm1(hidden_states, timestep)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states, timestep), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.Sequential(nn.Linear(dim, inner_dim), GELU2(), nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, hidden_states):
        return self.net(hidden_states)


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
