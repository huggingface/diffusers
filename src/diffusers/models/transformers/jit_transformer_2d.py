# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    if len(shape_lens) != 1:
        raise ValueError("tensors must all have the same number of dimensions")
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*(list(t.shape) for t in tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]

    if not all(len(set(t[1])) <= 2 for t in expandable_dims):
        raise ValueError("invalid dimensions for broadcastable concatenation")

    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.view(*x.shape[:-2], -1)


class JiTRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            _, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype)

            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0), persistent=False)
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0), persistent=False)
        else:
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]), persistent=False)
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]), persistent=False)

    def forward(self, t):
        seq_len = t.shape[1]
        freqs_cos = self.freqs_cos[:seq_len].to(t.dtype)
        freqs_sin = self.freqs_sin[:seq_len].to(t.dtype)

        return t * freqs_cos[:, None, :] + rotate_half(t) * freqs_sin[:, None, :]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class JiTPatchEmbed(nn.Module):
    """Image to Patch Embedding with Bottleneck"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class JiTTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class JiTLabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class JiTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = rope(q)
            k = rope(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class JiTSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.attn = JiTAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = JiTSwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.act = nn.SiLU()
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def forward(self, x, c, feat_rope=None):
        # Apply activation
        c = self.act(c)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention block
        norm_x = self.norm1(x)
        modulated_x = modulate(norm_x, shift_msa, scale_msa)
        attn_out = self.attn(modulated_x, rope=feat_rope)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block
        norm_x = self.norm2(x)
        modulated_x = modulate(norm_x, shift_mlp, scale_mlp)
        mlp_out = self.mlp(modulated_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, but got {embed_dim}")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, but got {embed_dim}")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class JiTTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        in_context_len: int = 32,
        in_context_start: int = 4,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.norm_eps = norm_eps
        self.gradient_checkpointing = False

        # Time and Class Embedding
        self.t_embedder = JiTTimestepEmbedder(hidden_size)
        self.y_embedder = JiTLabelEmbedder(num_classes, hidden_size)

        # Patch Embedding
        self.x_embedder = JiTPatchEmbed(
            img_size=sample_size,
            patch_size=patch_size,
            in_chans=in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
            bias=True,
        )

        # Positional Embedding (Fixed Sin-Cos)
        num_patches = self.x_embedder.num_patches
        pos_embed = get_2d_sincos_pos_embed(hidden_size, int(num_patches**0.5))
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=True)

        # In-context Embedding
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size))

        # RoPE
        half_head_dim = hidden_size // num_attention_heads // 2
        hw_seq_len = sample_size // patch_size
        self.feat_rope = JiTRotaryEmbedding(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = JiTRotaryEmbedding(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        # Blocks
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size,
                    num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attention_dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    proj_drop=dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    eps=norm_eps,
                )
                for i in range(num_layers)
            ]
        )

        # Final Layer
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps)
        self.linear_final = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        self.act_final = nn.SiLU()
        self.adaLN_modulation_final = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        class_labels: torch.LongTensor,
        return_dict: bool = True,
    ):

        t_emb = self.t_embedder(timestep, dtype=hidden_states.dtype)
        y_emb = self.y_embedder(class_labels)

        # Ensure embeddings match hidden_states dtype
        y_emb = y_emb.to(dtype=hidden_states.dtype)

        c = t_emb + y_emb

        # Patch Embed
        x = self.x_embedder(hidden_states)
        x = x + self.pos_embed.to(x.dtype)

        # Blocks
        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb.to(in_context_tokens.dtype)
                x = torch.cat([in_context_tokens, x], dim=1)

            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    c,
                    rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, feat_rope=rope)

        # Slice off in-context tokens
        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]

        # Final Layer
        c = self.act_final(c)
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear_final(x)

        # Unpatchify
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, self.patch_size, self.patch_size, self.out_channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        output = x.reshape(shape=(x.shape[0], self.out_channels, h * self.patch_size, w * self.patch_size))

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
