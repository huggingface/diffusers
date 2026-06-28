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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import get_2d_sincos_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)



def rotate_half(x):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.view(*x.shape[:-2], -1)


class JiTRotaryEmbedding(nn.Module):
    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, theta=10000, num_cls_token=0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        
        freqs_h = freqs[:, None, :].expand(-1, freqs.shape[0], -1)
        freqs_w = freqs[None, :, :].expand(freqs.shape[0], -1, -1)
        freqs = torch.cat((freqs_h, freqs_w), dim=-1)

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
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
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


class JiTAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "JiTAttention", hidden_states: torch.Tensor, rope=None) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(2, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(2, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(2, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rope is not None:
            query = rope(query)
            key = rope(key)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            dropout_p=attn.attn_drop if attn.training else 0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class JiTAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = JiTAttnProcessor
    _available_processors = [JiTAttnProcessor]

    def __init__(
        self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0, eps=1e-6, processor=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.to_out = nn.ModuleList([
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        ])

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, x, rope=None):
        return self.processor(self, x, rope=rope)


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




class JiTTransformer2DModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    A 2D Transformer for pixel-space, class-conditional image generation, as introduced in
    [JiT](https://github.com/LTH14/JiT). It operates directly on image patches (no VAE), predicts the clean image, and
    is trained with flow matching.

    Parameters:
        sample_size (`int`, defaults to `256`): Input image resolution.
        patch_size (`int`, defaults to `16`): Patch size of the bottleneck patch embedder.
        in_channels (`int`, defaults to `3`): Number of input/output image channels.
        hidden_size (`int`, defaults to `768`): Transformer hidden dimension.
        num_layers (`int`, defaults to `12`): Number of transformer blocks.
        num_attention_heads (`int`, defaults to `12`): Number of attention heads.
        mlp_ratio (`float`, defaults to `4.0`): SwiGLU feed-forward expansion ratio.
        num_classes (`int`, defaults to `1000`): Number of class labels (an extra null class is added for CFG).
        bottleneck_dim (`int`, defaults to `128`): Channel dimension of the patch-embedding bottleneck.
        in_context_len (`int`, defaults to `32`): Number of in-context conditioning tokens.
        in_context_start (`int`, defaults to `4`): Block index at which in-context tokens are injected.
        norm_eps (`float`, defaults to `1e-6`): Epsilon for the RMSNorm layers.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _no_split_modules = ["JiTBlock"]
    _repeated_blocks = ["JiTBlock"]

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

        num_patches = self.x_embedder.num_patches
        grid_size = int(num_patches**0.5)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=hidden_size,
            grid_size=grid_size,
            base_size=grid_size,
            output_type="pt",
        )
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=True)

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

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(block, x, c, rope)
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
