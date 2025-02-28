# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import FeedForward
from ..attention_processor import Attention
from ..embeddings import TimestepEmbedding, Timesteps, PixArtAlphaTextProjection
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # i2v task
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states


        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if grid_sizes is not None and freqs is not None:
            query = rope_apply(query, grid_sizes, freqs)
            key = rope_apply(key, grid_sizes, freqs)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # i2v task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            
            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.cuda.amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.cuda.amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = nn.LayerNorm(out_features, elementwise_affine=False)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embedding_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        
        self.image_embedder = None
        if image_embedding_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embedding_dim, dim)

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_hidden_states_image: Optional[torch.Tensor] = None):
        timestep = self.timesteps_proj(timestep)
        with torch.amp.autocast(dtype=torch.float32):
            temb = self.time_embedder(timestep)
            timestep_proj = self.time_proj(self.act_fn(temb))
        assert temb.dtype == torch.float32 and timestep_proj.dtype == torch.float32
        
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
        
        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTransformerBlock(nn.Module):
    def __init__(self,
        dim,
        ffn_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        added_kv_proj_dim=None
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        grid_sizes,
        freqs,
    ):
        assert temb.dtype == torch.float32
        with torch.cuda.amp.autocast(dtype=torch.float32):
            temb = (self.scale_shift_table + temb).chunk(6, dim=1)
        assert temb[0].dtype == torch.float32

        # self-attention
        attn_hidden_states = self.norm1(hidden_states).float() * (1 + temb[1]) + temb[0]

        attn_hidden_states = self.attn1(
            hidden_states=attn_hidden_states,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states + attn_hidden_states * temb[2]

        # cross-attention
        attn_hidden_states = self.norm3(hidden_states)
        attn_hidden_states = self.attn2(
            hidden_states=attn_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            grid_sizes=None,
            freqs=None,
        )
        hidden_states = hidden_states + attn_hidden_states

        # ffn
        ffn_hidden_states = self.norm2(hidden_states).float() * (1 + temb[4]) + temb[3]
        ffn_hidden_states = self.ffn(ffn_hidden_states)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states + ffn_hidden_states * temb[5]
        
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "text_embedding", "time_embedding", "time_projection", "norm"]
    _no_split_modules = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_embedding_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.out_channels = out_channels
        self.patch_size = patch_size

        # 1. Patch embedding
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embedding_dim=image_embedding_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
            for _ in range(num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert attention_head_dim % 2 == 0
        self.freqs = torch.cat([
            rope_params(1024, attention_head_dim - 4 * (attention_head_dim // 6)),
            rope_params(1024, 2 * (attention_head_dim // 6)),
            rope_params(1024, 2 * (attention_head_dim // 6))
        ], dim=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.freqs.device != hidden_states.device:
            self.freqs = self.freqs.to(hidden_states.device)
        
        hidden_states = self.patch_embedding(hidden_states)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[1:], dtype=torch.long) for u in hidden_states]
        )
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # hidden_states = hidden_states.flatten(2).transpose(1, 2) # (b, l, c)
        # seq_lens = torch.tensor([u.size(0) for u in hidden_states], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        # hidden_states = torch.cat([
        #     torch.cat([u.unsqueeze(0), u.new_zeros(1, seq_len - u.size(0), u.size(1))],
        #             dim=1) for u in hidden_states
        # ])

        # with torch.cuda.amp.autocast(dtype=torch.float32):
        #     e = self.time_embedding(
        #         sinusoidal_embedding_1d(self.freq_dim, timestep).float())
        #     e0 = self.time_projection(e).unflatten(1, (6, -1))
        #     assert e.dtype == torch.float32 and e0.dtype == torch.float32
        
        # context_lens = None
        # encoder_hidden_states = self.text_embedding(encoder_hidden_states)
        # if self.add_img_emb:
        #     img_encoder_hidden_states = kwargs.get('img_encoder_hidden_states', None)
        #     if img_encoder_hidden_states is None:
        #         raise ValueError('`add_img_emb` is set but `img_encoder_hidden_states` is not provided.')
        #     img_encoder_hidden_states = self.img_emb(img_encoder_hidden_states)
        #     encoder_hidden_states = torch.concat([img_encoder_hidden_states, encoder_hidden_states], dim=1)
        
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)
        
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    timestep_proj,
                    encoder_hidden_states,
                    grid_sizes,
                    self.freqs,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    timestep_proj,
                    encoder_hidden_states,
                    grid_sizes,
                    self.freqs,
                )

        # Output projection
        with torch.amp.autocast(dtype=torch.float32):
            temb = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + temb[1]) + temb[0]
            hidden_states = self.proj_out(hidden_states)
        
        hidden_states = hidden_states.type_as(encoder_hidden_states)

        # 5. Unpatchify
        hidden_states = self.unpatchify(hidden_states, grid_sizes)
        hidden_states = torch.stack(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)


    def unpatchify(self, hidden_states, grid_sizes):
        c = self.out_channels
        out = []
        for u, v in zip(hidden_states, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

