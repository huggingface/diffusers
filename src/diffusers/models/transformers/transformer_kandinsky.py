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
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm

logger = logging.get_logger(__name__)


if torch.cuda.get_device_capability()[0] >= 9:
    try:
        from flash_attn_interface import flash_attn_func as FA
    except:
        FA = None
        
    try:
        from flash_attn import flash_attn_func as FA
    except:
        FA = None
else:
    try:
        from flash_attn import flash_attn_func as FA
    except:
        FA = None


# @torch.compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x, scale, shift):
    return (norm(x) * (scale + 1.0) + shift).to(torch.bfloat16)

# @torch.compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x, out, gate):
    return (x + gate * out).to(torch.bfloat16)

# @torch.compile()
@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x, rope):
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(torch.bfloat16)


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=dim, dtype=torch.float32)
        / dim
    )
    return freqs


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim, time_dim, max_period=10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer(
            "freqs", get_freqs(model_dim // 2, max_period), persistent=False
        )
        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    def forward(self, time):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        return time_embed


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)
    
    def forward(self, x):
        batch_size, duration, height, width, dim = x.shape
        x = (
            x.view(
                batch_size,
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1], 
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


class RoPE1D(nn.Module):
    """
    1D Rotary Positional Embeddings for text sequences.
    
    Args:
        dim: Dimension of the rotary embeddings
        max_pos: Maximum sequence length
        max_period: Maximum period for sinusoidal embeddings
    """
    
    def __init__(self, dim, max_pos=1024, max_period=10000.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos):
        """
        Args:
            pos: Position indices of shape [seq_len] or [batch_size, seq_len]
            
        Returns:
            Rotary embeddings of shape [seq_len, 1, 2, 2]
        """
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class RoPE3D(nn.Module):
    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, shape, pos, scale_factor=(1.0, 1.0, 1.0)):
        batch_size, duration, height, width = shape
        
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args_t_expanded = args_t.view(1, duration, 1, 1, -1).expand(batch_size, -1, height, width, -1)
        args_h_expanded = args_h.view(1, 1, height, 1, -1).expand(batch_size, duration, -1, width, -1)
        args_w_expanded = args_w.view(1, 1, 1, width, -1).expand(batch_size, duration, height, -1, -1)

        args = torch.cat([args_t_expanded, args_h_expanded, args_w_expanded], dim=-1)
        
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    def forward(self, x):
        return self.out_layer(self.activation(x))


class MultiheadSelfAttentionEnc(nn.Module):
    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def forward(self, x, rope):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        query = self.query_norm(query.float()).type_as(query)
        key = self.key_norm(key.float()).type_as(key)

        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        # Use torch's scaled_dot_product_attention
        # print(query.shape, key.shape, value.shape, "QKV MultiheadSelfAttentionEnc SHAPE")
        # out = F.scaled_dot_product_attention(
        #     query.permute(0, 2, 1, 3),
        #     key.permute(0, 2, 1, 3),
        #     value.permute(0, 2, 1, 3),
        # ).permute(0, 2, 1, 3).flatten(-2, -1)
        
        out = FA(q=query, k=key, v=value).flatten(-2, -1)

        out = self.out_layer(out)
        return out


class MultiheadSelfAttentionDec(nn.Module):
    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def forward(self, x, rope, sparse_params=None):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        query = self.query_norm(query.float()).type_as(query)
        key = self.key_norm(key.float()).type_as(key)

        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        # Use standard attention (can be extended with sparse attention)
        # out = F.scaled_dot_product_attention(
        #     query.permute(0, 2, 1, 3),
        #     key.permute(0, 2, 1, 3),
        #     value.permute(0, 2, 1, 3),
        # ).permute(0, 2, 1, 3).flatten(-2, -1)
        
        # print(query.shape, key.shape, value.shape, "QKV MultiheadSelfAttentionDec SHAPE")
        
        out = FA(q=query, k=key, v=value).flatten(-2, -1)

        out = self.out_layer(out)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def forward(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)
        
        shape, cond_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)
        
        query = self.query_norm(query.float()).type_as(query)
        key = self.key_norm(key.float()).type_as(key)
        
        # out = F.scaled_dot_product_attention(
        #     query.permute(0, 2, 1, 3),
        #     key.permute(0, 2, 1, 3),
        #     value.permute(0, 2, 1, 3),
        # ).permute(0, 2, 1, 3).flatten(-2, -1)
        
        # print(query.shape, key.shape, value.shape, "QKV MultiheadCrossAttention SHAPE")

        out = FA(q=query, k=key, v=value).flatten(-2, -1)

        out = self.out_layer(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionEnc(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope):
        self_attn_params, ff_params = torch.chunk(
            self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift)
        out = self.self_attention(out, rope)
        x = apply_gate_sum(x, out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionDec(model_dim, head_dim)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, visual_embed, text_embed, time_embed, rope, sparse_params):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift)
        visual_out = self.self_attention(visual_out, rope, sparse_params)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift)
        visual_out = self.cross_attention(visual_out, text_embed)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift)
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        return visual_embed


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

    def forward(self, visual_embed, text_embed, time_embed):
        # Handle the new batch dimension: [batch, duration, height, width, model_dim]
        batch_size, duration, height, width, _ = visual_embed.shape
        
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        
        # Apply modulation with proper broadcasting for the new shape
        visual_embed = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale[:, None, None, None],  # [batch, 1, 1, 1, model_dim] -> [batch, 1, 1, 1]
            shift[:, None, None, None],  # [batch, 1, 1, 1, model_dim] -> [batch, 1, 1, 1]
        ).type_as(visual_embed)
        
        x = self.out_layer(visual_embed)

        # Now x has shape [batch, duration, height, width, patch_prod * visual_dim]
        x = (
            x.view(
                batch_size,
                duration,
                height, 
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1], 
                self.patch_size[2],
            )
            .permute(0, 5, 1, 6, 2, 7, 3, 4)  # [batch, patch_t, duration, patch_h, height, patch_w, width, features]
            .flatten(1, 2)  # [batch, patch_t * duration, height, patch_w, width, features]
            .flatten(2, 3)  # [batch, patch_t * duration, patch_h * height, width, features]  
            .flatten(3, 4)  # [batch, patch_t * duration, patch_h * height, patch_w * width]
        )
        return x
                    
                                 
@maybe_allow_in_graph
class Kandinsky5Transformer3DModel(ModelMixin, ConfigMixin):
    r"""
    A 3D Transformer model for video generation used in Kandinsky 5.0.

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods implemented for all models (such as downloading or saving).

    Args:
        in_visual_dim (`int`, defaults to 16):
            Number of channels in the input visual latent.
        out_visual_dim (`int`, defaults to 16):
            Number of channels in the output visual latent.
        time_dim (`int`, defaults to 512):
            Dimension of the time embeddings.
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            Patch size for the visual embeddings (temporal, height, width).
        model_dim (`int`, defaults to 1792):
            Hidden dimension of the transformer model.
        ff_dim (`int`, defaults to 7168):
            Intermediate dimension of the feed-forward networks.
        num_text_blocks (`int`, defaults to 2):
            Number of transformer blocks in the text encoder.
        num_visual_blocks (`int`, defaults to 32):
            Number of transformer blocks in the visual decoder.
        axes_dims (`Tuple[int]`, defaults to `(16, 24, 24)`):
            Dimensions for the rotary positional embeddings (temporal, height, width).
        visual_cond (`bool`, defaults to `True`):
            Whether to use visual conditioning (for image/video conditioning).
        in_text_dim (`int`, defaults to 3584):
            Dimension of the text embeddings from Qwen2.5-VL.
        in_text_dim2 (`int`, defaults to 768):
            Dimension of the pooled text embeddings from CLIP.
    """

    @register_to_config
    def __init__(
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        time_dim: int = 512,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        model_dim: int = 1792,
        ff_dim: int = 7168,
        num_text_blocks: int = 2,
        num_visual_blocks: int = 32,
        axes_dims: Tuple[int, int, int] = (16, 24, 24),
        visual_cond: bool = True,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
    ):
        super().__init__()

        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond

        # Calculate head dimension for attention
        head_dim = sum(axes_dims)

        # Determine visual embedding dimension based on conditioning
        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim

        # 1. Embedding layers
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        # 2. Rotary positional embeddings
        self.text_rope_embeddings = RoPE1D(head_dim)
        self.visual_rope_embeddings = RoPE3D(axes_dims)

        # 3. Transformer blocks
        self.text_transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim)
            for _ in range(num_text_blocks)
        ])

        self.visual_transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
            for _ in range(num_visual_blocks)
        ])

        # 4. Output layer
        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_text_embed: torch.Tensor,
        timestep: torch.Tensor,
        visual_rope_pos: List[torch.Tensor],
        text_rope_pos: torch.Tensor,
        scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass of the Kandinsky 5.0 3D Transformer.

        Args:
            hidden_states (`torch.Tensor`):
                Input visual latent tensor of shape `(batch_size, num_frames, height, width, channels)`.
            encoder_hidden_states (`torch.Tensor`):
                Text embeddings from Qwen2.5-VL of shape `(batch_size, sequence_length, text_dim)`.
            pooled_text_embed (`torch.Tensor`):
                Pooled text embeddings from CLIP of shape `(batch_size, pooled_text_dim)`.
            timestep (`torch.Tensor`):
                Timestep tensor of shape `(batch_size,)` or `(batch_size * num_frames,)`.
            visual_rope_pos (`List[torch.Tensor]`):
                List of tensors for visual rotary positional embeddings [temporal, height, width].
            text_rope_pos (`torch.Tensor`):
                Tensor for text rotary positional embeddings.
            scale_factor (`Tuple[float, float, float]`, defaults to `(1.0, 1.0, 1.0)`):
                Scale factors for rotary positional embeddings.
            sparse_params (`Dict[str, Any]`, *optional*):
                Parameters for sparse attention.
            return_dict (`bool`, defaults to `True`):
                Whether to return a dictionary or a tensor.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
                If `return_dict` is `True`, a [`~models.transformer_2d.Transformer2DModelOutput`] is returned,
                otherwise a `tuple` where the first element is the sample tensor.
        """
        batch_size, num_frames, height, width, channels = hidden_states.shape

        # 1. Process text embeddings
        text_embed = self.text_embeddings(encoder_hidden_states)
        time_embed = self.time_embeddings(timestep)

        # Add pooled text embedding to time embedding
        pooled_embed = self.pooled_text_embeddings(pooled_text_embed)
        time_embed = time_embed + pooled_embed

        # visual_embed shape: [batch_size, seq_len, model_dim]
        visual_embed = self.visual_embeddings(hidden_states)

        # 3. Text rotary embeddings
        text_rope = self.text_rope_embeddings(text_rope_pos)

        # 4. Text transformer blocks
        i = 0
        for text_block in self.text_transformer_blocks:
            if self.gradient_checkpointing and self.training:
                text_embed = torch.utils.checkpoint.checkpoint(
                    text_block, text_embed, time_embed, text_rope, use_reentrant=False
                )
                
            else:
                text_embed = text_block(text_embed, time_embed, text_rope)

            i += 1

        # 5. Prepare visual rope
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
                
        # visual_embed = visual_embed.reshape(visual_embed.shape[0], -1, visual_embed.shape[-1])
        # visual_rope = visual_rope.view(visual_rope.shape[0], -1, *list(visual_rope.shape[-4:]))
        visual_embed = visual_embed.flatten(1, 3)
        visual_rope = visual_rope.flatten(1, 3)
        
        # 6. Visual transformer blocks
        i = 0
        for visual_block in self.visual_transformer_blocks:
            if self.gradient_checkpointing and self.training:
                visual_embed = torch.utils.checkpoint.checkpoint(
                    visual_block,
                    visual_embed,
                    text_embed,
                    time_embed,
                    visual_rope,
                    # visual_rope_flat,
                    sparse_params,
                    use_reentrant=False,
                )
            else:                
                visual_embed = visual_block(
                    visual_embed, text_embed, time_embed, visual_rope, sparse_params
                )
                
                i += 1

        # 7. Output projection
        visual_embed = visual_embed.reshape(batch_size, num_frames, height // 2, width // 2, -1)
        output = self.out_layer(visual_embed, text_embed, time_embed)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
