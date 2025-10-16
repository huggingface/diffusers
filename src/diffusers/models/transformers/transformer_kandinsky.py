# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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
from torch import BoolTensor, IntTensor, Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, FeedForward
from ..cache_utils import CacheMixin
from ..embeddings import (
    TimestepEmbedding,
    get_1d_rotary_pos_embed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm

logger = logging.get_logger(__name__)


def exist(item):
    return item is not None


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=dim, dtype=torch.float32)
        / dim
    )
    return freqs


def fractal_flatten(x, rope, shape, block_mask=False):
    if block_mask:
        pixel_size = 8
        x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=1)
        rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=1)
        x = x.flatten(1, 2)
        rope = rope.flatten(1, 2)
    else:
        x = x.flatten(1, 3)
        rope = rope.flatten(1, 3)
    return x, rope


def fractal_unflatten(x, shape, block_mask=False):
    if block_mask:
        pixel_size = 8
        x = x.reshape(x.shape[0], -1, pixel_size**2, *x.shape[2:])
        x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=1)
    else:
        x = x.reshape(*shape, *x.shape[2:])
    return x


def local_patching(x, shape, group_size, dim=0):
    batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        g1,
        height // g2,
        g2,
        width // g3,
        g3,
        *x.shape[dim + 3 :],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 2,
        dim + 4,
        dim + 1,
        dim + 3,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def local_merge(x, shape, group_size, dim=0):
    batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        height // g2,
        width // g3,
        g1,
        g2,
        g3,
        *x.shape[dim + 2 :],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 3,
        dim + 1,
        dim + 4,
        dim + 2,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x


def nablaT_v2(
    q: Tensor,
    k: Tensor,
    sta: Tensor,
    thr: float = 0.9,
) -> BlockMask:
    # Map estimation
    B, h, S, D = q.shape
    s1 = S // 64
    qa = q.reshape(B, h, s1, 64, D).mean(-2)
    ka = k.reshape(B, h, s1, 64, D).mean(-2).transpose(-2, -1)
    map = qa @ ka

    map = torch.softmax(map / math.sqrt(D), dim=-1)
    # Map binarization
    vals, inds = map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))

    mask = torch.logical_or(mask, sta)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None
    )


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x, scale, shift):
    return (norm(x) * (scale + 1.0) + shift).to(torch.bfloat16)


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x, out, gate):
    return (x + gate * out).to(torch.bfloat16)


@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x, rope):
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(torch.bfloat16)


class Kandinsky5TimeEmbeddings(nn.Module):
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

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        return time_embed

    def reset_dtype(self):
        self.freqs = get_freqs(self.model_dim // 2, self.max_period)


class Kandinsky5TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class Kandinsky5VisualEmbeddings(nn.Module):
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


class Kandinsky5RoPE1D(nn.Module):
    def __init__(self, dim, max_pos=1024, max_period=10000.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer(f"args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos):
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)

    def reset_dtype(self):
        freq = get_freqs(self.dim // 2, self.max_period).to(self.args.device)
        pos = torch.arange(self.max_pos, dtype=freq.dtype, device=freq.device)
        self.args = torch.outer(pos, freq)


class Kandinsky5RoPE3D(nn.Module):
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

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).repeat(
                    batch_size, 1, height, width, 1
                ),
                args_h.view(1, 1, height, 1, -1).repeat(
                    batch_size, duration, 1, width, 1
                ),
                args_w.view(1, 1, 1, width, -1).repeat(
                    batch_size, duration, height, 1, 1
                ),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)

    def reset_dtype(self):
        for i, (axes_dim, ax_max_pos) in enumerate(zip(self.axes_dims, self.max_pos)):
            freq = get_freqs(axes_dim // 2, self.max_period).to(self.args_0.device)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype, device=freq.device)
            setattr(self, f"args_{i}", torch.outer(pos, freq))


class Kandinsky5Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        return self.out_layer(self.activation(x))


class Kandinsky5SDPAAttentionProcessor(nn.Module):
    """Custom attention processor for standard SDPA attention"""

    def __call__(
        self,
        attn,
        query,
        key,
        value,
        **kwargs,
    ):
        # Process attention with the given query, key, value tensors
        
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(query, key, value).transpose(1, 2).contiguous().flatten(-2, -1)

        return out


class Kandinsky5NablaAttentionProcessor(nn.Module):
    """Custom attention processor for Nabla attention"""
    
    @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
    def __call__(
        self,
        attn,
        query,
        key,
        value,
        sparse_params=None,
        **kwargs,
    ):
        if sparse_params is None:
            raise ValueError("sparse_params is required for Nabla attention")

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        block_mask = nablaT_v2(
            query,
            key,
            sparse_params["sta_mask"],
            thr=sparse_params["P"],
        )
        out = (
            flex_attention(query, key, value, block_mask=block_mask)
            .transpose(1, 2)
            .contiguous()
        )
        out = out.flatten(-2, -1)
        return out


class Kandinsky5MultiheadSelfAttentionEnc(nn.Module):
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

        # Initialize attention processor
        self.sdpa_processor = Kandinsky5SDPAAttentionProcessor()

    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def scaled_dot_product_attention(self, query, key, value):
        # Use the processor
        return self.sdpa_processor(attn=self, query=query, key=key, value=value, **{})

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        out = self.scaled_dot_product_attention(query, key, value)

        out = self.out_l(out)
        return out


class Kandinsky5MultiheadSelfAttentionDec(nn.Module):
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

        # Initialize attention processors
        self.sdpa_processor = Kandinsky5SDPAAttentionProcessor()
        self.nabla_processor = Kandinsky5NablaAttentionProcessor()

    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def attention(self, query, key, value):
        # Use the processor
        return self.sdpa_processor(attn=self, query=query, key=key, value=value, **{})

    def nabla(self, query, key, value, sparse_params=None):
        # Use the processor
        return self.nabla_processor(
            attn=self,
            query=query,
            key=key,
            value=value,
            sparse_params=sparse_params,
            **{},
        )

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope, sparse_params=None):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        if sparse_params is not None:
            out = self.nabla(query, key, value, sparse_params=sparse_params)
        else:
            out = self.attention(query, key, value)

        out = self.out_l(out)
        return out


class Kandinsky5MultiheadCrossAttention(nn.Module):
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

        # Initialize attention processor
        self.sdpa_processor = Kandinsky5SDPAAttentionProcessor()

    def get_qkv(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape, cond_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def attention(self, query, key, value):
        # Use the processor
        return self.sdpa_processor(attn=self, query=query, key=key, value=value, **{})

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, cond):
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.attention(query, key, value)
        out = self.out_l(out)
        return out


class Kandinsky5FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class Kandinsky5OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky5Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

    def forward(self, visual_embed, text_embed, time_embed):
        shift, scale = torch.chunk(
            self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        visual_embed = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale[:, None, None],
            shift[:, None, None],
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        batch_size, duration, height, width, _ = x.shape
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
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )
        return x


class Kandinsky5TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.text_modulation = Kandinsky5Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky5MultiheadSelfAttentionEnc(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)

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


class Kandinsky5TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.visual_modulation = Kandinsky5Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky5MultiheadSelfAttentionDec(model_dim, head_dim)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = Kandinsky5MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)

    def forward(self, visual_embed, text_embed, time_embed, rope, sparse_params):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(
            self.self_attention_norm, visual_embed, scale, shift
        )
        visual_out = self.self_attention(visual_out, rope, sparse_params)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(
            self.cross_attention_norm, visual_embed, scale, shift
        )
        visual_out = self.cross_attention(visual_out, text_embed)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(
            self.feed_forward_norm, visual_embed, scale, shift
        )
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        return visual_embed


class Kandinsky5Transformer3DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    A 3D Diffusion Transformer model for video-like data.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        attention_type: str = "regular",
        attention_causal: bool = None,  # Default for Nabla: false
        attention_local: bool = None,  # Default for Nabla: false
        attention_glob: bool = None,  # Default for Nabla: false
        attention_window: int = None,  # Default for Nabla: 3
        attention_P: float = None,  # Default for Nabla: 0.9
        attention_wT: int = None,  # Default for Nabla: 11
        attention_wW: int = None,  # Default for Nabla: 3
        attention_wH: int = None,  # Default for Nabla: 3
        attention_add_sta: bool = None,  # Default for Nabla: true
        attention_method: str = None,  # Default for Nabla: "topcdf"
    ):
        super().__init__()

        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.attention_type = attention_type

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim

        # Initialize embeddings
        self.time_embeddings = Kandinsky5TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = Kandinsky5TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = Kandinsky5TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = Kandinsky5VisualEmbeddings(
            visual_embed_dim, model_dim, patch_size
        )

        # Initialize positional embeddings
        self.text_rope_embeddings = Kandinsky5RoPE1D(head_dim)
        self.visual_rope_embeddings = Kandinsky5RoPE3D(axes_dims)

        # Initialize transformer blocks
        self.text_transformer_blocks = nn.ModuleList(
            [
                Kandinsky5TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim)
                for _ in range(num_text_blocks)
            ]
        )

        self.visual_transformer_blocks = nn.ModuleList(
            [
                Kandinsky5TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
                for _ in range(num_visual_blocks)
            ]
        )

        # Initialize output layer
        self.out_layer = Kandinsky5OutLayer(
            model_dim, time_dim, out_visual_dim, patch_size
        )
        self.gradient_checkpointing = False

    def prepare_text_embeddings(
        self, text_embed, time, pooled_text_embed, x, text_rope_pos
    ):
        """Prepare text embeddings and related components"""
        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(time)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        text_rope = text_rope.unsqueeze(dim=0)
        return text_embed, time_embed, text_rope, visual_embed

    def prepare_visual_embeddings(
        self, visual_embed, visual_rope_pos, scale_factor, sparse_params
    ):
        """Prepare visual embeddings and related components"""
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(
            visual_shape, visual_rope_pos, scale_factor
        )
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(
            visual_embed, visual_rope, visual_shape, block_mask=to_fractal
        )
        return visual_embed, visual_shape, to_fractal, visual_rope

    def process_text_transformer_blocks(self, text_embed, time_embed, text_rope):
        """Process text through transformer blocks"""
        for text_transformer_block in self.text_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                text_embed = self._gradient_checkpointing_func(
                    text_transformer_block, text_embed, time_embed, text_rope
                )
            else:
                text_embed = text_transformer_block(text_embed, time_embed, text_rope)
        return text_embed

    def process_visual_transformer_blocks(
        self, visual_embed, text_embed, time_embed, visual_rope, sparse_params
    ):
        """Process visual through transformer blocks"""
        for visual_transformer_block in self.visual_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                visual_embed = self._gradient_checkpointing_func(
                    visual_transformer_block,
                    visual_embed,
                    text_embed,
                    time_embed,
                    visual_rope,
                    sparse_params,
                )
            else:
                visual_embed = visual_transformer_block(
                    visual_embed, text_embed, time_embed, visual_rope, sparse_params
                )
        return visual_embed

    def prepare_output(
        self, visual_embed, visual_shape, to_fractal, text_embed, time_embed
    ):
        """Prepare the final output"""
        visual_embed = fractal_unflatten(
            visual_embed, visual_shape, block_mask=to_fractal
        )
        x = self.out_layer(visual_embed, text_embed, time_embed)
        return x

    def forward(
        self,
        hidden_states: torch.FloatTensor,  # x
        encoder_hidden_states: torch.FloatTensor,  # text_embed
        timestep: Union[torch.Tensor, float, int],  # time
        pooled_projections: torch.FloatTensor,  # pooled_text_embed
        visual_rope_pos: Tuple[int, int, int],
        text_rope_pos: torch.LongTensor,
        scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, torch.FloatTensor]:
        """
        Forward pass of the Kandinsky5 3D Transformer.

        Args:
            hidden_states (`torch.FloatTensor`): Input visual states
            encoder_hidden_states (`torch.FloatTensor`): Text embeddings
            timestep (`torch.Tensor` or `float` or `int`): Current timestep
            pooled_projections (`torch.FloatTensor`): Pooled text embeddings
            visual_rope_pos (`Tuple[int, int, int]`): Position for visual RoPE
            text_rope_pos (`torch.LongTensor`): Position for text RoPE
            scale_factor (`Tuple[float, float, float]`, optional): Scale factor for RoPE
            sparse_params (`Dict[str, Any]`, optional): Parameters for sparse attention
            return_dict (`bool`, optional): Whether to return a dictionary

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `torch.FloatTensor`:
            The output of the transformer
        """
        x = hidden_states
        text_embed = encoder_hidden_states
        time = timestep
        pooled_text_embed = pooled_projections

        # Prepare text embeddings and related components
        text_embed, time_embed, text_rope, visual_embed = self.prepare_text_embeddings(
            text_embed, time, pooled_text_embed, x, text_rope_pos
        )

        # Process text through transformer blocks
        text_embed = self.process_text_transformer_blocks(
            text_embed, time_embed, text_rope
        )

        # Prepare visual embeddings and related components
        visual_embed, visual_shape, to_fractal, visual_rope = (
            self.prepare_visual_embeddings(
                visual_embed, visual_rope_pos, scale_factor, sparse_params
            )
        )

        # Process visual through transformer blocks
        visual_embed = self.process_visual_transformer_blocks(
            visual_embed, text_embed, time_embed, visual_rope, sparse_params
        )

        # Prepare final output
        x = self.prepare_output(
            visual_embed, visual_shape, to_fractal, text_embed, time_embed
        )

        if not return_dict:
            return x

        return Transformer2DModelOutput(sample=x)
