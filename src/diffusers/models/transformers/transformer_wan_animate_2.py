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
from functools import lru_cache, partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ..modeling_utils import ModelMixin


try:
    from flash_attn_interface import flash_attn_varlen_func

    FLASH_VER = 3
except ModuleNotFoundError:
    try:
        from flash_attn import flash_attn_varlen_func

        FLASH_VER = 2
    except ModuleNotFoundError:
        flash_attn_varlen_func = None
        FLASH_VER = None

from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw

# Lazy compile: compile on first call instead of at import time
_flex_compiled = None


def _get_compiled_flex_attention():
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(_flex_attention_raw, dynamic=False, mode="max-autotune", fullgraph=True)
    return _flex_compiled


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale
    # apply attention
    if FLASH_VER == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_VER == 2
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def flex_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    block_mask=None,
    kernel_options=None,
    dtype=torch.bfloat16,
    score_mod=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda"
    lq, lk, out_dtype = q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    assert lq % 128 == 0, "q_len must be divisible by 128."
    assert lk % 128 == 0, "k_len must be divisible by 128."

    # preprocess query
    if q_lens is None:
        q = half(q)
    else:
        q = half(q)
        assert q_lens.max() == q_lens.min(), "varlen of query is not supported"

    # preprocess key, value
    if k_lens is None:
        k, v = half(k), half(v)
    else:
        k, v = half(k), half(v)
        assert k_lens.max() == k_lens.min(), "varlen of key is not supported"

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    x = _get_compiled_flex_attention()(
        query=q.transpose(2, 1),
        key=k.transpose(2, 1),
        value=v.transpose(2, 1),
        block_mask=block_mask,
        kernel_options=kernel_options,
        score_mod=score_mod,
    ).transpose(2, 1)

    return x.type(out_dtype)


def _score_mod_impl(score, b_idx, h_idx, q_idx, kv_idx, hw: int, log_scale: float):
    condition = (kv_idx >= hw) & (kv_idx < 2 * hw)
    return torch.where(condition, score + log_scale, score)


@lru_cache(maxsize=32)
def _get_score_mod(hw: int, log_scale: float = -1.0):
    return partial(_score_mod_impl, hw=hw, log_scale=log_scale)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast(device_type="cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000, offset=0):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len) + offset,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast(device_type="cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, time_stride=1):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][: f * time_stride : time_stride].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):
    """
    LayerNorm without learnable affine parameters.
    """

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, *args, method, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def pre_attention(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        return q, k, v

    def post_attention(self, x):
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(SelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, use_img_emb=True):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.use_img_emb = use_img_emb
        if use_img_emb:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, counter=0):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        if self.use_img_emb:
            context_img = context[:, :257]
            context = context[:, 257:]
        else:
            context = context

        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if self.use_img_emb:
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        if self.use_img_emb:
            img_x = img_x.flatten(2)
            x = x + img_x
        x = self.o(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_img_emb=True,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)

        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)

        self.norm3 = LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, use_img_emb=use_img_emb)

        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, *args, method, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def pre_self_attention(self, x, e):
        assert e.dtype == torch.float32
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        q, k, v = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], method="pre_attention")
        return q, k, v, e

    def post_self_attention(self, x):
        x = self.self_attn(x, method="post_attention")
        return x

    def cross_attention(self, x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            x = x + y * e[5]
        return x


class IncontextAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        refer_stride=1,
        use_img_emb=True,
        sparse_type=0,
        log_scale=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.refer_stride = refer_stride
        self.sparse_type = sparse_type
        self.log_scale = log_scale

        self.block = AttentionBlock(
            dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, use_img_emb=use_img_emb
        )

    def forward(self, *args, method, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def forward_ref(self, x_ref, index, k_cache, v_cache, context_ref, freqs_ref, grid_sizes_ref, e_ref, context_lens):
        q_ref, k_ref, v_ref, e_ref = self.block(x_ref, e_ref, method="pre_self_attention")

        k_cache[index] = k_ref
        v_cache[index] = v_ref
        q_ref_add_rope = rope_apply(q_ref, grid_sizes_ref, freqs_ref, self.refer_stride)
        k_ref_add_rope = rope_apply(k_ref, grid_sizes_ref, freqs_ref, self.refer_stride)

        ref_f, ref_h, ref_w = grid_sizes_ref[0].tolist()
        ref_vail_len = ref_f * ref_h * ref_w

        xout_ref = flash_attention(
            q=q_ref_add_rope,
            k=k_ref_add_rope,
            v=v_ref,
            k_lens=torch.tensor([ref_vail_len], dtype=torch.long),
            window_size=self.window_size,
        )

        y_ref = self.block(xout_ref, method="post_self_attention")

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            x_ref = x_ref + y_ref * e_ref[2]

        x_ref = self.block(x_ref, context_ref, context_lens, e_ref, method="cross_attention")

        return x_ref

    def forward_gen(
        self,
        x,
        index,
        k_cache,
        v_cache,
        block_mask,
        context,
        freqs,
        freqs_ref,
        grid_sizes,
        grid_sizes_ref,
        origin_len,
        origin_area,
        e,
        context_lens,
    ):
        origin_latent_f = origin_len // 4 + 1
        origin_latent_hw = origin_area[0] * origin_area[1] // 256
        origin_max_len = (origin_latent_f + 1) * origin_latent_hw
        origin_ref_max_len = origin_latent_f * origin_latent_hw

        f, h, w = grid_sizes[0].tolist()
        vail_len = f * h * w
        hw = h * w

        ref_f, ref_h, ref_w = grid_sizes_ref[0].tolist()
        ref_vail_len = ref_f * ref_h * ref_w
        ref_hw = ref_h * ref_w

        q, k, v, e = self.block(x, e, method="pre_self_attention")

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        k_ref, v_ref = k_cache[index], v_cache[index]
        k_ref = rope_apply(k_ref, grid_sizes_ref, freqs_ref, self.refer_stride)

        B, _, N, C = q.shape
        device, dtype = q.device, q.dtype

        target_q_len = math.ceil(origin_max_len / 128) * 128
        target_ref_len = math.ceil(origin_ref_max_len / 128) * 128
        target_kv_len = target_q_len + target_ref_len

        q_padding = q[:, vail_len:].clone()

        q_incontext = torch.zeros(B, target_q_len, N, C, device=device, dtype=dtype)
        k_incontext = torch.zeros(B, target_kv_len, N, C, device=device, dtype=dtype)
        v_incontext = torch.zeros(B, target_kv_len, N, C, device=device, dtype=dtype)

        q_src = q[:, :vail_len].view(B, f, hw, N, C)
        k_src = k[:, :vail_len].view(B, f, hw, N, C)
        v_src = v[:, :vail_len].view(B, f, hw, N, C)

        q_incontext[:, : f * origin_latent_hw].view(B, f, origin_latent_hw, N, C)[:, :, :hw] = q_src
        k_incontext[:, : f * origin_latent_hw].view(B, f, origin_latent_hw, N, C)[:, :, :hw] = k_src
        v_incontext[:, : f * origin_latent_hw].view(B, f, origin_latent_hw, N, C)[:, :, :hw] = v_src

        k_ref_src = k_ref[:, :ref_vail_len].view(B, ref_f, ref_hw, N, C)
        v_ref_src = v_ref[:, :ref_vail_len].view(B, ref_f, ref_hw, N, C)

        k_incontext[:, target_q_len : target_q_len + ref_f * origin_latent_hw].view(B, ref_f, origin_latent_hw, N, C)[
            :, :, :ref_hw
        ] = k_ref_src
        v_incontext[:, target_q_len : target_q_len + ref_f * origin_latent_hw].view(B, ref_f, origin_latent_hw, N, C)[
            :, :, :ref_hw
        ] = v_ref_src

        score_mod = _get_score_mod(hw=int(origin_latent_hw), log_scale=self.log_scale)

        xout_full = flex_attention(
            q=q_incontext,
            k=k_incontext,
            v=v_incontext,
            block_mask=block_mask,
            kernel_options=None,
            score_mod=score_mod,
        )

        xout_valid = xout_full[:, : f * origin_latent_hw]
        xout_valid = xout_valid.view(B, f, origin_latent_hw, N, C)
        xout_vail = xout_valid[:, :, :hw]
        xout_vail = xout_vail.reshape(B, f * hw, N, C)
        xout = torch.cat([xout_vail, q_padding], dim=1)

        y = self.block(xout, method="post_self_attention")

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            x = x + y * e[2]

        x = self.block(x, context, context_lens, e, method="cross_attention")
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanAnimate2Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A Transformer model for video-like data used in the Wan-Animate-2 model.

    Wan-Animate-2 uses an in-context attention mechanism with KV cache: a reference video is first encoded
    (``forward_ref``) to cache K/V tensors, then the generation forward (``forward_gen``) uses the cached
    K/V with a block mask (``flex_attention``) and score modification (``log_scale``) for frame-level
    sparse in-context attention.

    Args:
        patch_size (`tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        text_len (`int`, defaults to `512`):
            Fixed length for text embeddings.
        in_dim (`int`, defaults to `36`):
            The number of channels in the input (2 * latent_channels + 4 for mask channel).
        dim (`int`, defaults to `5120`):
            The number of channels in the transformer.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        text_dim (`int`, defaults to `4096`):
            Input dimension for text embeddings.
        out_dim (`int`, defaults to `16`):
            The number of channels in the output.
        num_heads (`int`, defaults to `40`):
            The number of attention heads.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        use_img_emb (`bool`, defaults to `True`):
            Whether to use CLIP image embedding.
        refer_offset_t (`int`, defaults to `1`):
            RoPE offset for the temporal dimension of the reference.
        refer_offset_h (`int`, defaults to `0`):
            RoPE offset for the height dimension of the reference.
        refer_offset_w (`int`, defaults to `-1`):
            RoPE offset for the width dimension of the reference. -1 means use the generation grid size.
        refer_stride (`int`, defaults to `1`):
            Stride for RoPE application on the reference.
        sparse_type (`int`, defaults to `0`):
            Sparse attention type.
        log_scale (`float`, defaults to `0.0`):
            Log scale for score modification in in-context attention.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "img_emb", "norm"]
    _no_split_modules = ["IncontextAttentionBlock"]
    _repeated_blocks = ["IncontextAttentionBlock"]
    _keep_in_fp32_modules = [
        "time_embedding",
        "time_projection",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "modulation",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        use_img_emb: bool = True,
        refer_offset_t: int = 1,
        refer_offset_h: int = 0,
        refer_offset_w: int = -1,
        refer_stride: int = 1,
        sparse_type: int = 0,
        log_scale: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_img_emb = use_img_emb
        self.refer_offset_t = refer_offset_t
        self.refer_offset_h = refer_offset_h
        self.refer_offset_w = refer_offset_w
        self.refer_stride = refer_stride
        self.sparse_type = sparse_type
        self.log_scale = log_scale

        # [Denoising Transformer]
        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # blocks
        self.blocks = nn.ModuleList(
            [
                IncontextAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    refer_stride,
                    use_img_emb=use_img_emb,
                    sparse_type=sparse_type,
                    log_scale=log_scale,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if use_img_emb:
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()
        self.gradient_checkpointing = False
        self.block_masks = {}
        self.block_mask_grid_sizes = {}

    def create_mask(self, origin_len, origin_area, device):
        origin_latent_f = origin_len // 4 + 1
        hw = int(np.prod(origin_area).item() // 256)

        q_len = (origin_latent_f + 1) * hw
        k_len = origin_latent_f * hw

        q_len_total = math.ceil(q_len / 128) * 128
        k_extra_len_total = math.ceil(k_len / 128) * 128
        k_len_total = q_len_total + k_extra_len_total

        q_limit = q_len
        k_limit = k_len
        q_total = q_len_total

        def attention_mask_logic(b, h, q_idx, kv_idx):
            q_valid = q_idx < q_limit
            is_base_attention = kv_idx < q_limit

            q_frame = q_idx // hw
            is_first_part = kv_idx < q_total

            kv_frame_1 = kv_idx // hw
            kv_is_valid_1 = kv_idx < q_limit

            rel_kv_idx = kv_idx - q_total
            kv_frame_2 = (rel_kv_idx // hw) + 1
            kv_is_valid_2 = rel_kv_idx < k_limit

            kv_frame = torch.where(is_first_part, kv_frame_1, kv_frame_2)
            kv_is_valid = torch.where(is_first_part, kv_is_valid_1, kv_is_valid_2)

            is_cond_attention = (q_frame == kv_frame) & kv_is_valid

            return q_valid & (is_base_attention | is_cond_attention)

        block_mask = create_block_mask(
            attention_mask_logic,
            B=None,
            H=None,
            Q_LEN=q_len_total,
            KV_LEN=k_len_total,
            device=device,
            _compile=True,
        )
        return block_mask

    def forward(self, *args, method, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def forward_ref(
        self,
        x_ref,
        grid_sizes,
        k_cache,
        v_cache,
        clip_fea_ref,
        y_ref,
        context_ref,
        seq_len_ref,
        t,
    ):
        device = self.patch_embedding.weight.device
        # [reference]
        x_ref = [torch.cat([u, v], dim=0) for u, v in zip(x_ref, y_ref)]
        # embeddings
        x_ref = [self.patch_embedding(u.unsqueeze(0)) for u in x_ref]
        grid_sizes_ref = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x_ref])
        x_ref = [u.flatten(2).transpose(1, 2) for u in x_ref]
        seq_lens_ref = torch.tensor([u.size(1) for u in x_ref], dtype=torch.long)
        assert seq_lens_ref.max() <= seq_len_ref
        x_ref = torch.cat([torch.cat([u, u.new_zeros(1, seq_len_ref - u.size(1), u.size(2))], dim=1) for u in x_ref])

        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        d = self.dim // self.num_heads

        if self.refer_offset_t < 0:
            self.refer_offset_t = grid_sizes[0][0].item()
        if self.refer_offset_h < 0:
            self.refer_offset_h = grid_sizes[0][1].item()
        if self.refer_offset_w < 0:
            self.refer_offset_w = grid_sizes[0][2].item()

        self.freqs_ref = torch.cat(
            [
                rope_params(512, d - 4 * (d // 6), offset=self.refer_offset_t),
                rope_params(512, 2 * (d // 6), offset=self.refer_offset_h),
                rope_params(512, 2 * (d // 6), offset=self.refer_offset_w),
            ],
            dim=1,
        )
        if self.freqs_ref.device != device:
            self.freqs_ref = self.freqs_ref.to(device)

        # time embeddings ref
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            e_ref = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t * 0 + 1).float())
            e0_ref = self.time_projection(e_ref).unflatten(1, (6, self.dim))
            assert e_ref.dtype == torch.float32 and e0_ref.dtype == torch.float32

        # [context_ref]
        context_ref = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context_ref])
        )

        if self.use_img_emb:
            context_clip_ref = self.img_emb(clip_fea_ref)
            context_ref = torch.concat([context_clip_ref, context_ref], dim=1)

        context_lens = None
        # arguments
        kwargs = {
            "e_ref": e0_ref,
            "grid_sizes_ref": grid_sizes_ref,
            "freqs_ref": self.freqs_ref,
            "context_ref": context_ref,
            "context_lens": context_lens,
        }

        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x_ref = self._gradient_checkpointing_func(
                    block.forward_ref,
                    x_ref,
                    idx,
                    k_cache,
                    v_cache,
                    **kwargs,
                )
            else:
                x_ref = block(x_ref, idx, k_cache, v_cache, method="forward_ref", **kwargs)

    def forward_gen(
        self,
        x,
        k_cache,
        v_cache,
        clip_fea,
        y,
        context,
        seq_len,
        t,
        grid_sizes_ref,
        origin_len,
        origin_area,
        is_uncondtion=False,
    ):
        # [denoising]
        # params
        device = self.patch_embedding.weight.device
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        d = self.dim // self.num_heads
        self.freqs = torch.cat(
            [
                rope_params(512, d - 4 * (d // 6)),
                rope_params(512, 2 * (d // 6)),
                rope_params(512, 2 * (d // 6)),
            ],
            dim=1,
        )
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.refer_offset_t < 0:
            self.refer_offset_t = grid_sizes[0][0].item()
        if self.refer_offset_h < 0:
            self.refer_offset_h = grid_sizes[0][1].item()
        if self.refer_offset_w < 0:
            self.refer_offset_w = grid_sizes[0][2].item()

        self.freqs_ref = torch.cat(
            [
                rope_params(512, d - 4 * (d // 6), offset=self.refer_offset_t),
                rope_params(512, 2 * (d // 6), offset=self.refer_offset_h),
                rope_params(512, 2 * (d // 6), offset=self.refer_offset_w),
            ],
            dim=1,
        )
        if self.freqs_ref.device != device:
            self.freqs_ref = self.freqs_ref.to(device)

        # time embeddings
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # [context]
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        if self.use_img_emb:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        block_mask_id = (origin_len, origin_area[0], origin_area[1])
        if block_mask_id not in self.block_masks:
            self.block_masks[block_mask_id] = self.create_mask(origin_len, origin_area, x.device)
        block_mask = self.block_masks[block_mask_id]

        # arguments
        kwargs = {
            "e": e0,
            "block_mask": block_mask,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "grid_sizes_ref": grid_sizes_ref,
            "freqs_ref": self.freqs_ref,
            "context_lens": context_lens,
            "origin_area": origin_area,
            "origin_len": origin_len,
        }

        for idx, block in enumerate(self.blocks):
            if is_uncondtion and idx == 9:
                continue
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(
                    block.forward_gen,
                    x,
                    idx,
                    k_cache,
                    v_cache,
                    **kwargs,
                )
            else:
                x = block(x, idx, k_cache, v_cache, method="forward_gen", **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def load_from_official_state_dict(self, state_dict):
        """Load weights from the official Wan-Animate-2 checkpoint."""
        self.load_state_dict(state_dict, strict=True)
