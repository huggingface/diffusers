# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import logging
from ...utils.peft_utils import apply_lora_scale
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Each vision-language image slot represents a 2×2 group of latent tokens.
_IMG_TOKENS_PER_SLOT = 4

# `create_block_mask` quantizes the mask to 128-token blocks.
_FLEX_BLOCK_SIZE = 128

# flex_attention is optional. When available we use a compiled flex_attention with a BlockMask for
# efficient single-pass block-causal attention. When unavailable, we fall back to an exact multi-pass
# SDPA prefill that processes each image block with bidirectional attention and text segments with
# causal attention, matching the block-causal mask exactly.
_FLEX_AVAILABLE = False
_compiled_flex_attention = None
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    _FLEX_AVAILABLE = True
except ImportError:
    BlockMask = None
    flex_attention = None


def _get_compiled_flex_attention():
    """Return a compiled flex_attention, compiling on first call.

    Compiling is required for the block-sparse kernel that avoids materializing the full Q@K^T matrix. Without it
    flex_attention falls back to a dense fp32 math path that OOMs on long sequences.
    """
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


class QwenImage21KVLayerCache:
    """Per-layer KV cache for text and condition-image prefix tokens.

    Stores K and V projections (post-RoPE) for the prefix extracted during the first denoising step. Tensor format:
    ``(batch_size, num_prefix_tokens, num_heads, head_dim)``.
    """

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def store(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k, self.v

    @property
    def is_populated(self) -> bool:
        return self.k is not None

    def clear(self):
        self.k = None
        self.v = None


class QwenImage21KVCache:
    """Container for all transformer blocks' prefix KV caches."""

    def __init__(self, num_layers: int):
        self.layer_caches = [QwenImage21KVLayerCache() for _ in range(num_layers)]

    def get_layer(self, layer_idx: int) -> QwenImage21KVLayerCache:
        return self.layer_caches[layer_idx]

    def clear(self):
        for cache in self.layer_caches:
            cache.clear()


# Copied from diffusers.models.transformers.transformer_qwenimage.apply_rotary_emb_qwen
def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenImage21TemporalTimesteps(nn.Module):
    r"""Sinusoidal timestep embedding. `cos` occupies the first half of the channels and `sin` the second."""

    def __init__(self, timestep_dim: int, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor

        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False
        )

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    r"""
    RMSNorm whose learnable weight is stored zero-centered: the effective scale is `weight + 1`, computed in fp32.
    Checkpoints therefore store `scale - 1`.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class QwenImage21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states))


class QwenImage21AdaLayerNormContinuous(nn.Module):
    r"""
    Final adaptive norm. Scale only — this variant emits no shift, so `linear` maps to `embedding_dim` rather than `2 *
    embedding_dim`.
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1 + scale)


def _select_modulation_rows(params: torch.Tensor, target_token_mask: torch.Tensor | None) -> torch.Tensor:
    r"""
    Broadcast per-sample modulation `params` over the token axis.

    With `causal_condition`, `params` holds `batch_size + 1` rows: rows `[0, batch_size)` come from the real timestep
    and the trailing row from `t = 0`. Text and condition-image tokens take the `t = 0` row, target-image tokens take
    their own sample's row.

    Args:
        params (`torch.Tensor`): `(batch_size, dim)` without `causal_condition`, else `(batch_size + 1, dim)`.
        target_token_mask (`torch.Tensor`, *optional*): `(seq_len,)` bool, `True` at target-image positions. `None`
            disables the split and every token uses its own sample's row.
    """
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


def build_qwenimage21_block_causal_mask(
    image_ids: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
):
    r"""
    Build the block-causal [`~torch.nn.attention.flex_attention.BlockMask`] for Qwen-Image 2.1.

    The mask is `(q_idx >= kv_idx) or same_image_block`: the joint text/image sequence is causal, while every image
    block — each condition image and the target image — is internally bidirectional. Text tokens are strictly causal.
    Cross-sample isolation is implicit in diffusers because samples live on the batch axis.

    Positions masked out by `encoder_hidden_states_mask` are excluded as *keys* so right-padded prompts cannot be
    attended to. They are kept as queries so their rows are never fully masked.

    Provided as a module-level function so callers can build the mask outside the transformer's compiled region.

    Args:
        image_ids (`torch.Tensor`): `(seq_len,)` int, `-1` at text positions and a unique non-negative id per image
            block. See [`~QwenImage21Transformer2DModel.build_token_metadata`].
        encoder_hidden_states_mask (`torch.Tensor`, *optional*): `(batch_size, seq_len)` bool over the joint sequence.
        batch_size (`int`): Number of samples; the mask varies across the batch only through
            `encoder_hidden_states_mask`.
    """
    seq_len = image_ids.shape[0]
    padded_seq_len = int(math.ceil(seq_len / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE)

    image_ids = F.pad(image_ids, (0, padded_seq_len - seq_len), value=-1)
    if encoder_hidden_states_mask is None:
        key_valid = torch.ones(batch_size, padded_seq_len, dtype=torch.bool, device=device)
    else:
        key_valid = F.pad(encoder_hidden_states_mask.bool(), (0, padded_seq_len - seq_len), value=False)

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        is_padding = (q_idx >= seq_len) | (kv_idx >= seq_len)
        q_image_id, kv_image_id = image_ids[q_idx], image_ids[kv_idx]
        same_image_block = (q_image_id == kv_image_id) & (q_image_id >= 0)
        allowed = ((q_idx >= kv_idx) | same_image_block) & key_valid[batch_idx, kv_idx]
        return allowed & ~is_padding

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=padded_seq_len,
        KV_LEN=padded_seq_len,
        device=device,
        _compile=False,
    )


def _qwenimage21_prepare_qkv(
    attn: "QwenImage21Attention",
    hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor | None,
    layer_cache: QwenImage21KVLayerCache | None,
    kv_cache_mode: str | None,
    cache_write_slice: slice | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Shared QKV projection, norm, RoPE and KV-cache bookkeeping for both processors."""
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    query = attn.norm_q(query).to(value.dtype)
    key = attn.norm_k(key).to(value.dtype)

    if rotary_emb is not None:
        query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
        key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)

    if layer_cache is not None:
        if kv_cache_mode == "extract" and cache_write_slice is not None:
            # `clone()`, not `contiguous()`: at batch size 1 the prefix slice already counts as contiguous
            # (size-1 dims are ignored), so `contiguous()` returns the same view and the cache would pin the
            # whole prefill K/V for every step of the denoising loop.
            layer_cache.store(
                key[:, cache_write_slice].clone(),
                value[:, cache_write_slice].clone(),
            )
        elif kv_cache_mode == "cached":
            cached_k, cached_v = layer_cache.get()
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)
        elif kv_cache_mode == "extend":
            # Read existing cache (if any), prepend to current KV for attention, then store the full
            # concatenated KV back. Used by the multi-pass SDPA prefill to accumulate segment-by-segment.
            if layer_cache.is_populated:
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim=1)
                value = torch.cat([cached_v, value], dim=1)
            layer_cache.store(key.contiguous(), value.contiguous())

    seq_len_q = query.shape[1]
    return query, key, value, seq_len_q


class QwenImage21FlexAttnProcessor:
    r"""
    Attention processor for Qwen-Image 2.1 using compiled `flex_attention` with a `BlockMask` for exact block-causal
    attention. This is the recommended path and is required for high resolutions (2048²+) where a dense score matrix
    would OOM.

    ``flex_attention`` is compiled on the first forward pass so the block-sparse kernel is used instead of the dense
    fallback. The first call will be slower due to compilation.
    """

    _attention_backend = "flex"
    _parallel_config = None

    def __call__(
        self,
        attn: "QwenImage21Attention",
        hidden_states: torch.Tensor,
        attention_mask: Any | None = None,
        rotary_emb: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )

        seq_len_kv = key.shape[1]
        if isinstance(attention_mask, BlockMask):
            pad_q = int(math.ceil(seq_len_q / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE) - seq_len_q
            pad_kv = int(math.ceil(seq_len_kv / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE) - seq_len_kv
            if pad_q:
                query = F.pad(query.transpose(1, 3), (0, pad_q)).transpose(1, 3)
            if pad_kv:
                key = F.pad(key.transpose(1, 3), (0, pad_kv)).transpose(1, 3)
                value = F.pad(value.transpose(1, 3), (0, pad_kv)).transpose(1, 3)

            hidden_states = _get_compiled_flex_attention()(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                block_mask=attention_mask,
            ).transpose(1, 2)
        else:
            # Decode path or no BlockMask: full attention via SDPA.
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask if not isinstance(attention_mask, type(None)) and not is_causal else None,
                dropout_p=0.0,
                is_causal=is_causal,
                backend=None,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class QwenImage21SDPAAttnProcessor:
    r"""
    SDPA attention processor for Qwen-Image 2.1. Use this when `flex_attention` is not available.

    The block-causal mask is implemented exactly via a multi-pass prefill orchestrated by the model's `forward`: each
    image block in the prefix is processed with bidirectional attention within the block and full attention to all
    preceding segments, and text segments get a causal mask. The target image attends fully to the cached prefix +
    itself. This matches the block-causal mask without requiring `flex_attention`.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "QwenImage21Attention",
        hidden_states: torch.Tensor,
        attention_mask: Any | None = None,
        rotary_emb: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask if not isinstance(attention_mask, type(None)) and not is_causal else None,
            dropout_p=0.0,
            is_causal=is_causal,
            backend=None,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class QwenImage21Attention(torch.nn.Module, AttentionModuleMixin):
    r"""
    Attention module for [`QwenImage21TransformerBlock`]. Projection layout matches the legacy
    [`~models.attention_processor.Attention`] so Qwen-Image 2.x checkpoints load into it unchanged.
    """

    _default_processor_cls = QwenImage21FlexAttnProcessor if _FLEX_AVAILABLE else QwenImage21SDPAAttnProcessor
    _available_processors = [QwenImage21FlexAttnProcessor, QwenImage21SDPAAttnProcessor]

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6, processor: Any | None = None):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        # Read by `AttentionModuleMixin.fuse_projections`; 2.1 has no biases anywhere.
        self.use_bias = False

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.set_processor(processor if processor is not None else self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


@maybe_allow_in_graph
class QwenImage21TransformerBlock(nn.Module):
    r"""
    Single-stream block. Modulation is not learned per block — the parent model computes one shared `modulation` tensor
    and every block slices its own scales and gates out of it.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attention_mask: Any | None = None,
        target_token_mask: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_output = self.attn(
            hidden_states=img_modulated,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            is_causal=is_causal,
        )
        hidden_states = hidden_states + img_gate1.tanh() * attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class QwenImage21Rope(nn.Module):
    r"""
    3-axis (frame, height, width) rotary embedding over the joint text/image sequence.

    Text tokens advance a shared position on all three axes. Every image block freezes the frame axis at the position
    reached by the preceding text and lays its tokens out on a height/width grid centred on zero, so a block's spatial
    positions do not depend on where it sits in the sequence.
    """

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        pos_index = torch.arange(8192)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.freqs = [
            torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0)
            for dim in axes_dim
        ]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self, img_shapes: list[tuple[int, int, int]], image_pad_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        self.freqs = [freq.to(device) for freq in self.freqs]

        frame_index, height_index, width_index = [], [], []
        image_height_index, image_width_index = [], []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()

        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))

        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index = frame_index.clone()
        width_index = frame_index.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)

        return torch.cat([self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)


class QwenImage21Transformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    The single-stream Transformer used by Qwen-Image 2.1.

    Text and image latents share one sequence: condition-image tokens are substituted into the text stream at the
    positions the vision-language encoder reserved for them, and the target image's tokens are appended. A single
    shared `modulation` projection feeds every block, so blocks hold no modulation parameters of their own.

    Two behaviours distinguish 2.1:

    - **Block-causal attention** — attention follows `(q_idx >= kv_idx) or same_image_block`, so the sequence is causal
      while each image block stays internally bidirectional. The `flex` attention backend gives efficient single-pass
      attention; without it the model uses an exact multi-pass SDPA prefill that processes each block separately.
    - `causal_condition` — text and condition-image tokens are modulated from `t = 0` instead of the sampled timestep,
      which also makes their activations timestep-independent and so cacheable across denoising steps.

    Args:
        patch_size (`int`, defaults to `1`):
            Side length of the latent patch folded into the channel dim. 2.1 consumes latents unpatched.
        in_channels (`int`, defaults to `64`):
            Latent channels of the input.
        out_channels (`int`, *optional*, defaults to `64`):
            Latent channels of the output. Falls back to `in_channels`.
        num_layers (`int`, defaults to `32`):
            Number of single-stream blocks.
        attention_head_dim (`int`, defaults to `128`):
            Channels per attention head.
        num_attention_heads (`int`, defaults to `32`):
            Number of attention heads.
        context_in_dim (`int`, defaults to `4096`):
            Channel dim of `encoder_hidden_states`.
        mlp_ratio (`int`, defaults to `3`):
            Feed-forward expansion factor.
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`):
            Rotary dims for the frame, height and width axes.
        eps (`float`, defaults to `1e-6`):
            Epsilon for the norm layers.
        causal_condition (`bool`, defaults to `True`):
            Modulate text and condition-image tokens from `t = 0`. Required for KV caching.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImage21TransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImage21TransformerBlock"]
    _skip_keys = ["kv_cache"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)

        # One shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False

    @staticmethod
    def build_token_metadata(
        image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Label every token of the joint sequence with the image block it belongs to.

        Block boundaries come from the token counts in `img_shapes`, not from runs of `True` in `image_pad_mask`: two
        condition images that happen to sit next to each other with no text between them form one run but must stay
        separate blocks, otherwise they would attend to each other bidirectionally.

        Args:
            image_pad_mask (`torch.Tensor`): `(seq_len,)` bool, `True` at image-token positions.
            img_shapes (`list[tuple[int, int, int]]`): Per-image `(frame, height, width)` in latent tokens, condition
                images first and the target image last.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: `image_ids` `(seq_len,)` with `-1` at text positions and a unique id
            per image block, and `target_token_mask` `(seq_len,)` marking the target image's tokens.
        """
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            raise ValueError(
                f"img_shapes accounts for {sum(block_lengths)} image tokens but image_pad_mask marks "
                f"{image_positions.numel()}."
            )

        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_lengths), device=image_pad_mask.device),
            torch.tensor(block_lengths, device=image_pad_mask.device),
        )
        image_ids[image_positions] = block_ids

        target_token_mask = torch.zeros_like(image_pad_mask)
        target_token_mask[image_positions[-block_lengths[-1] :]] = True
        return image_ids, target_token_mask

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Packed latents, condition images first and the target image last.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, context_in_dim)`):
                Text embeddings from the vision-language encoder.
            timestep (`torch.Tensor`):
                Current denoising step, scaled to `[0, 1]`.
            img_shapes (`list[list[tuple[int, int, int]]]`):
                Per-sample list of `(frame, height, width)` in latent tokens, condition images first and the target
                image last. All samples must share a layout.
            img_mask (`torch.Tensor` of shape `(batch_size, vlm_sequence_length)`):
                `True` at the vision-language encoder's image slots, each standing for a `2x2` group of latent tokens.
            encoder_hidden_states_mask (`torch.Tensor`, *optional*):
                `(batch_size, text_sequence_length)` bool marking valid text tokens. Padded positions are excluded from
                attention.
            kv_cache (`QwenImage21KVCache`, *optional*):
                Cache container. Pass together with `kv_cache_mode` to enable prefix KV caching.
            kv_cache_mode (`str`, *optional*):
                `"extract"` to prefill the cache (first denoising step), `"cached"` to decode from it (later steps).
                Requires `causal_condition=True`.
        """

        batch_size = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Each vision-language image slot stands for 2x2 latent tokens, so expand those positions four-fold and drop
        # the actual latents into them. Samples share a layout, hence the single row.
        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states

        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.config.causal_condition:
            # Extra t=0 row; text and condition-image tokens modulate from it. `modulation_mask` selects which row
            # each token reads, and is `None` when every token shares the sampled timestep.
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        if kv_cache is not None and not self.config.causal_condition:
            raise ValueError(
                "kv_cache requires `causal_condition=True`. The cache is only valid because text and condition-image "
                "tokens modulate from t=0, which makes their activations independent of the denoising step."
            )
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(
                f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}."
            )

        # Right-padded prompt positions must never be attended to, on any path. Text positions of the joint sequence
        # line up, in order, with the non-image positions of the vision-language sequence — the two are interleaved,
        # so the mask cannot be sliced off as a prefix.
        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(
                batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        prefix_len = int((~target_token_mask).sum())
        is_decode = kv_cache_mode == "cached"

        if is_decode:
            # Only the target image's queries are recomputed. The block-causal mask degenerates to full attention
            # for target rows (they can see the entire prefix + their own block), so no structural mask is needed.
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
            cache_write_slice = None
            use_multi_pass = False
        elif _FLEX_AVAILABLE:
            # flex path: single-pass with a compiled BlockMask
            cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
            attention_mask = build_qwenimage21_block_causal_mask(
                image_ids, joint_key_valid, batch_size, hidden_states.device
            )
            use_multi_pass = False
        else:
            # Multi-pass SDPA: exact block-causal attention without flex_attention.
            # The prefix is split into segments at image-block boundaries (using image_ids). Image-block
            # segments get full (bidirectional) attention within themselves, and text segments get a causal
            # mask within the segment. Both attend fully to all preceding segments via an accumulating KV
            # cache ("extend" mode). This exactly matches the block-causal mask.
            # The target image pass is unchanged: full attention over [cached prefix, target].
            use_multi_pass = True
            cache_write_slice = None

        if use_multi_pass:
            prefix_hs = joint_hidden_states[:, :prefix_len]
            target_hs = joint_hidden_states[:, prefix_len:]
            prefix_rope = rotary_emb[:prefix_len]
            target_rope = rotary_emb[prefix_len:]
            prefix_mod_mask = modulation_mask[:prefix_len] if modulation_mask is not None else None
            target_mod_mask = modulation_mask[prefix_len:] if modulation_mask is not None else None

            # Build segment boundaries from image_ids in the prefix. Consecutive tokens with the same
            # image_id form one segment (-1 = text, >=0 = image block).
            prefix_ids = image_ids[:prefix_len]
            segments = []
            if prefix_len > 0:
                seg_start = 0
                for i in range(1, prefix_len):
                    if prefix_ids[i] != prefix_ids[i - 1]:
                        segments.append((seg_start, i))
                        seg_start = i
                segments.append((seg_start, prefix_len))

            for index_block, block in enumerate(self.transformer_blocks):
                layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
                accumulated_cache = QwenImage21KVLayerCache()
                segment_outputs = []

                for seg_start, seg_end in segments:
                    seg_hs = prefix_hs[:, seg_start:seg_end]
                    seg_rope = prefix_rope[seg_start:seg_end]
                    seg_mod = prefix_mod_mask[seg_start:seg_end] if prefix_mod_mask is not None else None

                    # Text segments (image_id == -1) get a causal mask within the segment so that
                    # each text token only sees earlier text tokens + the full cached prefix. Image
                    # blocks get None (full / bidirectional), which is exact for the block-causal mask.
                    seg_mask = None
                    seg_is_text = prefix_ids[seg_start].item() < 0
                    seg_len = seg_end - seg_start
                    if seg_is_text and seg_len > 1:
                        cached_len = accumulated_cache.k.shape[1] if accumulated_cache.is_populated else 0
                        prefix_visible = torch.ones(seg_len, cached_len, dtype=torch.bool, device=hidden_states.device)
                        causal_part = torch.tril(
                            torch.ones(seg_len, seg_len, dtype=torch.bool, device=hidden_states.device)
                        )
                        seg_mask = torch.cat([prefix_visible, causal_part], dim=1)[None, None]

                    seg_hs = block(
                        hidden_states=seg_hs,
                        modulation=modulation,
                        rotary_emb=seg_rope,
                        attention_mask=seg_mask,
                        target_token_mask=seg_mod,
                        layer_cache=accumulated_cache,
                        kv_cache_mode="extend",
                        cache_write_slice=None,
                    )
                    segment_outputs.append(seg_hs)

                prefix_hs = torch.cat(segment_outputs, dim=1) if segment_outputs else prefix_hs

                # Store the accumulated prefix cache into the main cache for this layer.
                if layer_cache is not None and accumulated_cache.is_populated:
                    layer_cache.store(*accumulated_cache.get())

                # Target: full attention over [cached prefix, target].
                target_cache = (
                    layer_cache
                    if layer_cache is not None
                    else (accumulated_cache if accumulated_cache.is_populated else None)
                )
                target_hs = block(
                    hidden_states=target_hs,
                    modulation=modulation,
                    rotary_emb=target_rope,
                    attention_mask=None,
                    target_token_mask=target_mod_mask,
                    layer_cache=target_cache,
                    kv_cache_mode="cached" if target_cache is not None else None,
                    cache_write_slice=None,
                )

            joint_hidden_states = torch.cat([prefix_hs, target_hs], dim=1)
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    joint_hidden_states = self._gradient_checkpointing_func(
                        block,
                        joint_hidden_states,
                        modulation,
                        rotary_emb,
                        attention_mask,
                        modulation_mask,
                        layer_cache,
                        kv_cache_mode,
                        cache_write_slice,
                    )
                else:
                    joint_hidden_states = block(
                        hidden_states=joint_hidden_states,
                        modulation=modulation,
                        rotary_emb=rotary_emb,
                        attention_mask=attention_mask,
                        target_token_mask=modulation_mask,
                        layer_cache=layer_cache,
                        kv_cache_mode=kv_cache_mode,
                        cache_write_slice=cache_write_slice,
                    )

        joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask)
        output = self.proj_out(joint_hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
