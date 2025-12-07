# Copyright 2025 The HuggingFace Team and SANA-Video Team. All rights reserved.
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
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers, BaseOutput
from ..attention import AttentionMixin
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle, RMSNorm

from dataclasses import dataclass


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



@dataclass
class SanaBlockKvCache:
    vk: Optional[torch.Tensor] = None
    k_sum: Optional[torch.Tensor] = None
    temporal_cache: Optional[torch.Tensor] = None
    _enable_save: bool = False

    def disable_save(self):
        self._enable_save = False

    def enable_save(self):
        self._enable_save = True

    def maybe_save(
        self, 
        vk: Optional[torch.Tensor]=None, 
        k_sum: Optional[torch.Tensor]=None, 
        temporal_cache: Optional[torch.Tensor]=None,
    ):
        if not self._enable_save:
            return

        if vk is not None:
            self.vk = vk.detach().clone()
        if k_sum is not None:
            self.k_sum = k_sum.detach().clone()
        if temporal_cache is not None:
            self.temporal_cache = temporal_cache.detach().clone()


@dataclass
class SanaVideoCausalTransformer3DModelOutput(BaseOutput):
    """
    The output of [`SanaVideoCausalTransformer3DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_frames, height, width, num_channels)`):
            The hidden states output conditioned on the `encoder_hidden_states` input.
        kv_cache (`SanaKvCache`, *optional*):
            The KV cache for the transformer blocks.
    """

    sample: "torch.Tensor"  # noqa: F821
    kv_caches: Optional[List[SanaBlockKvCache]] = None


class CachedGLUMBConvTemp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: Optional[str] = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

        self.conv_temp = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[SanaBlockKvCache] = None,
    ) -> Tuple[torch.Tensor, Optional[SanaBlockKvCache]]:
        """
        hidden_states: shape [B, T, H, W, C]
        kv_cache: SanaBlockKvCache, with optional cached states (only temporal_cache is used here for temporal)
        """

        if self.residual_connection:
            residual = hidden_states

        batch_size, num_frames, height, width, num_channels = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * num_frames, height, width, num_channels).permute(0, 3, 1, 2)

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        # Temporal aggregation with kv_cache support
        hidden_states_temporal = hidden_states.view(batch_size, num_frames, num_channels, height * width).permute(
            0, 2, 1, 3
        )

        padding_size = self.conv_temp.kernel_size[0] // 2  # usually 1
        hidden_states_temporal_in = hidden_states_temporal
        padded_size = 0

        # If using cache, prepend cached frames from last chunk along time axis (dim 2)
        if kv_cache is not None:
            if kv_cache.temporal_cache is not None:
                hidden_states_temporal_in = torch.cat([kv_cache.temporal_cache, hidden_states_temporal], dim=2)
                padded_size = kv_cache.temporal_cache.shape[2]
            # Save last padding_size frames for next chunk
            kv_cache.maybe_save(
                temporal_cache=hidden_states_temporal[:, :, -padding_size:, :],
            )

        t_conv_out = self.conv_temp(hidden_states_temporal_in)[:, :, padded_size:]
        hidden_states = hidden_states_temporal + t_conv_out
        hidden_states = hidden_states.permute(0, 2, 3, 1).view(batch_size, num_frames, height, width, num_channels)

        if self.norm_type == "rms_norm":
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states, kv_cache


class SanaCausalLinearAttnProcessor1_0:
    r"""
    Processor for implementing causal linear attention with KV cache support.
    Designed for autoregressive generation scenarios.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        kv_cache: Optional[SanaBlockKvCache] = None,
    ) -> Tuple[torch.Tensor, Optional[SanaBlockKvCache]]:
        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Project input to query, key, value
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape to multi-head format: B, N, C -> B, N, H, C
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply lightweight linear attention kernel (ReLU)
        query = F.relu(query)
        key = F.relu(key)

        # Apply rotary position embeddings
        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query_rotate = apply_rotary_emb(query, *rotary_emb)
            key_rotate = apply_rotary_emb(key, *rotary_emb)

        # Permute to attention computation format: B, N, H, C -> B, H, C, N
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        query_rotate = query_rotate.permute(0, 2, 3, 1)
        key_rotate = key_rotate.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)

        # Cast to float for numerical stability
        query_rotate, key_rotate, value = query_rotate.float(), key_rotate.float(), value.float()

        # Linear attention computation with KV cache support
        # Compute key sum for normalization: sum over sequence dimension
        k_sum = key.sum(dim=-1, keepdim=True).transpose(-2, -1)  # B, H, 1, C

        # Compute value-key product: V @ K^T
        scores = torch.matmul(value, key_rotate.transpose(-1, -2))  # B, H, C, C

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            cached_vk, cached_k_sum = kv_cache.vk, kv_cache.k_sum
            kv_cache.maybe_save(vk=scores, k_sum=k_sum)
            if cached_vk is not None and cached_k_sum is not None:
                scores = scores + cached_vk
                k_sum = k_sum + cached_k_sum

        # Normalization factor: 1 / (K_sum @ Q + epsilon)
        z = 1 / (k_sum @ query + 1e-15)

        # Final attention output: (V @ K^T) @ Q
        hidden_states = torch.matmul(scores, query_rotate)

        # Apply normalization
        hidden_states = hidden_states * z

        # Reshape back: B, H, C, N -> B, N, C
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, kv_cache


# Copied from transformers.transformer_sana_video.WanRotaryPosEmbed
class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


# Copied from transformers.transformer_sana_video.SanaModulatedNorm
class SanaModulatedNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, scale_shift_table: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (scale_shift_table[None, None] + temb[:, :, None].to(scale_shift_table.device)).unbind(dim=2)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


# Copied from transformers.transformer_sana_video.SanaCombinedTimestepGuidanceEmbeddings
class SanaCombinedTimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.guidance_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor = None, hidden_dtype: torch.dtype = None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        guidance_proj = self.guidance_condition_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=hidden_dtype))
        conditioning = timesteps_emb + guidance_emb

        return self.linear(self.silu(conditioning)), conditioning


# Copied from transformers.transformer_sana_video.SanaAttnProcessor2_0
class SanaAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SanaAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SanaVideoCausalTransformerBlock(nn.Module):
    r"""
    Transformer block with KV cache support for causal linear attention.
    Used in LongSana for autoregressive generation.
    """

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 20,
        attention_head_dim: int = 112,
        dropout: float = 0.0,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 3.0,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        rope_max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        # 1. Self Attention - must use causal linear attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads if qk_norm is not None else None,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            processor=SanaCausalLinearAttnProcessor1_0(),
        )

        # 2. Cross Attention
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.attn2 = Attention(
                query_dim=dim,
                qk_norm=qk_norm,
                kv_heads=num_cross_attention_heads if qk_norm is not None else None,
                cross_attention_dim=cross_attention_dim,
                heads=num_cross_attention_heads,
                dim_head=cross_attention_head_dim,
                dropout=dropout,
                bias=True,
                out_bias=attention_out_bias,
                processor=SanaAttnProcessor2_0(),
            )

        # 3. Feed-forward - must use cached conv
        self.ff = CachedGLUMBConvTemp(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        frames: int = None,
        height: int = None,
        width: int = None,
        rotary_emb: Optional[torch.Tensor] = None,
        kv_cache: Optional[SanaBlockKvCache] = None,
    ) -> Tuple[torch.Tensor, Optional[SanaBlockKvCache]]:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None] + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)

        # 2. Self Attention with KV cache
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        # Causal linear attention always supports kv_cache
        attn_output, kv_cache = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention (no cache)
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward with KV cache
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_hidden_states = norm_hidden_states.unflatten(1, (frames, height, width))

        # Cached conv always supports kv_cache
        ff_output, kv_cache = self.ff(
            norm_hidden_states,
            kv_cache=kv_cache,
        )

        ff_output = ff_output.flatten(1, 3)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states, kv_cache


class SanaVideoCausalTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, AttentionMixin):
    r"""
    A 3D Transformer model with KV cache support for LongSana autoregressive generation.

    This model extends Sana-Video with causal linear attention and cached convolutions
    to enable efficient long video generation through chunked processing.

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `20`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `112`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of Transformer blocks to use.
        num_cross_attention_heads (`int`, *optional*, defaults to `20`):
            The number of heads to use for cross-attention.
        cross_attention_head_dim (`int`, *optional*, defaults to `112`):
            The number of channels in each head for cross-attention.
        cross_attention_dim (`int`, *optional*, defaults to `2240`):
            The number of channels in the cross-attention output.
        caption_channels (`int`, defaults to `2304`):
            The number of channels in the caption embeddings.
        mlp_ratio (`float`, defaults to `2.5`):
            The expansion ratio to use in the GLUMBConv layer.
        dropout (`float`, defaults to `0.0`):
            The dropout probability.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in the attention layer.
        sample_size (`int`, defaults to `32`):
            The base size of the input latent.
        patch_size (`int`, defaults to `1`):
            The size of the patches to use in the patch embedding layer.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether to use elementwise affinity in the normalization layer.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value for the normalization layer.
        qk_norm (`str`, *optional*, defaults to `None`):
            The normalization to use for the query and key.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SanaVideoCausalTransformerBlock", "SanaModulatedNorm"]
    _skip_layerwise_casting_patterns = ["patch_embedding", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        num_attention_heads: int = 20,
        attention_head_dim: int = 112,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 30,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        guidance_embeds: bool = False,
        guidance_embeds_scale: float = 0.1,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        rope_max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Additional condition embeddings
        if guidance_embeds:
            self.time_embed = SanaCombinedTimestepGuidanceEmbeddings(inner_dim)
        else:
            self.time_embed = AdaLayerNormSingle(inner_dim)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # 3. Transformer blocks - use causal versions
        self.transformer_blocks = nn.ModuleList(
            [
                SanaVideoCausalTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output blocks
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = SanaModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, math.prod(patch_size) * out_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        kv_caches: Optional[List[SanaBlockKvCache]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], SanaVideoCausalTransformer3DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep.flatten(), guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        timestep = timestep.view(batch_size, -1, timestep.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks with KV cache
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # Note: gradient checkpointing doesn't support kv_cache (requires tuple return)
            if kv_caches is not None:
                logger.warning("KV cache is not supported with gradient checkpointing. Disabling KV cache.")
                kv_caches = None

            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states, _ = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    rotary_emb,
                    kv_cache=None,
                )
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                # Get kv_cache for this block if available
                block_kv_cache = kv_caches[index_block] if kv_caches is not None else None

                hidden_states, block_kv_cache = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    rotary_emb,
                    kv_cache=block_kv_cache,
                )

                # Handle return value (could be tensor or tuple)
                if kv_caches is not None:
                    kv_caches[index_block] = block_kv_cache

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, kv_caches)

        return SanaVideoCausalTransformer3DModelOutput(sample=output, kv_cache=kv_caches)
