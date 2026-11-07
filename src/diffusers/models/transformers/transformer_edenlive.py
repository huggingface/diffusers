# Copyright 2026 The VAST-AI Team. All rights reserved.
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
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import apply_lora_scale
from ...utils.torch_utils import get_module_execution_device, maybe_adjust_dtype_for_device, maybe_allow_in_graph
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


def _apply_rotary_emb(hidden_states, rotary_emb):
    freqs_cos, freqs_sin = rotary_emb
    first, second = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = first * cos - second * sin
    output[..., 1::2] = first * sin + second * cos
    return output


def _get_qkv_projections(attn, hidden_states, encoder_hidden_states=None):
    context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if attn.fused_projections:
        if not attn.is_cross_attention:
            return attn.to_qkv(hidden_states).chunk(3, dim=-1)
        query = attn.to_q(hidden_states)
        key, value = attn.to_kv(context).chunk(2, dim=-1)
        return query, key, value
    return attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)


def _get_added_kv_projections(attn, image_context):
    if attn.fused_projections:
        return attn.to_added_kv(image_context).chunk(2, dim=-1)
    return attn.add_k_proj(image_context), attn.add_v_proj(image_context)


@dataclass
class _EdenLiveLayerCache:
    key: torch.Tensor
    value: torch.Tensor
    num_tokens: int = 0


@dataclass
class EdenLiveCausalState:
    """KV storage for one fixed-resolution rollout and guidance branch. Positions count latent frames."""

    layers: list[_EdenLiveLayerCache]
    chunk_size: int
    max_context_frames: int
    sink_frames: int
    next_frame: int = 0
    valid: bool = True

    def reset(self) -> None:
        """Clear committed token counts and reset the frame cursor and validity, retaining allocated cache storage."""
        for layer in self.layers:
            layer.num_tokens = 0
        self.next_frame = 0
        self.valid = True


class EdenLiveAttnProcessor:
    _attention_backend = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        *,
        text_context_length=512,
    ):
        image_context = None
        if attn.add_k_proj is not None:
            if encoder_hidden_states is None:
                raise ValueError("Image cross-attention requires image and text embeddings.")
            image_length = encoder_hidden_states.shape[1] - text_context_length
            if image_length <= 0:
                raise ValueError("The 14B model requires CLIP image embeddings before the text embeddings.")
            image_context = encoder_hidden_states[:, :image_length]
            encoder_hidden_states = encoder_hidden_states[:, image_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        image_output = None
        if image_context is not None:
            image_key, image_value = _get_added_kv_projections(attn, image_context)
            image_key = attn.norm_added_k(image_key).unflatten(2, (attn.heads, -1))
            image_value = image_value.unflatten(2, (attn.heads, -1))
            image_output = (
                dispatch_attention_fn(
                    query,
                    image_key,
                    image_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )
                .flatten(2, 3)
                .type_as(query)
            )

        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        output = output.flatten(2, 3).type_as(query)
        if image_output is not None:
            output = output + image_output
        return attn.to_out[1](attn.to_out[0](output))


class EdenLiveCausalAttnProcessor:
    _attention_backend = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        *,
        layer_cache=None,
        key_rotary_emb=None,
        cache_update=False,
        cache_window_tokens=None,
        cache_sink_tokens=0,
    ):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("The causal processor only supports unmasked self-attention.")

        query, key, value = _get_qkv_projections(attn, hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
        if layer_cache is None:
            if rotary_emb is not None:
                key = _apply_rotary_emb(key, rotary_emb)
            output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            output = output.flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](output))
        new_tokens = key.shape[1]
        capacity = layer_cache.key.shape[1]
        stored_end = layer_cache.num_tokens
        if key.dtype != layer_cache.key.dtype or key.device != layer_cache.key.device:
            raise ValueError("KV cache and attention must have the same device and dtype; recreate the causal state.")
        if new_tokens > capacity:
            raise ValueError("The current chunk exceeds the allocated KV cache.")

        if cache_update:
            if stored_end + new_tokens > capacity:
                sink = min(cache_sink_tokens, capacity - new_tokens)
                keep = capacity - sink - new_tokens
                recent_start = max(sink, stored_end - keep)
                retained = stored_end - recent_start
                if retained:
                    layer_cache.key[:, sink : sink + retained].copy_(
                        layer_cache.key[:, recent_start:stored_end].clone()
                    )
                    layer_cache.value[:, sink : sink + retained].copy_(
                        layer_cache.value[:, recent_start:stored_end].clone()
                    )
                stored_end = sink + retained
            layer_cache.key[:, stored_end : stored_end + new_tokens].copy_(key.detach())
            layer_cache.value[:, stored_end : stored_end + new_tokens].copy_(value.detach())
            stored_end += new_tokens
            layer_cache.num_tokens = stored_end
            sink = min(cache_sink_tokens, stored_end)
            recent_start = max(sink, stored_end - (cache_window_tokens - sink))
            key = torch.cat((layer_cache.key[:, :sink], layer_cache.key[:, recent_start:stored_end]), dim=1)
            value = torch.cat((layer_cache.value[:, :sink], layer_cache.value[:, recent_start:stored_end]), dim=1)
        else:
            sink = min(cache_sink_tokens, cache_window_tokens - new_tokens, stored_end)
            recent_start = max(sink, stored_end - (cache_window_tokens - sink - new_tokens))
            key = torch.cat((layer_cache.key[:, :sink], layer_cache.key[:, recent_start:stored_end], key), dim=1)
            value = torch.cat(
                (layer_cache.value[:, :sink], layer_cache.value[:, recent_start:stored_end], value), dim=1
            )

        key = _apply_rotary_emb(key, key_rotary_emb)
        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        output = output.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


class EdenLivePatchEmbedding(nn.Conv3d):
    def __init__(self, in_channels, out_channels, patch_size, proxy_channels, condition_rmsnorm):
        super().__init__(in_channels + proxy_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.proxy_channels = proxy_channels
        self.condition_norm = nn.RMSNorm(proxy_channels, eps=1e-6) if condition_rmsnorm else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.condition_norm is not None:
            base, condition = hidden_states[:, : -self.proxy_channels], hidden_states[:, -self.proxy_channels :]
            condition = condition.permute(0, 2, 3, 4, 1)
            condition = self.condition_norm(condition.float()).to(hidden_states.dtype)
            hidden_states = torch.cat((base, condition.permute(0, 4, 1, 2, 3)), dim=1)
        return super().forward(hidden_states)


class EdenLiveAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = EdenLiveAttnProcessor
    _available_processors = [EdenLiveAttnProcessor, EdenLiveCausalAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
        processor=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor or EdenLiveAttnProcessor())

    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.fuse_projections
    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.unfuse_projections
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, **kwargs)


class EdenLiveImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class EdenLiveTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = EdenLiveImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        timestep = timestep.to(dtype=encoder_hidden_states.dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.cat((encoder_hidden_states_image, encoder_hidden_states), dim=1)
        return temb, timestep_proj, encoder_hidden_states


class EdenLiveRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
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

        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, torch.empty(0).device)

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

    def forward(
        self, hidden_states: torch.Tensor, start_frame: int = 0, num_frames: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, input_frames, height, width = hidden_states.shape
        num_frames = input_frames if num_frames is None else num_frames
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        if start_frame + ppf > self.max_seq_len or max(pph, ppw) > self.max_seq_len:
            raise ValueError("The requested rotary position exceeds rope_max_seq_len.")
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.to(device=hidden_states.device, dtype=hidden_states.dtype).split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.to(device=hidden_states.device, dtype=hidden_states.dtype).split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][start_frame : start_frame + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][start_frame : start_frame + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class EdenLiveTransformerBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6, added_kv_proj_dim=None):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = EdenLiveAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps, processor=EdenLiveCausalAttnProcessor()
        )
        self.attn2 = EdenLiveAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=EdenLiveAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        *,
        layer_cache=None,
        key_rotary_emb=None,
        cache_update=False,
        cache_window_tokens=None,
        cache_sink_tokens=0,
        text_context_length=512,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            modulation = (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                tensor.squeeze(2) for tensor in modulation
            )
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            key_rotary_emb=key_rotary_emb,
            cache_update=cache_update,
            cache_window_tokens=cache_window_tokens,
            cache_sink_tokens=cache_sink_tokens,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            text_context_length=text_context_length,
        )
        hidden_states = hidden_states + attn_output
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


class EdenLiveTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    r"""
    Proxy-conditioned video transformer with bidirectional and chunk-causal inference.

    Args:
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            Temporal and spatial patch dimensions. The temporal patch size must be one.
        num_attention_heads (`int`, defaults to `40`):
            Number of attention heads.
        attention_head_dim (`int`, defaults to `128`):
            Channels per attention head.
        in_channels (`int`, defaults to `36`):
            Input latent channels before appending proxy features. This includes mask and image latent channels for
            14B.
        out_channels (`int`, defaults to `16`):
            Output latent channels.
        text_dim (`int`, defaults to `4096`):
            Input text embedding dimension.
        freq_dim (`int`, defaults to `256`):
            Sinusoidal timestep embedding dimension.
        ffn_dim (`int`, defaults to `13824`):
            Feed-forward intermediate dimension.
        num_layers (`int`, defaults to `40`):
            Number of transformer blocks.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            Query and key normalization. Only the checkpoint's normalization across heads is supported.
        eps (`float`, defaults to `1e-6`):
            Normalization epsilon.
        image_dim (`int`, *optional*):
            Input CLIP image embedding dimension for the 14B model.
        added_kv_proj_dim (`int`, *optional*):
            Dimension of projected image embeddings consumed by image cross-attention.
        rope_max_seq_len (`int`, defaults to `1024`):
            Maximum rotary position on each temporal or spatial axis.
        proxy_channels (`int`, defaults to `1024`):
            Projected DINO feature channels appended before patch embedding.
        proxy_input_channels (`int`, defaults to `1024`):
            DINO feature channels before the learned proxy projection.
        condition_rmsnorm (`bool`, defaults to `False`):
            Whether to RMS-normalize proxy features before patch embedding, as required by the 14B checkpoint.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["EdenLiveTransformerBlock", "EdenLiveTimeTextImageEmbedding", "EdenLivePatchEmbedding"]
    _repeated_blocks = ["EdenLiveTransformerBlock"]
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _keep_in_fp32_modules = ["proxy_projection"]
    _skip_keys = ["causal_state"]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        qk_norm: str = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        proxy_channels: int = 1024,
        proxy_input_channels: int = 1024,
        condition_rmsnorm: bool = False,
    ):
        super().__init__()
        if len(patch_size) != 3 or patch_size[0] != 1 or min(patch_size) < 1:
            raise ValueError("EdenLive requires temporal patch size 1 and positive spatial patch sizes.")
        if qk_norm != "rms_norm_across_heads":
            raise ValueError("EdenLive checkpoints use rms_norm_across_heads.")
        if min(proxy_input_channels, proxy_channels, num_layers) < 1:
            raise ValueError("proxy_input_channels, proxy_channels and num_layers must be positive.")
        if (image_dim is None) != (added_kv_proj_dim is None):
            raise ValueError("image_dim and added_kv_proj_dim must both be set for the 14B image-conditioned model.")
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = EdenLiveRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.cache_rope = EdenLiveRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.proxy_projection = nn.Linear(proxy_input_channels, proxy_channels)
        self.patch_embedding = EdenLivePatchEmbedding(
            in_channels, inner_dim, patch_size, proxy_channels, condition_rmsnorm
        )
        self.condition_embedder = EdenLiveTimeTextImageEmbedding(
            inner_dim, freq_dim, inner_dim * 6, text_dim, image_dim
        )
        self.blocks = nn.ModuleList(
            [
                EdenLiveTransformerBlock(inner_dim, ffn_dim, num_attention_heads, eps, added_kv_proj_dim)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.gradient_checkpointing = False

    def create_causal_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        chunk_size: int = 3,
        max_context_frames: int = 18,
        sink_frames: int = 6,
        device=None,
        dtype=None,
    ) -> EdenLiveCausalState:
        r"""
        Allocate independent KV storage for one guidance branch and rollout with fixed latent spatial dimensions.

        Args:
            batch_size (`int`): Batch size of the model inputs.
            height (`int`): Height of the latent input, before patch embedding.
            width (`int`): Width of the latent input, before patch embedding.
            chunk_size (`int`, defaults to `3`): Number of latent frames per full chunk.
            max_context_frames (`int`, defaults to `18`): Total context, including sink and current chunk.
            sink_frames (`int`, defaults to `6`): Initial latent frames retained when the context fills.
            device (`torch.device` or `str`, *optional*):
                Cache device for a model without `device_map`; defaults to the model execution device. With
                `device_map`, each layer's execution device takes precedence.
            dtype (`torch.dtype`, *optional*): Cache dtype; defaults to the model dtype.

        Returns:
            `EdenLiveCausalState`: Empty KV storage for one rollout and guidance branch.
        """
        _, patch_height, patch_width = self.config.patch_size
        if min(batch_size, height, width, chunk_size, max_context_frames) < 1:
            raise ValueError("Batch size, latent dimensions, chunk size and context size must be positive.")
        if height % patch_height or width % patch_width:
            raise ValueError("Latent dimensions must be divisible by the patch size.")
        if not 0 <= sink_frames < max_context_frames or max_context_frames < chunk_size:
            raise ValueError("The context must contain the current chunk and be larger than its nonnegative sink.")
        if max_context_frames % chunk_size or sink_frames % chunk_size:
            raise ValueError("The fixed context and sink must contain whole chunks.")
        tokens_per_frame = (height // patch_height) * (width // patch_width)
        shape = (
            batch_size,
            max_context_frames * tokens_per_frame,
            self.config.num_attention_heads,
            self.config.attention_head_dim,
        )
        layers = []
        for block in self.blocks:
            cache_device = (
                get_module_execution_device(block)
                if getattr(self, "hf_device_map", None)
                else device or get_module_execution_device(self)
            )
            layers.append(
                _EdenLiveLayerCache(
                    torch.zeros(shape, device=cache_device, dtype=dtype or self.dtype),
                    torch.zeros(shape, device=cache_device, dtype=dtype or self.dtype),
                )
            )
        return EdenLiveCausalState(
            layers,
            chunk_size,
            max_context_frames,
            sink_frames,
        )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        proxy_features: torch.Tensor | None = None,
        causal_state: EdenLiveCausalState | None = None,
        cache_update: bool = False,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        r"""
        Denoise one clip or chunk. A clean-chunk call with `cache_update=True` advances the causal state.

        Args:
            hidden_states (`torch.Tensor`):
                Input latents of shape `(batch_size, in_channels, num_frames, height, width)`, including the image
                conditioning channels for 14B.
            timestep (`torch.Tensor`):
                Diffusion timestep, as a scalar, `(batch_size,)`, `(batch_size, num_frames)`, or `(batch_size,
                num_patch_tokens)`. Preserve float32 timestep precision when using reduced-precision model weights.
            encoder_hidden_states (`torch.Tensor`):
                Text embeddings of shape `(batch_size, sequence_length, text_dim)`.
            encoder_hidden_states_image (`torch.Tensor`, *optional*):
                CLIP image embeddings required by the 14B model.
            proxy_features (`torch.Tensor`, *optional*):
                Required DINO features of shape `(batch_size, proxy_input_channels, num_frames, proxy_height,
                proxy_width)`. Features are projected and normalized before their spatial dimensions are interpolated
                to match the latent input.
            causal_state (`EdenLiveCausalState`, *optional*):
                KV storage created by `create_causal_state`. Use separate states for different guidance branches.
            cache_update (`bool`, defaults to `False`):
                Whether to commit a clean chunk to the KV state and advance its frame cursor. Denoising calls leave the
                cache unchanged.
            return_dict (`bool`, defaults to `True`):
                Whether to return a structured output instead of a tuple.
            attention_kwargs (`dict[str, Any]`, *optional*):
                Attention options, including the LoRA `scale` when using PEFT adapters.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The predicted latent tensor, with shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        if hidden_states.ndim != 5 or hidden_states.shape[1] != self.config.in_channels:
            raise ValueError(f"hidden_states must have shape [B, {self.config.in_channels}, T, H, W].")
        batch_size, _, num_frames, height, width = hidden_states.shape
        patch_frames, patch_height, patch_width = self.config.patch_size
        if num_frames < 1 or height % patch_height or width % patch_width:
            raise ValueError(
                "Input spatial dimensions must be divisible by the patch size and the clip must be nonempty."
            )
        if proxy_features is None or proxy_features.ndim != 5:
            raise ValueError("proxy_features must be supplied as a [B, C, T, H, W] tensor.")
        if proxy_features.shape[:2] != (batch_size, self.config.proxy_input_channels):
            raise ValueError("Proxy batch size and channels must match the configured model.")
        if proxy_features.shape[2] != num_frames:
            raise ValueError("Proxy features must have one control frame per latent frame.")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("encoder_hidden_states must have shape [B, sequence_length, text_dim].")
        if self.config.image_dim is not None and encoder_hidden_states_image is None:
            raise ValueError("This image-conditioned model requires CLIP encoder_hidden_states_image.")
        if self.config.image_dim is None and encoder_hidden_states_image is not None:
            raise ValueError("The 5B model does not use CLIP image embeddings.")

        tokens_per_frame = (height // patch_height) * (width // patch_width)
        num_tokens = num_frames * tokens_per_frame
        text_context_length = encoder_hidden_states.shape[1]
        key_rotary_emb = None
        cache_window_tokens = None
        cache_sink_tokens = 0
        query_start_frame = 0
        if cache_update and causal_state is None:
            raise ValueError("cache_update requires a causal state.")
        if causal_state is not None:
            state = causal_state
            if not state.valid:
                raise ValueError("A previous cache commit failed. Reset or recreate the causal state.")
            cache_shape = (
                batch_size,
                state.max_context_frames * tokens_per_frame,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
            )
            if len(state.layers) != len(self.blocks) or any(
                layer.key.shape != cache_shape or layer.value.shape != cache_shape for layer in state.layers
            ):
                raise ValueError("The causal state does not match the model's batch size or token dimensions.")
            if state.next_frame % state.chunk_size or num_frames > state.chunk_size:
                raise ValueError("Causal calls must follow whole chunks; a shorter chunk can only finish a rollout.")
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                raise ValueError("Mutable causal inference state cannot be used with gradient checkpointing.")
            query_start_frame = min(state.next_frame, state.max_context_frames - state.chunk_size)
            visible_frames = query_start_frame + num_frames
            cache_window_tokens = visible_frames * tokens_per_frame
            cache_sink_tokens = state.sink_frames * tokens_per_frame
            key_rotary_emb = self.cache_rope(hidden_states, num_frames=visible_frames)

        rotary_emb = self.rope(hidden_states, start_frame=query_start_frame)
        projected_features = []
        for frame in proxy_features.unbind(2):
            frame = frame.permute(0, 2, 3, 1).contiguous().to(device=hidden_states.device, dtype=torch.float32)
            with torch.autocast(hidden_states.device.type, dtype=torch.bfloat16, enabled=hidden_states.is_cuda):
                projected = self.proxy_projection(frame)
                projected = F.normalize(projected, p=2, dim=-1).float()
            projected_features.append(projected.permute(0, 3, 1, 2))
        proxy_features = torch.stack(projected_features, dim=2)
        proxy_features = proxy_features.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if proxy_features.shape[2:] != hidden_states.shape[2:]:
            proxy_features = F.interpolate(
                proxy_features, size=hidden_states.shape[2:], mode="trilinear", align_corners=False
            )
        hidden_states = self.patch_embedding(torch.cat((hidden_states, proxy_features), dim=1))
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.ndim == 1 and timestep.shape[0] == 1:
            timestep = timestep.expand(batch_size)
        if timestep.ndim not in (1, 2) or timestep.shape[0] != batch_size:
            raise ValueError("timestep must be scalar, [B], [B, latent_frames], or [B, patch_tokens].")
        timestep_seq_len = None
        if timestep.ndim == 2:
            if timestep.shape[1] == num_frames:
                timestep = timestep.repeat_interleave(tokens_per_frame, dim=1)
            if timestep.shape[1] != num_tokens:
                raise ValueError("Per-token timesteps must match the latent frame count or patch token count.")
            timestep_seq_len = num_tokens
            timestep = timestep.flatten()
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=timestep_seq_len
        )
        timestep_proj = timestep_proj.unflatten(2 if timestep_seq_len is not None else 1, (6, -1))
        if cache_update:
            causal_state.valid = False
        for index, block in enumerate(self.blocks):
            kwargs = {
                "layer_cache": causal_state.layers[index] if causal_state is not None else None,
                "key_rotary_emb": key_rotary_emb,
                "cache_update": cache_update,
                "cache_window_tokens": cache_window_tokens,
                "cache_sink_tokens": cache_sink_tokens,
                "text_context_length": text_context_length,
            }
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    partial(block, **kwargs), hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, **kwargs)

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states_dtype = hidden_states.dtype
        hidden_states = self.norm_out(hidden_states.float())
        hidden_states = (hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)).to(
            hidden_states_dtype
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            num_frames,
            height // patch_height,
            width // patch_width,
            patch_frames,
            patch_height,
            patch_width,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if cache_update:
            causal_state.next_frame += num_frames
            causal_state.valid = True
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
