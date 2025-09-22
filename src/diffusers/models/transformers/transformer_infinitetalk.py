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

import itertools
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm
from .transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.transformers.transformer_wan._get_qkv_projection
def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


# Copied from diffusers.models.transformers.transformer_wan._get_added_qkv_projection
def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


def assign_label_ranges(
    class_range: int = 25, class_interval: int = 5, num_audio_streams: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Assigns label ranges for each audio stream in [0, ..., class_range - 1] of size class_interval. The labels will be
    assigned in a zig-zag pattern: the first stream will get labels [0, ..., class_interval - 1], the second stream
    will get labels [class_range - 1 - class_interval, ..., class_range - 1], and so on.

    Note that as compared to the original code, the `class_range` and `class_interval` defined here are 1 greater. For
    example, in the original code the default class_range and class_interval are 24 and 4, respectively.
    """
    num_label_ranges = class_range // class_interval
    if num_audio_streams is None:
        num_audio_streams = num_label_ranges

    if num_audio_streams > num_label_ranges:
        raise ValueError(
            f"`num_audio_streams` is {num_audio_streams} but there are only {num_label_ranges} label ranges for "
            f"`class_range` {class_range} and `class_interval` {class_interval}. Please increase the number of label "
            f"ranges (e.g. by increasing `class_range`)."
        )

    # Prepare zig-zag list of label range indices [0, num_label_ranges - 1, 1, ...]
    split = math.ceil(num_label_ranges / 2)
    range_indices = list(range(num_label_ranges))
    even_indices = range_indices[:split]
    odd_indices = list(reversed(range_indices[split:]))
    # Interleave the even and odd indices lists
    label_indices = [x for x in itertools.chain(*itertools.zip_longest(even_indices, odd_indices)) if x is not None]

    label_ranges = []
    for i in range(num_audio_streams):
        label_range_idx = label_indices[i]
        label_range = (label_range_idx * class_interval, (label_range_idx + 1) * class_interval - 1)
        label_ranges.append(label_range)

    return label_ranges


def normalize_and_scale(
    x: torch.Tensor, source_range: Tuple[int, int], target_range: Tuple[int, int], eps: float = 1e-8
) -> torch.Tensor:
    """Linearly rescales a tensor to have values in [target_min, target_max] (as specified by target_range)."""
    source_min, source_max = source_range
    target_min, target_max = target_range

    normalized = (x - source_min) / (source_max - source_min + eps)
    rescaled = normalized * (target_max - target_min) + target_min
    return rescaled


# Based on transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def calculate_ref_attn_map(
    query: torch.Tensor,
    key: torch.Tensor,
    target_masks: torch.Tensor,
    mode: str = "mean",
    attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # query shape: [B, query_seq_len, H, D]
    # key shape: [B, key_seq_len, H, D]
    # target_masks shape: [num_classes, key_seq_len]
    key = key.to(device=query.device, dtype=query.dtype)
    target_masks = target_masks.to(query.dtype)

    # Calculate scaled dot-product attention
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attn_weights = query @ key.transpose(-2, -1)

    if attn_bias is not None:
        attn_weights = attn_weights + attn_bias

    attn_scores = attn_weights.softmax(-1)  # [B, H, query_seq_len, key_seq_len]

    ref_attn_maps = []
    for target_mask in target_masks:
        target_mask = target_mask[None, None, None, ...]
        ref_attn_map = attn_scores * target_mask
        # [B, H, query_seq_len, key_seq_len] --> [B, H, query_seq_len]
        ref_attn_map = ref_attn_map.sum(dim=-1) / target_mask.sum()
        ref_attn_map = ref_attn_map.permute(0, 2, 1)  # [B, H, query_seq_len] --> [B, query_seq_len, H]

        # [B, query_seq_len, H] --> [B, query_seq_len]
        if mode == "mean":
            ref_attn_map = ref_attn_map.mean(-1)
        elif mode == "max":
            ref_attn_map = ref_attn_map.max(-1)

        ref_attn_maps.append(ref_attn_map)

    # List[[B, query_seq_len]] --> [B * num_classes, query_seq_len]
    return torch.concat(ref_attn_maps, dim=0)


def get_attn_map_with_target(
    query: torch.Tensor,
    key: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    target_masks: Optional[torch.Tensor] = None,
    num_splits: int = 2,
) -> torch.Tensor:
    """
    Calculates an attention map from the self-attention operation on video latents. In InfiniteTalk, this is used for
    audio cross-attention to identify the subjects corresponding to each audio stream (if there are multiple audio
    streams).

    Args:
        query (`torch.Tensor` of shape `(batch_size, seq_len, num_heads, head_dim)`):
            The query from the attention operation. For InfiniteTalk, this would be from the video latents.
        key (`torch.Tensor` of shape `(batch_size, seq_len, num_heads, head_dim)`):
            The key from the attention operation. For InfiniteTalk, this would be from the video latents.
        grid_shape (`Tuple[int, int, int]`):
            Shape of the spatio-temporal patch grid created after the patchify operation, a 3-tuple (N_t, N_h, N_w).
        target_mask (`torch.Tensor` of shape `(num_classes, seq_len), *optional*`)
            TODO
        num_splits (`int`, *optional*, defaults to `2`):
            Number of chunks to split the attention map into. For InfiniteTalk, this would correspond to the number of
            audio streams (number of speakers in the video).

    Returns:
        `torch.Tensor` of shape `(num_classes, seq_len)`: attention maps for each class
    """
    _, num_patches_height_dim, num_patches_width_dim = grid_shape
    _, seq_len, num_heads, _ = query.shape
    num_classes, _ = target_masks.shape

    spatial_seq_len = num_patches_height_dim * num_patches_width_dim
    chunk_size = num_heads // num_splits

    key = key[:, :spatial_seq_len]  # [B, L, H, D] --> [B, spatial_seq_len, H, D]

    ref_attn_maps = torch.zeros(num_classes, seq_len).to(device=query.device, dtype=query.dtype)
    for i in range(num_splits):
        query_chunk = query[:, :, i * chunk_size : (i + 1) * chunk_size, :]
        key_chunk = key[:, :, i * chunk_size : (i + 1) * chunk_size, :]
        ref_attn_map_per_head = calculate_ref_attn_map(query_chunk, key_chunk, target_masks)
        ref_attn_maps += ref_attn_map_per_head

    return ref_attn_maps / num_splits


# Like WanAttnProcessor, but returns an attention map for multi-stream audio support
class InfiniteTalkAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "InfiniteTalkAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        target_masks: Optional[torch.Tensor] = None,
        num_audio_streams: int = 1,
        text_context_length: int = 512,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - text_context_length
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

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

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if num_audio_streams > 1:
            # Calculate attention map from the attention outputs
            ref_attn_maps = get_attn_map_with_target(query, key, grid_shape, target_masks, num_audio_streams)
            return hidden_states, ref_attn_maps
        else:
            return hidden_states


# Like WanAttention, but can optionally return an attention map
class InfiniteTalkAttention(WanAttention):
    _default_processor_cls = InfiniteTalkAttnProcessor
    _available_processors = [InfiniteTalkAttnProcessor]

    # Override WanAttention.forward to include extra args for calculating the attention mask
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        target_masks: Optional[torch.Tensor] = None,
        num_audio_streams: int = 1,
        text_context_length: int = 512,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            grid_shape,
            target_masks,
            num_audio_streams,
            text_context_length,
            **kwargs,
        )


class RotaryPositionalEmbedding1D(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2) / self.dim))  # [ceil(D / 2),]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @lru_cache(maxsize=32)
    def precompute_freqs_cis(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids shape: [B, S]
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)  # [B, D / 2, 1]
        inv_freq_expanded = inv_freq_expanded.to(position_ids.device)
        position_ids_expanded = position_ids[:, None, :]  # [B, 1, S]

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # [B, S, D / 2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [B, S, D]
        return freqs

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        unsqueeze_dim: int = 1,
        upcast_to_float: bool = True,
    ):
        # x shape: [B, H, S, D] if unsqueeze_dim == 1; in general, x should be a MHA-query-like 4D tensor
        # position_ids shape: [B, S]
        freqs = self.precompute_freqs_cis(position_ids).to(x.device)  # [B, S, D]

        if upcast_to_float:
            original_dtype = x.dtype
            x = x.float()
            freqs = freqs.float()

        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        x_embed = (x * cos) + (rotate_half(x) * sin)

        if upcast_to_float:
            x_embed = x_embed.to(dtype=original_dtype)

        return x_embed


class InfiniteTalkAudioCrossAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "InfiniteTalkAudioAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ref_attn_maps: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_audio_streams: int = 1,
    ) -> torch.Tensor:
        grid_size_t, _, _ = grid_shape  # Number of patches N_t in the temporal dimension after patchification
        batch_size, seq_len, hidden_dim = hidden_states.shape
        _, audio_seq_len, _ = encoder_hidden_states.shape
        dim_head = attn.inner_dim // attn.heads
        dim_head_kv = attn.kv_inner_dim // attn.heads

        # For audio cross-attention, reshape such that the seq_len runs over only the spatial dims
        hidden_states = hidden_states.reshape(batch_size * grid_size_t, -1, hidden_dim)  # [B * N_t, S, C]

        # q: [B * N_t, S, C] --> [B * N_t, S, D]; k,v: [B * N_t, S_a, C_a] --> [B * N_t, S_a, D_kv]
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)  # [B * N_t, H, S, D_head]
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if num_audio_streams > 1:
            attn_map_max_values = ref_attn_maps.max(dim=1)
            attn_map_min_values = ref_attn_maps.min(dim=1)
            attn_max_min_values = torch.cat(
                [attn_map_max_values[:, None, None], attn_map_min_values[:, None, None]], dim=2
            )

            # Create normalized attention maps for each audio stream
            normalized_maps = []
            for i, attn_map in enumerate(ref_attn_maps):
                stream_min_value = attn_max_min_values[i, :, 0].min()
                stream_max_value = attn_max_min_values[i, :, 0].max()

                normalized_attn_map = normalize_and_scale(
                    attn_map, (stream_min_value, stream_max_value), attn.rope_label_ranges[i]
                )
                normalized_maps.append(normalized_attn_map)

            # Create the normalized map for the background (not associated with any audio stream)
            background = torch.full(
                (ref_attn_maps.size(1),),
                attn.rope_bak,
                dtype=normalized_maps[0].dtype,
                device=normalized_maps[0].device,
            )
            normalized_maps.append(background)

            # Now get the query pos ids for the 1D RoPE application
            normalized_attn_map = torch.stack(normalized_maps, dim=1)
            max_indices = ref_attn_maps.argmax(dim=0)
            normalized_position_ids = normalized_attn_map[range(ref_attn_maps.size(1)), max_indices]  # [seq_len,]

            # For InfiniteTalk audio cross-attention, only apply RoPE when we have multiple audio streams (as part of)
            # the L-RoPE mechanism)
            query = query.reshape(batch_size, attn.heads, -1, dim_head)
            query = attn.rope_1d(query, normalized_position_ids)
            query = query.reshape(batch_size * grid_size_t, attn.heads, -1,)

            # Calculate position ids for the key
            per_stream_chunk_size = audio_seq_len // num_audio_streams
            per_audio_frame = torch.zeros(audio_seq_len, dtype=key.dtype, device=key.device)
            for i in range(num_audio_streams):
                label_range_min, label_range_max = attn.rope_label_ranges[i]
                label_range_midpoint = (label_range_min + label_range_max) / 2
                per_audio_frame[i * per_stream_chunk_size : (i + 1) * per_stream_chunk_size] = label_range_midpoint
            encoder_position_ids = torch.concat([per_audio_frame] * grid_size_t, dim=0)

            key = key.reshape(batch_size, attn.heads, -1, dim_head_kv)
            key = self.rope_id(key, encoder_position_ids)
            key = key.reshape(batch_size * grid_size_t, attn.heads, -1, dim_head_kv)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Reshape hidden_states back to original shape
        hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return hidden_states


class InfiniteTalkAudioAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = InfiniteTalkAudioCrossAttnProcessor
    _available_processors = [InfiniteTalkAudioCrossAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        qk_norm: bool = False,
        qkv_bias: bool = True,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        class_range: int = 25,
        class_interval: int = 5,
        rope_theta: float = 10000.0,
        is_cross_attention: Optional[bool] = None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=qkv_bias)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=qkv_bias)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        if qk_norm:
            self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
            self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        # InfiniteTalk audio cross-attention-specific state
        self.class_interval = class_interval
        self.class_range = class_range
        self.num_supported_audio_streams = class_interval // class_range
        self.rope_label_ranges = assign_label_ranges(class_range, class_interval)
        # Label assigned to the background (stuff not associated with any audio stream)
        self.rope_bak = int(self.class_range // 2)

        # 1D RoPE for L-RoPE mechanism
        self.rope_1d = RotaryPositionalEmbedding1D(dim_head, rope_theta)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ref_attn_maps: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_audio_streams: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            ref_attn_maps,
            grid_shape,
            num_audio_streams,
            **kwargs,
        )


class InfiniteTalkAudioAdapter(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int] = 46080,  # seq_len * blocks * channels = 5 * 12 * 768 = 46080
        input_dim_vf: Optional[int] = 110592,  # seq_len_vf * blocks * channels = 12 * 12 * 768 = 110592
        intermediate_dim: int = 512,
        output_dim: int = 768,
        seq_len: int = 5,
        seq_len_vf: int = 12,
        blocks: int = 12,
        channels: int = 768,
        context_tokens: int = 32,
        norm_output_audio: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim or seq_len * blocks * channels
        self.input_dim_vf = input_dim_vf or seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.context_tokens = context_tokens

        self.proj = nn.Linear(self.input_dim, self.intermediate_dim)
        self.proj_vf = nn.Linear(self.input_dim_vf, self.intermediate_dim)
        self.proj_concat = nn.Linear(self.intermediate_dim, self.intermediate_dim)
        self.proj_context = nn.Linear(self.intermediate_dim, self.context_tokens * self.output_dim)

        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds: torch.Tensor, audio_embeds_vf: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, window_size, blocks, channels = audio_embeds.shape
        batch_size_vf, num_frames_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        video_length = num_frames + num_frames_vf

        # Process first frame of audio embedding
        audio_embeds = audio_embeds.view(batch_size * num_frames, window_size * blocks * channels)
        audio_embeds = F.relu(self.proj(audio_embeds))
        audio_embeds = audio_embeds.view(batch_size, num_frames, -1)

        # Process subsequent frames of audio embedding
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf * num_frames_vf, window_size_vf * blocks_vf * channels_vf)
        audio_embeds_vf = F.relu(self.proj_vf(audio_embeds_vf))
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, num_frames_vf, -1)

        audio_embeds_concat = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_concat, num_frames_concat, hidden_size_concat = audio_embeds_concat.shape
        audio_embeds_concat = audio_embeds_concat.view(-1, hidden_size_concat)

        audio_embeds_concat = F.relu(self.proj_concat(audio_embeds_concat))
        context_tokens = self.proj_context(audio_embeds_concat)

        context_tokens = self.norm(context_tokens)
        context_tokens = context_tokens.view(batch_size, video_length, -1, self.context_tokens * self.output_dim)
        return context_tokens


@maybe_allow_in_graph
class InfiniteTalkTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        audio_qk_norm: bool = False,
        audio_qkv_bias: bool = True,
        audio_class_range: int = 25,
        audio_class_interval: int = 5,
        audio_rope_theta: float = 10000.0,
        audio_cross_attn_norm: bool = True,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = InfiniteTalkAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=InfiniteTalkAttnProcessor(),
        )

        # 2. Text Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Audio Cross-Attention
        self.attn3 = InfiniteTalkAudioAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            qk_norm=audio_qk_norm,
            qkv_bias=audio_qkv_bias,
            cross_attention_dim_head=dim // num_heads,
            class_range=audio_class_range,
            class_interval=audio_class_interval,
            rope_theta=audio_rope_theta,
            processor=InfiniteTalkAudioCrossAttnProcessor(),
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True) if audio_cross_attn_norm else nn.Identity()

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        target_masks: Optional[torch.Tensor] = None,
        num_audio_streams: int = 1,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=rotary_emb,
            grid_shape=grid_shape,
            target_masks=target_masks,
            num_audio_streams=num_audio_streams,
        )
        if isinstance(attn_output, tuple):
            # When we have multiple audio streams, we will return attention maps for use in the audio cross-attention
            # layer
            attn_output, ref_attn_maps = attn_output
        else:
            ref_attn_maps = None
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Text Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Audio Cross-attention
        norm_hidden_states = self.norm3(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn3(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            attention_mask=None,
            rotary_emb=None,
            ref_attn_maps=ref_attn_maps,
            grid_shape=grid_shape,
            num_audio_streams=num_audio_streams,
        )
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class InfiniteTalkTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
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
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

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
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        vae_temporal_downscale: int = 4,
        audio_layers: int = 12,
        audio_channels: int = 768,
        audio_window: int = 5,
        audio_dim: int = 512,
        audio_output_dim: int = 768,
        audio_context_tokens: int = 32,
        norm_audio_output: bool = True,
        audio_qk_norm: bool = False,
        audio_qkv_bias: bool = True,
        audio_class_range: int = 25,
        audio_class_interval: int = 5,
        audio_rope_theta: float = 10000.0,
        audio_cross_attn_norm: bool = True,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Audio adapter for audio embedding
        audio_window_vf = audio_window + vae_temporal_downscale - 1
        self.audio_adapter = InfiniteTalkAudioAdapter(
            input_dim=audio_window * audio_channels * audio_layers,
            input_dim_vf=audio_window_vf * audio_channels * audio_layers,
            intermediate_dim=audio_dim,
            output_dim=audio_output_dim,
            seq_len=audio_window,
            seq_len_vf=audio_window_vf,
            blocks=audio_layers,
            channels=audio_channels,
            context_tokens=audio_context_tokens,
            norm_output_audio=norm_audio_output,
        )

        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                InfiniteTalkTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    audio_qk_norm=audio_qk_norm,
                    audio_qkv_bias=audio_qkv_bias,
                    audio_class_range=audio_class_range,
                    audio_class_interval=audio_class_interval,
                    audio_rope_theta=audio_rope_theta,
                    audio_cross_attn_norm=audio_cross_attn_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        grid_shape = (post_patch_num_frames, post_patch_height, post_patch_width)

        # 1. Prepare RoPE positional embeddings and patch embedding
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 2. Prepare combined time, text, and image embeddings
        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 3. Prepare audio embedding using the audio adapter
        audio_cond = encoder_hidden_states_audio.to(device=hidden_states.device, dtype=hidden_states.dtype)
        audio_cond_first_frame = audio_cond[:, :1, ...]
        audio_cond_subsequent_frames = audio_cond[:, 1:, ...]
        audio_cond_subsequent_frames = audio_cond_subsequent_frames.unflatten(1, (post_patch_num_frames, -1))

        middle_index = self.config.audio_window // 2
        audio_cond_subsequent_frames_first = audio_cond_subsequent_frames[:, :, :1, :middle_index + 1, ...]
        audio_cond_subsequent_frames_mid = audio_cond_subsequent_frames[:, :, 1:-1, middle_index:middle_index+1, ...]
        audio_cond_subsequent_frames_last = audio_cond_subsequent_frames[:, :, -1:, middle_index:, ...]
        audio_cond_subsequent_frames_first = audio_cond_subsequent_frames_first.flatten(2, 3)
        audio_cond_subsequent_frames_mid = audio_cond_subsequent_frames_mid.flatten(2, 3)
        audio_cond_subsequent_frames_last = audio_cond_subsequent_frames_last.flatten(2, 3)

        audio_cond_subsequent_frames = torch.cat(
            [audio_cond_subsequent_frames_first, audio_cond_subsequent_frames_mid, audio_cond_subsequent_frames_last],
            dim=2,
        )
        encoder_hidden_states_audio = self.audio_adapter(audio_cond_first_frame, audio_cond_subsequent_frames)
        num_audio_streams = len(encoder_hidden_states_audio)
        encoder_hidden_states_audio = torch.cat(encoder_hidden_states_audio.split(1), dim=2)

        # 4. Transformer blocks
        # If supplied, convert ref_target_masks from spatial to patch token mask
        if ref_target_masks is not None:
            # expected ref_target_masks shape: [num_audio_streams, latent_height, latent_width]
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32)
            token_ref_target_masks = F.interpolate(
                ref_target_masks, size=(post_patch_height, post_patch_width), mode="nearest"
            )
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1)
            token_ref_target_masks = token_ref_target_masks.to(dtype=hidden_states.dtype)
        else:
            token_ref_target_masks = None

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_audio,
                    timestep_proj,
                    rotary_emb,
                    grid_shape,
                    token_ref_target_masks,
                    num_audio_streams,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_audio,
                    timestep_proj,
                    rotary_emb,
                    grid_shape,
                    token_ref_target_masks,
                    num_audio_streams,
                )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
