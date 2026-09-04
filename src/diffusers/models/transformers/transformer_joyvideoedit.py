# Copyright 2026 The JoyAI-Video-Edit Team and The HuggingFace Team. All rights reserved.
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
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...hooks.joyvideoedit_kv_cache import _JOYVIDEOEDIT_KV_CACHE_HOOK, JoyVideoEditKVCacheState
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


# Visual-token roles encoded by the source-id RoPE.
SOURCE_ID_TARGET = 0.0
SOURCE_ID_EDIT_CONDITION = 1.0
SOURCE_ID_EXTRA_REF_IMAGE = 2.0

TIME_FREQ_DIM = 256

NORM_EPS = 1e-6
NUM_MODULATION_CHUNKS = 6

SELF_ATTN_MODE_REF_IMAGE_CACHE = "ref_image_cache"


# ---------------------------------------------------------------------------
# Rotary position embedding utilities
# ---------------------------------------------------------------------------


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply rotary embeddings to a `(B, L, H, D)` tensor."""
    cos = freqs_cis[0].unsqueeze(2).to(x.device)
    sin = freqs_cis[1].unsqueeze(2).to(x.device)

    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)

    return (x.float() * cos + x_rotated * sin).type_as(x)


def _concat_kv_entries(
    entries: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    cached_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Concatenate a sequence of stored KV-cache entries into a single `(key, value)` pair.

    Entries stored with `pre_rope=True` were cached *before* RoPE was applied to their key (so that RoPE can be
    re-derived from `current_temporal_ids` at read time instead of being frozen at write time). For those entries, RoPE
    is applied here using consecutive slices of `cached_freqs_cis`, in the same order the entries are concatenated.
    """
    keys = []
    values = []
    pre_rope_offset = 0

    for entry in entries:
        if entry is None:
            continue
        key = entry.get("key")
        value = entry.get("value")
        if key is None or value is None:
            continue

        key = key.to(device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)

        if entry.get("pre_rope", False) and cached_freqs_cis is not None:
            cos_all, sin_all = cached_freqs_cis
            seg_len = key.shape[1]
            cos_seg = cos_all[..., pre_rope_offset : pre_rope_offset + seg_len, :]
            sin_seg = sin_all[..., pre_rope_offset : pre_rope_offset + seg_len, :]
            key = _apply_rotary_emb(key, (cos_seg, sin_seg))
            pre_rope_offset += seg_len

        keys.append(key)
        values.append(value)

    if not keys:
        return None, None

    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


def _clone_kv_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().clone()


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------


class JoyVideoEditModulate(nn.Module):
    """Wan-style learnable modulation table.

    Produces `factor` modulation vectors by adding the conditioning signal to a learnable parameter table.
    """

    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.randn(1, factor, hidden_size, dtype=dtype, device=device) / hidden_size**0.5,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


# ---------------------------------------------------------------------------
# Attention processor
# ---------------------------------------------------------------------------


class JoyVideoEditAttnProcessor:
    """Joint self-attention processor for `JoyVideoEditAttention`.

    Computes fused QKV projections for the image and (optionally) text streams, applies per-head RMSNorm and 3D RoPE
    (with the source-id RoPE already folded into `image_rotary_emb` / `text_rotary_emb` by the caller), then runs joint
    attention over `[img, txt, *cached_kv]`. KV-cache read/write is handled here since it must happen right after the
    image stream's key/value are produced (before the joint concat with text).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        pass

    def __call__(
        self,
        attn: "JoyVideoEditAttention",
        hidden_states: torch.Tensor,  # image stream (B, S_img, D)
        encoder_hidden_states: torch.Tensor | None = None,  # text stream (B, S_txt, D)
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,  # text padding mask (B, S_txt), True = keep
        skip_text_stream: bool = False,
        kv_cache_reader: Callable[[int | None], Iterable[dict[str, torch.Tensor]]] | None = None,
        kv_cache_writer: Callable[[int | None, torch.Tensor, torch.Tensor], None] | None = None,
        layer_idx: int | None = None,
        kv_cache_pre_rope: bool = False,
        cached_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        heads = attn.heads

        img_qkv = attn.img_attn_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)
        img_query = img_query.unflatten(-1, (heads, -1))
        img_key = img_key.unflatten(-1, (heads, -1))
        img_value = img_value.unflatten(-1, (heads, -1))

        if kv_cache_pre_rope:
            # Cache the key right after QK-norm but *before* RoPE, so cached entries can be re-rotated later from
            # whatever `current_temporal_ids` apply at read time (needed for non-contiguous / relative temporal ids).
            img_key_for_cache = attn.img_attn_k_norm(img_key)
        img_query = attn.img_attn_q_norm(img_query)
        img_key = attn.img_attn_k_norm(img_key)
        if image_rotary_emb is not None:
            img_query, img_key = (
                _apply_rotary_emb(img_query, image_rotary_emb),
                _apply_rotary_emb(img_key, image_rotary_emb),
            )
        if not kv_cache_pre_rope:
            img_key_for_cache = img_key

        if not skip_text_stream:
            txt_qkv = attn.txt_attn_qkv(encoder_hidden_states)
            txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)
            txt_query = txt_query.unflatten(-1, (heads, -1))
            txt_key = txt_key.unflatten(-1, (heads, -1))
            txt_value = txt_value.unflatten(-1, (heads, -1))
            txt_query = attn.txt_attn_q_norm(txt_query)
            txt_key = attn.txt_attn_k_norm(txt_key)

        if skip_text_stream:
            query = img_query
            key = img_key
            value = img_value
        else:
            query = torch.cat((img_query, txt_query), dim=1)
            key = torch.cat((img_key, txt_key), dim=1)
            value = torch.cat((img_value, txt_value), dim=1)

        if kv_cache_writer is not None:
            kv_cache_writer(layer_idx, img_key_for_cache, img_value)

        if kv_cache_reader is not None:
            cached_key, cached_value = _concat_kv_entries(
                kv_cache_reader(layer_idx),
                device=query.device,
                dtype=query.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
            )
        else:
            cached_key = cached_value = None

        if cached_key is not None:
            key = torch.cat([cached_key, key], dim=1)
            value = torch.cat([cached_value, value], dim=1)

        # Build the joint-attention mask so padded text tokens never contribute to any query's softmax. Only the text
        # stream carries padding; the image stream and any cached image KV are always valid. The key order is
        # `[cached_img_kv, img, txt]` (text last), so the mask is all-ones over the leading visual span and equals
        # `encoder_hidden_states_mask` over the trailing text span. Masking the key positions (broadcast over queries
        # via the `(B, 1, 1, S_key)` shape) is sufficient — the padded text queries produce garbage rows that are
        # discarded, since only the image tokens are read out downstream.
        attn_mask = None
        if not skip_text_stream and encoder_hidden_states_mask is not None:
            num_visual_keys = key.shape[1] - encoder_hidden_states.shape[1]
            visual_mask = encoder_hidden_states_mask.new_ones((key.shape[0], num_visual_keys))
            attn_mask = torch.cat([visual_mask, encoder_hidden_states_mask], dim=1)[:, None, None, :]

        joint_hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(query.dtype)

        if skip_text_stream:
            img_attn_output = joint_hidden_states
            txt_attn_output = None
        else:
            img_attn_output = joint_hidden_states[:, : hidden_states.shape[1]]
            txt_attn_output = joint_hidden_states[:, hidden_states.shape[1] :]

        img_attn_output = attn.img_attn_proj(img_attn_output)
        if txt_attn_output is not None:
            txt_attn_output = attn.txt_attn_proj(txt_attn_output)

        return img_attn_output, txt_attn_output


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------


class JoyVideoEditAttention(nn.Module, AttentionModuleMixin):
    """Joint attention module for JoyVideoEdit double-stream blocks.

    Wraps the fused QKV projections, per-head RMSNorm, and output projections for both the image and text streams.
    Delegates the attention computation (RoPE, joint attention, KV-cache read/write) to a pluggable
    `JoyVideoEditAttnProcessor`.
    """

    _default_processor_cls = JoyVideoEditAttnProcessor
    _available_processors = [JoyVideoEditAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = NORM_EPS,
        processor=None,
    ):
        super().__init__()

        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.img_attn_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.img_attn_q_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.img_attn_k_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.img_attn_proj = nn.Linear(inner_dim, dim, bias=True)

        self.txt_attn_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.txt_attn_q_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.txt_attn_k_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.txt_attn_proj = nn.Linear(inner_dim, dim, bias=True)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.processor(self, hidden_states, encoder_hidden_states, image_rotary_emb, **kwargs)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class JoyVideoEditTransformerBlock(nn.Module):
    """Double-stream transformer block for JoyVideoEdit."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        eps: float = NORM_EPS,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_width_ratio)

        self.img_mod = JoyVideoEditModulate(dim, factor=NUM_MODULATION_CHUNKS)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        self.txt_mod = JoyVideoEditModulate(dim, factor=NUM_MODULATION_CHUNKS)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        self.attn = JoyVideoEditAttention(dim, num_attention_heads, attention_head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        kv_cache_reader: Callable[[int | None], Iterable[dict[str, torch.Tensor]]] | None = None,
        kv_cache_writer: Callable[[int | None, torch.Tensor, torch.Tensor], None] | None = None,
        layer_idx: int | None = None,
        skip_text_stream: bool = False,
        kv_cache_pre_rope: bool = False,
        cached_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(temb)
        if not skip_text_stream:
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(temb)

        img_modulated = self.img_norm1(hidden_states) * (1 + img_mod1_scale.unsqueeze(1)) + img_mod1_shift.unsqueeze(1)
        txt_modulated = None
        if not skip_text_stream:
            txt_modulated = self.txt_norm1(encoder_hidden_states) * (
                1 + txt_mod1_scale.unsqueeze(1)
            ) + txt_mod1_shift.unsqueeze(1)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            skip_text_stream=skip_text_stream,
            kv_cache_reader=kv_cache_reader,
            kv_cache_writer=kv_cache_writer,
            layer_idx=layer_idx,
            kv_cache_pre_rope=kv_cache_pre_rope,
            cached_freqs_cis=cached_freqs_cis,
        )

        hidden_states = hidden_states + img_attn * img_mod1_gate.unsqueeze(1)
        img_mod2_modulated = self.img_norm2(hidden_states) * (
            1 + img_mod2_scale.unsqueeze(1)
        ) + img_mod2_shift.unsqueeze(1)
        hidden_states = hidden_states + self.img_mlp(img_mod2_modulated) * img_mod2_gate.unsqueeze(1)

        if not skip_text_stream:
            encoder_hidden_states = encoder_hidden_states + txt_attn * txt_mod1_gate.unsqueeze(1)
            txt_mod2_modulated = self.txt_norm2(encoder_hidden_states) * (
                1 + txt_mod2_scale.unsqueeze(1)
            ) + txt_mod2_shift.unsqueeze(1)
            encoder_hidden_states = encoder_hidden_states + self.txt_mlp(txt_mod2_modulated) * txt_mod2_gate.unsqueeze(
                1
            )

        return hidden_states, encoder_hidden_states


# Copied from diffusers.models.transformers.transformer_joyimage.JoyImageTimeTextImageEmbedding with JoyImage->JoyVideoEdit
class JoyVideoEditTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class JoyVideoEditTransformer3DModel(ModelMixin, ConfigMixin, AttentionMixin, CacheMixin):
    """JoyVideoEdit streaming video-editing transformer.

    A dual-stream MM-DiT with source-id RoPE, cross-chunk KV caching, and reference-video latent conditioning for
    chunk-wise causal video editing.
    """

    _skip_layerwise_casting_patterns = ["img_in", "condition_embedder", "norm"]
    _no_split_modules = ["JoyVideoEditTransformerBlock"]
    _supports_gradient_checkpointing = True
    _keep_in_fp32_modules = [
        "time_embedder",
        "norm1",
        "norm2",
        "norm_out",
    ]
    _repeated_blocks = ["JoyVideoEditTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: list[int] = [1, 1, 1],
        in_channels: int = 64,
        out_channels: int | None = None,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        text_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        num_layers: int = 40,
        rope_dim_list: list[int] = [16, 56, 56],
        theta: int = 256,
        chunk_size: int = 1,
        local_window_size: int = 3,
        global_sink_chunk: bool = True,
        source_id_rope_dim: int = 128,
        source_id_rope_theta: float = 256.0,
    ):
        if chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")
        if local_window_size <= 0:
            raise ValueError(f"`local_window_size` must be positive, got {local_window_size}.")
        if source_id_rope_dim < 0 or source_id_rope_dim % 2 != 0:
            raise ValueError(f"`source_id_rope_dim` must be a non-negative even integer, got {source_id_rope_dim}.")

        super().__init__()

        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_dim_list = rope_dim_list
        self.theta = theta
        self.source_id_rope_dim = int(source_id_rope_dim)
        self.source_id_rope_theta = float(source_id_rope_theta)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        attention_head_dim = hidden_size // num_attention_heads

        self.img_in = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = JoyVideoEditTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=TIME_FREQ_DIM,
            time_proj_dim=hidden_size * NUM_MODULATION_CHUNKS,
            text_embed_dim=text_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                JoyVideoEditTransformerBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=NORM_EPS)
        self.proj_out = nn.Linear(hidden_size, self.out_channels * math.prod(patch_size))

        self.gradient_checkpointing = False
        self._kv_cache_chunk_id = None
        self._kv_cache_selected_chunk_ids = None
        self._kv_cache_pre_rope = False

    # ------------------------------------------------------------------
    # KV-cache bookkeeping
    #
    # Clean chunks are stored per layer and reused as attention context by later chunks. Cache tensors live in the
    # state managed by `JoyVideoEditKVCacheHook`; the model controls cache selection, assembly, and eviction.
    # ------------------------------------------------------------------

    def _kv_cache_state(self) -> JoyVideoEditKVCacheState:
        registry = getattr(self, "_diffusers_hook", None)
        hook = registry.get_hook(_JOYVIDEOEDIT_KV_CACHE_HOOK) if registry is not None else None
        if hook is None:
            raise RuntimeError(
                "The JoyVideoEdit KV cache is not enabled. Call `enable_cache(JoyVideoEditKVCacheConfig())` before "
                "using `kv_cache_mode`/`kv_cache_selected_chunk_ids`."
            )
        return hook.state_manager.get_state()

    def configure_inference_kv_cache(
        self,
        *,
        chunk_id: int | None = None,
        selected_chunk_ids: list[int] | None = None,
        pre_rope: bool = False,
    ) -> None:
        self._kv_cache_chunk_id = chunk_id
        self._kv_cache_selected_chunk_ids = list(selected_chunk_ids) if selected_chunk_ids is not None else None
        self._kv_cache_pre_rope = bool(pre_rope)

    def _read_layer_kv_cache(self, layer_idx: int | None) -> list[dict[str, torch.Tensor]]:
        if layer_idx is None:
            return []
        chunk_cache = self._kv_cache_state().chunk_cache
        selected_chunk_ids = self._kv_cache_selected_chunk_ids or []
        layer_entries = []
        for selected_chunk_id in selected_chunk_ids:
            chunk_store = chunk_cache.get(selected_chunk_id)
            if chunk_store is None:
                continue
            entry = chunk_store.get(layer_idx)
            if entry is not None:
                layer_entries.append(entry)
        return layer_entries

    def _write_layer_kv_cache(
        self,
        layer_idx: int | None,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if layer_idx is None or self._kv_cache_chunk_id is None:
            return
        state = self._kv_cache_state()
        chunk_store = state.chunk_cache.setdefault(self._kv_cache_chunk_id, {})
        chunk_store[layer_idx] = {
            "key": _clone_kv_tensor(key),
            "value": _clone_kv_tensor(value),
            "pre_rope": bool(self._kv_cache_pre_rope),
        }

    def evict_kv_cache_chunks(self, chunk_ids_to_keep: set) -> None:
        """Drop every cached chunk whose id is not in `chunk_ids_to_keep`."""
        state = self._kv_cache_state()
        evict_ids = [cid for cid in state.chunk_cache if cid not in chunk_ids_to_keep]
        for cid in evict_ids:
            del state.chunk_cache[cid]

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def get_rotary_pos_embed_from_ids(
        self,
        *,
        frame_ids: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build 3D `(temporal, height, width)` RoPE from explicit per-token temporal positions.

        `frame_ids` has shape `(batch_size, sequence_length)` and supports non-contiguous or relative ids.
        """
        post_patch_height, post_patch_width = spatial_shape
        device = frame_ids.device
        temporal_positions = frame_ids.to(dtype=torch.float32)
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_positions.shape[1] % spatial_tokens_per_frame != 0:
            raise ValueError(
                f"`frame_ids` length {temporal_positions.shape[1]} is not divisible by spatial token count "
                f"{spatial_tokens_per_frame}."
            )

        h_positions = torch.arange(post_patch_height, dtype=torch.float32, device=device)
        w_positions = torch.arange(post_patch_width, dtype=torch.float32, device=device)
        h_grid, w_grid = torch.meshgrid(h_positions, w_positions, indexing="ij")
        num_frames_in_grid = temporal_positions.shape[1] // spatial_tokens_per_frame
        batch_size = temporal_positions.shape[0]
        h_positions = h_grid.reshape(-1).repeat(num_frames_in_grid).unsqueeze(0).expand(batch_size, -1)
        w_positions = w_grid.reshape(-1).repeat(num_frames_in_grid).unsqueeze(0).expand(batch_size, -1)

        head_dim = self.hidden_size // self.num_attention_heads
        rope_dim_list = self.rope_dim_list
        if sum(rope_dim_list) != head_dim:
            raise ValueError("sum(rope_dim_list) should equal to head_dim of attention layer")

        cos_list = []
        sin_list = []
        for dim, positions in zip(rope_dim_list, (temporal_positions, h_positions, w_positions)):
            cos, sin = get_1d_rotary_pos_embed(dim, positions.reshape(-1), theta=self.theta, use_real=True)
            cos_list.append(cos.unflatten(0, (batch_size, -1)))
            sin_list.append(sin.unflatten(0, (batch_size, -1)))
        vis_freqs = (torch.cat(cos_list, dim=2), torch.cat(sin_list, dim=2))

        return vis_freqs

    def generate_source_id_rope(
        self,
        source_id: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token rotary phase encoding a token's *role* (target / edit-condition / extra-ref-image), independent
        of its spatiotemporal position. Only the first `source_id_rope_dim` head channels carry a role-dependent angle;
        the remaining channels get an identity rotation (cos=1, sin=0). Composed with the 3D spatiotemporal RoPE in
        `forward` via `new_cos = cos_3d * cos_role - sin_3d * sin_role` (i.e. adding the two rotation angles), so a
        token's final rotary phase is "spatiotemporal position" + "role".
        """
        role_dim = max(0, min(int(self.source_id_rope_dim), int(head_dim)))

        half_head = head_dim // 2
        cos_half = torch.ones(*source_id.shape, half_head, device=device, dtype=torch.float32)
        sin_half = torch.zeros(*source_id.shape, half_head, device=device, dtype=torch.float32)

        inv_freq = 1.0 / (
            self.source_id_rope_theta ** (torch.arange(0, role_dim, 2, device=device, dtype=torch.float32) / role_dim)
        )

        angles = source_id.unsqueeze(-1) * inv_freq
        cos_half[..., : role_dim // 2] = torch.cos(angles)
        sin_half[..., : role_dim // 2] = torch.sin(angles)
        return (
            cos_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
            sin_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
        )

    @staticmethod
    def _get_patch_shape(latent: torch.Tensor, patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
        _, _, num_frames, height, width = latent.shape
        return (
            num_frames // patch_size[0],
            height // patch_size[1],
            width // patch_size[2],
        )

    @staticmethod
    def _get_token_frame_ids(
        post_patch_shape: tuple[int, int, int],
        device: torch.device,
        temporal_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_frames, post_patch_height, post_patch_width = post_patch_shape
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_ids is None:
            frame_ids = torch.arange(num_frames, device=device, dtype=torch.long).unsqueeze(0)
        else:
            frame_ids = torch.as_tensor(temporal_ids, device=device, dtype=torch.long)
            if frame_ids.ndim != 2 or frame_ids.shape[1] != num_frames:
                raise ValueError(
                    f"`temporal_ids` must have shape `(batch_size, {num_frames})`, got {tuple(frame_ids.shape)}."
                )
        return frame_ids.repeat_interleave(spatial_tokens_per_frame, dim=1)

    # ------------------------------------------------------------------
    # Unpatchify
    # ------------------------------------------------------------------

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        c = self.out_channels
        pt, ph, pw = self.patch_size
        if t * h * w != x.shape[1]:
            raise ValueError(f"Expected t*h*w ({t * h * w}) to equal x.shape[1] ({x.shape[1]})")

        x = x.reshape(x.shape[0], t, h, w, c, pt, ph, pw)
        # (B, T, H, W, C, Pt, Ph, Pw) -> (B, C, T, Pt, H, Ph, W, Pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        ref_video_latent: torch.Tensor | None = None,
        current_temporal_ids: torch.Tensor | None = None,
        cached_temporal_ids: torch.Tensor | None = None,
        kv_cache_mode: str | None = None,
        kv_cache_chunk_id: int | None = None,
        kv_cache_selected_chunk_ids: list[int] | None = None,
        kv_cache_pre_rope: bool = False,
        self_attn_input_mode: str | None = None,
        skip_text_stream: bool = False,
        return_dict: bool = True,
    ):
        """
        The [`JoyVideoEditTransformer3DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_channels, num_frames, height, width)`):
                The noisy input latent patchified with `patch_size`.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Denoising timestep, one scalar per batch element.
            encoder_hidden_states (`torch.Tensor`):
                Text conditioning embeddings.
            encoder_hidden_states_mask (`torch.Tensor`, *optional*):
                Boolean/int padding mask over `encoder_hidden_states` (`True`/non-zero keeps the token). When given, it
                is folded into the joint self-attention so padded text tokens contribute nothing to any query's softmax
                — required for correct batched inference with variable-length prompts.
            ref_video_latent (`torch.Tensor`, *optional*):
                A reference/edit-condition video latent, patchified with the same `img_in` and concatenated into the
                image stream (tagged with the `SOURCE_ID_EDIT_CONDITION` role) so it participates in joint
                self-attention as extra context, without being part of the denoised output.
            current_temporal_ids (`torch.Tensor`, *optional*, shape `(batch_size, num_frames)`):
                Explicit per-frame temporal ids for `hidden_states` (and, if provided, `ref_video_latent`). Falls back
                to `0..num_frames-1` when omitted.
            cached_temporal_ids (`torch.Tensor`, *optional*):
                Temporal ids of the tokens stored in the selected KV-cache chunks, used to rebuild RoPE for
                `pre_rope`-cached entries at read time.
            kv_cache_mode (`str`, *optional*): One of `"store"`, `"reuse"`, or `None`.
            kv_cache_chunk_id (`int`, *optional*): Identifier under which this call's image-stream KV is stored.
            kv_cache_selected_chunk_ids (`list[int]`, *optional*): Chunk ids to read cached KV from.
            kv_cache_pre_rope (`bool`, *optional*, defaults to `False`):
                If `True`, cached keys are stored before RoPE and re-rotated at read time using `cached_temporal_ids`.
            self_attn_input_mode (`str`, *optional*):
                Set to `SELF_ATTN_MODE_REF_IMAGE_CACHE` when prefilling the KV cache from a static reference image
                (tags the current tokens with the `SOURCE_ID_EXTRA_REF_IMAGE` role instead of `SOURCE_ID_TARGET`).
            skip_text_stream (`bool`, *optional*, defaults to `False`):
                If `True`, the text stream is not projected/updated and attention only runs over the image stream (+
                any cached KV) -- used for KV-cache prefill calls where only the image-stream cache is wanted.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to wrap the image-stream sample into a [`~models.modeling_outputs.Transformer2DModelOutput`].
                Pass `return_dict=False` to return both image and text streams.

        Returns:
            `(img, txt)` when `return_dict=False`; otherwise a [`~models.modeling_outputs.Transformer2DModelOutput`]
            containing `img`.
        """
        if kv_cache_mode not in (None, "store", "reuse"):
            raise ValueError(f"Unsupported cache mode: {kv_cache_mode!r}.")
        if kv_cache_mode == "store" and kv_cache_chunk_id is None:
            raise ValueError("A cache chunk id is required in store mode.")
        if self_attn_input_mode not in (None, SELF_ATTN_MODE_REF_IMAGE_CACHE):
            raise ValueError(f"Unsupported self-attention input mode: {self_attn_input_mode!r}.")

        self.configure_inference_kv_cache(
            chunk_id=kv_cache_chunk_id,
            selected_chunk_ids=kv_cache_selected_chunk_ids,
            pre_rope=kv_cache_pre_rope,
        )

        batch_size = hidden_states.shape[0]
        patch_size = tuple(self.patch_size)
        current_patch_shape = self._get_patch_shape(hidden_states, patch_size)
        current_seq_len = math.prod(current_patch_shape)
        device = hidden_states.device

        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(
                device=encoder_hidden_states.device, dtype=torch.bool
            )

        hidden_tokens = self.img_in(hidden_states).flatten(2).transpose(1, 2).contiguous()
        temporal_ids = None
        if current_temporal_ids is not None:
            current_temporal_ids = torch.as_tensor(current_temporal_ids, device=device, dtype=torch.long)
            if current_temporal_ids.shape != (batch_size, current_patch_shape[0]):
                raise ValueError(
                    f"`current_temporal_ids` must have shape {(batch_size, current_patch_shape[0])}, "
                    f"got {tuple(current_temporal_ids.shape)}."
                )
            temporal_ids = current_temporal_ids

        if self_attn_input_mode == SELF_ATTN_MODE_REF_IMAGE_CACHE:
            current_source_id = torch.full(
                (current_seq_len,), SOURCE_ID_EXTRA_REF_IMAGE, device=device, dtype=torch.float32
            )
        else:
            current_source_id = torch.full((current_seq_len,), SOURCE_ID_TARGET, device=device, dtype=torch.float32)
        current_frame_ids = self._get_token_frame_ids(current_patch_shape, device, temporal_ids=temporal_ids)
        current_rotary = self.get_rotary_pos_embed_from_ids(
            frame_ids=current_frame_ids,
            spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
        )

        latent_segments = [hidden_tokens]
        rotary_segments = [current_rotary]
        source_id_segments = [current_source_id]

        if ref_video_latent is not None:
            if ref_video_latent.shape[0] != batch_size:
                raise ValueError(
                    f"Ref video latent batch size {ref_video_latent.shape[0]} does not match hidden states batch "
                    f"size {batch_size}."
                )
            ref_video_patch_shape = self._get_patch_shape(ref_video_latent, patch_size)
            if ref_video_patch_shape[1:] != current_patch_shape[1:]:
                raise ValueError(
                    "Ref video latent spatial patch shape must match noisy latent spatial patch shape: "
                    f"{ref_video_patch_shape[1:]} != {current_patch_shape[1:]}."
                )
            ref_video_tokens = self.img_in(ref_video_latent).flatten(2).transpose(1, 2).contiguous()
            video_frame_ids = self._get_token_frame_ids(ref_video_patch_shape, device, temporal_ids=temporal_ids)
            latent_segments.append(ref_video_tokens)
            rotary_segments.append(
                self.get_rotary_pos_embed_from_ids(
                    frame_ids=video_frame_ids,
                    spatial_shape=(ref_video_patch_shape[1], ref_video_patch_shape[2]),
                )
            )
            source_id_segments.append(
                torch.full((ref_video_tokens.shape[1],), SOURCE_ID_EDIT_CONDITION, device=device, dtype=torch.float32)
            )

        img = torch.cat(latent_segments, dim=1)
        visual_source_id = torch.cat(source_id_segments, dim=0).unsqueeze(0)
        # `torch.gather` requires an explicit index row for every batch element.
        current_indices = torch.arange(current_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        vis_freqs_cis = (
            torch.cat([rotary[0] for rotary in rotary_segments], dim=1),
            torch.cat([rotary[1] for rotary in rotary_segments], dim=1),
        )

        head_dim = self.hidden_size // self.num_attention_heads
        cos_3d, sin_3d = vis_freqs_cis
        cos_role, sin_role = self.generate_source_id_rope(
            source_id=visual_source_id,
            head_dim=head_dim,
            device=cos_3d.device,
            dtype=cos_3d.dtype,
        )
        # Compose the 3D spatiotemporal rotation with the role rotation: rotating by (angle_3d + angle_role) is
        # equivalent to cos(a+b) = cos(a)cos(b) - sin(a)sin(b), sin(a+b) = sin(a)cos(b) + cos(a)sin(b).
        new_cos = cos_3d * cos_role - sin_3d * sin_role
        new_sin = sin_3d * cos_role + cos_3d * sin_role
        vis_freqs_cis = (new_cos, new_sin)

        _, vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        vec = vec.unflatten(-1, (NUM_MODULATION_CHUNKS, -1))

        cached_freqs_cis = None
        if kv_cache_pre_rope and cached_temporal_ids is not None:
            cached_ids_tensor = torch.as_tensor(cached_temporal_ids, device=device, dtype=torch.long)
            if cached_ids_tensor.ndim != 2 or cached_ids_tensor.shape[0] != batch_size:
                raise ValueError(
                    "Cached temporal ids must have shape (batch_size, num_cached_frames), got "
                    f"{tuple(cached_ids_tensor.shape)}."
                )
            cached_frame_ids = self._get_token_frame_ids(
                (cached_ids_tensor.shape[1], current_patch_shape[1], current_patch_shape[2]),
                device,
                temporal_ids=cached_ids_tensor,
            )
            cached_freqs_cis = self.get_rotary_pos_embed_from_ids(
                frame_ids=cached_frame_ids,
                spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
            )

        for layer_idx, block in enumerate(self.double_blocks):
            kv_cache_reader = self._read_layer_kv_cache if kv_cache_mode == "reuse" else None
            kv_cache_writer = self._write_layer_kv_cache if kv_cache_mode == "store" else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # `_gradient_checkpointing_func` only forwards positional args to `torch.utils.checkpoint.checkpoint`,
                # so every block argument (including the kv-cache callables) must be passed positionally here in the
                # same order as `JoyVideoEditTransformerBlock.forward`.
                img, txt = self._gradient_checkpointing_func(
                    block,
                    img,
                    txt,
                    vec,
                    vis_freqs_cis,
                    encoder_hidden_states_mask,
                    kv_cache_reader,
                    kv_cache_writer,
                    layer_idx,
                    skip_text_stream,
                    kv_cache_pre_rope,
                    cached_freqs_cis,
                )
            else:
                img, txt = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    temb=vec,
                    image_rotary_emb=vis_freqs_cis,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    kv_cache_reader=kv_cache_reader,
                    kv_cache_writer=kv_cache_writer,
                    layer_idx=layer_idx,
                    skip_text_stream=skip_text_stream,
                    kv_cache_pre_rope=kv_cache_pre_rope,
                    cached_freqs_cis=cached_freqs_cis,
                )

        img = self.proj_out(self.norm_out(img))

        gather_index = current_indices.unsqueeze(-1).expand(-1, -1, img.shape[-1])
        img = torch.gather(img, dim=1, index=gather_index)
        img = self.unpatchify(img, current_patch_shape[0], current_patch_shape[1], current_patch_shape[2])

        if not return_dict:
            return (img, txt)
        return Transformer2DModelOutput(sample=img)
