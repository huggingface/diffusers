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

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


def rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Apply 3D rotary embeddings with the temporal band offset by `start_frame`.

    Computes in float64/complex128 and returns float32, matching the reference `causal_rope_apply`.
    """
    num_heads, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, num_heads, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def reference_rope_freqs(freqs, num_slots, tokens_per_slot, ref_grid, device):
    """Rotary frequencies for the reference-image tokens.

    Reference slots sit at large *negative* temporal positions (one stride of `max(tokens_per_slot, 256)` frames per
    slot) so they never collide with the rolling video window's temporal ids.
    """
    patch_t, patch_h, patch_w = ref_grid
    freq_dim = freqs.shape[1]
    f_band = freq_dim - 2 * (freq_dim // 3)

    temporal_step = max(tokens_per_slot, 256)
    neg_temporal = torch.tensor(
        [-(num_slots - i) * temporal_step for i in range(num_slots)], dtype=torch.float64, device=device
    )
    t_freqs = torch.outer(
        neg_temporal,
        1.0 / torch.pow(10000, torch.arange(0, 2 * f_band, 2, device=device, dtype=torch.float64).div(2 * f_band)),
    )
    t_freqs = torch.polar(torch.ones_like(t_freqs), t_freqs)

    freqs_split = freqs.split([f_band, freq_dim // 3, freq_dim // 3], dim=1)
    h_freqs = freqs_split[1][:patch_h].to(device)
    w_freqs = freqs_split[2][:patch_w].to(device)
    ref_freqs = torch.cat(
        [
            t_freqs[:, None, None, None, :].expand(num_slots, patch_t, patch_h, patch_w, f_band),
            h_freqs[None, None, :, None, :].expand(num_slots, patch_t, patch_h, patch_w, freq_dim // 3),
            w_freqs[None, None, None, :, :].expand(num_slots, patch_t, patch_h, patch_w, freq_dim // 3),
        ],
        dim=-1,
    ).reshape(num_slots * tokens_per_slot, 1, -1)
    return ref_freqs.to(torch.complex64)


def reference_rope_apply(x, freqs):
    """Apply the reference-slot rotary embeddings (float32/complex64, matching `rope_apply_with_refimg`)."""
    x_out = torch.view_as_complex(x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs.to(x.device)).flatten(3)
    return x_out.to(x.dtype)


class ABotWorldLayerKVCache:
    """Rolling K/V cache for one self-attention layer plus the layer's one-shot cross-attention cache.

    `key_raw` holds pre-RoPE keys (reference-token slots first, then the rolling video window), `key_roped` the
    post-RoPE video keys used for attention (re-based periodically so temporal positions stay within the rotary table),
    `value` the values. Tensor format: `(batch_size, num_tokens, num_heads, head_dim)`.
    """

    def __init__(self, batch_size, num_tokens, num_heads, head_dim, device, dtype):
        self.key_raw = torch.zeros(batch_size, num_tokens, num_heads, head_dim, device=device, dtype=dtype)
        self.key_roped = torch.zeros_like(self.key_raw)
        self.value = torch.zeros_like(self.key_raw)
        self.global_end_index = 0
        self.local_end_index = 0
        self.rope_base_frame = 0
        self.cross_key: torch.Tensor | None = None
        self.cross_value: torch.Tensor | None = None

    def reset(self):
        self.global_end_index = 0
        self.local_end_index = 0
        self.rope_base_frame = 0
        self.cross_key = None
        self.cross_value = None


class ABotWorldKVCache:
    """Container holding one [`ABotWorldLayerKVCache`] per transformer layer.

    Args:
        num_layers: Number of transformer layers.
        batch_size: Batch size of the rollout.
        num_tokens: Cache length in tokens: `ref_token_len + local_attn_size * tokens_per_frame`.
        ref_token_len: Number of reference-image tokens pinned at the start of the cache (never evicted).
        num_heads / head_dim / device / dtype: K/V tensor layout.
    """

    def __init__(self, num_layers, batch_size, num_tokens, ref_token_len, num_heads, head_dim, device, dtype):
        self.layer_caches = [
            ABotWorldLayerKVCache(batch_size, num_tokens, num_heads, head_dim, device, dtype)
            for _ in range(num_layers)
        ]
        self.ref_token_len = ref_token_len

    def get(self, layer_idx: int) -> ABotWorldLayerKVCache:
        return self.layer_caches[layer_idx]

    def reset(self):
        for cache in self.layer_caches:
            cache.reset()


class ABotWorldSelfAttnProcessor:
    r"""
    Causal windowed self-attention over a rolling K/V cache.

    Each forward writes the new block's keys/values into the cache (evicting the oldest video tokens once the
    `local_attn_size`-frame window is full; reference tokens at the head of the cache are never evicted) and attends
    the new block's queries over `[reference tokens | visible video window]`. Temporal rotary positions use an absolute
    counter relative to `rope_base_frame`, re-based whenever positions approach the rotary table limit — the attention
    logits only depend on position differences within the window, so re-basing does not change them.

    On the first block of a stream (`current_start == 0`) the reference tokens ride along in `hidden_states` (and in
    the queries); their pre-RoPE keys/values are pinned into the cache prefix.
    """

    _attention_backend = None
    _parallel_config = None

    # keep roped temporal positions well below the rotary-table length before re-basing
    _REBASE_MAX_POS = 256

    def __call__(
        self,
        attn: "ABotWorldAttention",
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        grid_sizes: torch.Tensor,
        kv_cache: ABotWorldLayerKVCache,
        current_start: int,
        query_ref_token_len: int,
        ref_token_len: int,
        ref_rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(hidden_states))
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
        video_token_len = query.shape[1] - query_ref_token_len
        num_video_frames = video_token_len // frame_seqlen
        video_grid_sizes = grid_sizes.clone()
        video_grid_sizes[:, 0] = num_video_frames

        cache_size = kv_cache.key_raw.shape[1]
        cache_current_end = ref_token_len + current_start + video_token_len

        # a new stream reuses the cache tensors; reset the rope base with the indices
        if kv_cache.global_end_index == 0:
            kv_cache.rope_base_frame = 0

        # roll the window: evict the oldest video tokens (reference tokens are pinned at the head)
        if cache_current_end > kv_cache.global_end_index and video_token_len + kv_cache.local_end_index > cache_size:
            num_evicted = video_token_len + kv_cache.local_end_index - cache_size
            num_rolled = kv_cache.local_end_index - num_evicted - ref_token_len
            src = slice(ref_token_len + num_evicted, ref_token_len + num_evicted + num_rolled)
            dst = slice(ref_token_len, ref_token_len + num_rolled)
            kv_cache.key_raw[:, dst] = kv_cache.key_raw[:, src].clone()
            kv_cache.key_roped[:, dst] = kv_cache.key_roped[:, src].clone()
            kv_cache.value[:, dst] = kv_cache.value[:, src].clone()
            local_end_index = kv_cache.local_end_index + cache_current_end - kv_cache.global_end_index - num_evicted
        else:
            local_end_index = kv_cache.local_end_index + cache_current_end - kv_cache.global_end_index
        local_start_index = local_end_index - video_token_len

        if query_ref_token_len > 0:
            kv_cache.key_raw[:, :ref_token_len] = key[:, :query_ref_token_len]
            kv_cache.value[:, :ref_token_len] = value[:, :query_ref_token_len]
        kv_cache.key_raw[:, local_start_index:local_end_index] = key[:, query_ref_token_len:]
        kv_cache.value[:, local_start_index:local_end_index] = value[:, query_ref_token_len:]

        # the visible video window, frame-aligned
        max_attention_tokens = attn.local_attn_size * frame_seqlen
        recent_start = max(ref_token_len, local_end_index - max_attention_tokens)
        recent_start += (local_end_index - recent_start) % frame_seqlen
        visible_video_frames = (local_end_index - recent_start) // frame_seqlen

        # temporal rotary positions: absolute counter relative to the rope base, re-based before it
        # approaches the rotary table limit (logits depend only on position differences in the window)
        abs_frame_start = current_start // frame_seqlen
        new_start_pos = abs_frame_start - kv_cache.rope_base_frame
        rebase_limit = min(self._REBASE_MAX_POS, rotary_emb.shape[0] - attn.local_attn_size - num_video_frames)
        if visible_video_frames > 0 and (new_start_pos + num_video_frames > rebase_limit or new_start_pos < 0):
            kv_cache.rope_base_frame = abs_frame_start + num_video_frames - visible_video_frames
            window_grid = grid_sizes.clone()
            window_grid[:, 0] = visible_video_frames
            kv_cache.key_roped[:, recent_start:local_end_index] = rope_apply(
                kv_cache.key_raw[:, recent_start:local_end_index], window_grid, rotary_emb, start_frame=0
            ).type_as(value)
            new_start_pos = abs_frame_start - kv_cache.rope_base_frame
        else:
            kv_cache.key_roped[:, local_start_index:local_end_index] = rope_apply(
                key[:, query_ref_token_len:], video_grid_sizes, rotary_emb, start_frame=new_start_pos
            ).type_as(value)

        roped_query = rope_apply(
            query[:, query_ref_token_len:], video_grid_sizes, rotary_emb, start_frame=new_start_pos
        ).type_as(value)

        attn_key = kv_cache.key_roped[:, recent_start:local_end_index]
        attn_value = kv_cache.value[:, recent_start:local_end_index]
        if ref_token_len > 0:
            ref_key = reference_rope_apply(kv_cache.key_raw[:, :ref_token_len], ref_rotary_emb).type_as(value)
            attn_key = torch.cat([ref_key, attn_key], dim=1)
            attn_value = torch.cat([kv_cache.value[:, :ref_token_len], attn_value], dim=1)
            if query_ref_token_len > 0:
                ref_query = reference_rope_apply(query[:, :query_ref_token_len], ref_rotary_emb).type_as(value)
                roped_query = torch.cat([ref_query, roped_query], dim=1)

        hidden_states = dispatch_attention_fn(
            roped_query,
            attn_key,
            attn_value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        kv_cache.global_end_index = cache_current_end
        kv_cache.local_end_index = local_end_index

        hidden_states = attn.to_out[0](hidden_states.flatten(2, 3))
        return hidden_states


class ABotWorldCrossAttnProcessor:
    r"""
    Text cross-attention with per-stream K/V caching: the text keys/values are projected once on the first block of a
    stream and reused for every subsequent block.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "ABotWorldAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_cache: ABotWorldLayerKVCache,
    ) -> torch.Tensor:
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))

        if kv_cache.cross_key is None:
            kv_cache.cross_key = attn.norm_k(attn.to_k(encoder_hidden_states)).unflatten(2, (attn.heads, -1))
            kv_cache.cross_value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            kv_cache.cross_key,
            kv_cache.cross_value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = attn.to_out[0](hidden_states.flatten(2, 3))
        return hidden_states


class ABotWorldAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = ABotWorldSelfAttnProcessor
    _available_processors = [ABotWorldSelfAttnProcessor, ABotWorldCrossAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int,
        eps: float,
        local_attn_size: int | None = None,
        is_cross_attention: bool = False,
        processor=None,
    ):
        super().__init__()
        self.heads = heads
        self.local_attn_size = local_attn_size
        self.is_cross_attention = is_cross_attention

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True)])
        self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


class ABotWorldResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class ABotWorldActionAdapter(nn.Module):
    """Encodes the broadcast action planes to the patch-token grid.

    Input `(B, action_in_channels, F, H_pix, W_pix)`; PixelUnshuffle + a stride-2 conv bring the spatial dims to the
    latent patch grid (`H_pix / (downscale_factor * 2)`), producing `(B, dim, F, H_patch, W_patch)` — added directly
    onto the patch-embedded video tokens.
    """

    def __init__(self, in_channels: int, dim: int, downscale_factor: int):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.conv = nn.Conv2d(
            in_channels * downscale_factor * downscale_factor, dim, kernel_size=(2, 2), stride=(2, 2), padding=0
        )
        self.residual_blocks = nn.Sequential(ABotWorldResidualBlock(dim))

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * num_frames, channels, height, width)
        x = self.residual_blocks(self.conv(self.pixel_unshuffle(x)))
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        return x.permute(0, 2, 1, 3, 4)


class ABotWorldTimeTextEmbedding(nn.Module):
    def __init__(self, dim: int, time_freq_dim: int, text_embed_dim: int):
        super().__init__()
        self.time_freq_dim = time_freq_dim
        self.time_embedder = nn.Sequential(nn.Linear(time_freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.text_embedder = nn.Sequential(
            nn.Linear(text_embed_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

    def sinusoidal_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        # matches the reference `sinusoidal_embedding_1d`: half-dim sin/cos over a 10000 theta, float64
        half = self.time_freq_dim // 2
        timestep = timestep.type(torch.float64)
        sinusoid = torch.outer(
            timestep, torch.pow(10000, -torch.arange(half, device=timestep.device).to(timestep.dtype).div(half))
        )
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype):
        # timestep: any shape; returns (temb [N, dim], temb_proj [*timestep.shape, 6, dim])
        temb = self.time_embedder(self.sinusoidal_embedding(timestep.flatten()).to(dtype))
        temb_proj = self.time_proj(temb).unflatten(1, (6, -1)).unflatten(0, timestep.shape)
        return temb, temb_proj


class ABotWorldTransformerBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float, local_attn_size: int):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = ABotWorldAttention(dim, num_heads, eps, local_attn_size=local_attn_size)
        self.attn2 = ABotWorldAttention(
            dim, num_heads, eps, is_cross_attention=True, processor=ABotWorldCrossAttnProcessor()
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
        grid_sizes: torch.Tensor,
        kv_cache: ABotWorldLayerKVCache,
        current_start: int,
        query_ref_token_len: int,
        ref_token_len: int,
        ref_rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        # temb_proj is [B, F, 6, C] frame-level modulation, or [B, L, 6, C] token-level on the first
        # block of a stream (where reference tokens with zero-timestep modulation ride along)
        token_level = temb_proj.shape[1] == hidden_states.shape[1]
        if token_level:
            num_frames, frame_seqlen = hidden_states.shape[1], 1
        else:
            num_frames, frame_seqlen = temb_proj.shape[1], hidden_states.shape[1] // temb_proj.shape[1]

        e = (self.scale_shift_table.unsqueeze(0).float() + temb_proj.float()).chunk(6, dim=2)

        def modulate(normed, shift, scale):
            if token_level:
                return normed * (1 + scale.squeeze(2)) + shift.squeeze(2)
            return ((normed.unflatten(1, (num_frames, frame_seqlen)) * (1 + scale)) + shift).flatten(1, 2)

        def gate(value, g):
            if token_level:
                return value * g.squeeze(2)
            return (value.unflatten(1, (num_frames, frame_seqlen)) * g).flatten(1, 2)

        norm_hidden_states = modulate(self.norm1(hidden_states.float()), e[0], e[1]).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
            grid_sizes=grid_sizes,
            kv_cache=kv_cache,
            current_start=current_start,
            query_ref_token_len=query_ref_token_len,
            ref_token_len=ref_token_len,
            ref_rotary_emb=ref_rotary_emb,
        )
        hidden_states = (hidden_states.float() + gate(attn_output.float(), e[2])).type_as(hidden_states)

        attn_output = self.attn2(
            self.norm2(hidden_states.float()).type_as(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = modulate(self.norm3(hidden_states.float()), e[3], e[4]).type_as(hidden_states)
        ffn_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + gate(ffn_output.float(), e[5])).type_as(hidden_states)
        return hidden_states


class ABotWorldTransformer3DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin):
    r"""
    The causal, action-conditioned video transformer from [ABot-World](https://github.com/amap-cvlab/ABot-World), a
    Wan2.2-TI2V-5B finetune for real-time interactive world generation.

    The model denoises one block of latent frames at a time: self-attention is windowed over the last `local_attn_size`
    frames through a rolling K/V cache ([`ABotWorldKVCache`]), reference-image tokens are pinned at the head of the
    cache, and keyboard-action planes are injected through a learned adapter added onto the patch tokens. Timesteps are
    per latent frame (`(batch, frames)`).
    """

    _repeated_blocks = ["ABotWorldTransformerBlock"]
    _no_split_modules = ["ABotWorldTransformerBlock"]
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _skip_keys = ["kv_cache"]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int = 48,
        text_dim: int = 4096,
        text_len: int = 512,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        eps: float = 1e-6,
        local_attn_size: int = 21,
        action_in_channels: int = 32,
        action_downscale_factor: int = 16,
        rope_max_seq_len: int = 1024,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.act_control_adapter = ABotWorldActionAdapter(action_in_channels, inner_dim, action_downscale_factor)
        self.condition_embedder = ABotWorldTimeTextEmbedding(inner_dim, freq_dim, text_dim)

        self.blocks = nn.ModuleList(
            [
                ABotWorldTransformerBlock(inner_dim, ffn_dim, num_attention_heads, eps, local_attn_size)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # kept as a plain float64/complex128 attribute (not a buffer) so `model.to(dtype)` never downcasts it;
        # moved to the execution device in `forward`, matching the reference
        head_dim = attention_head_dim
        self.rotary_freqs = torch.cat(
            [
                rope_params(rope_max_seq_len, head_dim - 4 * (head_dim // 6)),
                rope_params(rope_max_seq_len, 2 * (head_dim // 6)),
                rope_params(rope_max_seq_len, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor | None = None,
        action_scale: float = 1.0,
        reference_hidden_states: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        kv_cache: ABotWorldKVCache = None,
        current_start: int = 0,
        return_dict: bool = True,
    ):
        r"""
        Args:
            hidden_states: Noisy latents for one block, `(batch, in_channels, frames, height, width)`.
            timestep: Per-latent-frame timesteps, `(batch, frames)`.
            encoder_hidden_states: Text embeddings `(batch, seq_len, text_dim)`; zero-padded to `text_len` inside.
            action_hidden_states: Broadcast action planes `(batch, action_in_channels, frames, pixel_h, pixel_w)`.
            reference_hidden_states: Reference-image latents `(batch, num_slots, in_channels, 1, ref_h, ref_w)`.
                Only consumed on the first block of a stream (`current_start == 0`), where the reference tokens are
                pinned into the K/V cache; later blocks attend to them from the cache.
            reference_mask: Per-slot validity mask `(batch, num_slots)`.
            kv_cache: The stream's [`ABotWorldKVCache`], allocated with room for the reference tokens.
            current_start: Token offset of this block in the rollout: `start_frame * tokens_per_frame`.
        """
        if kv_cache is None:
            raise ValueError("`kv_cache` is required: this model only runs block-causal rollout with a K/V cache.")
        if self.rotary_freqs.device != hidden_states.device:
            self.rotary_freqs = self.rotary_freqs.to(hidden_states.device)
        freqs = self.rotary_freqs

        # patchify and add the action adapter's features onto the patch tokens
        hidden_states = self.patch_embedding(hidden_states)
        if action_hidden_states is not None:
            action_features = self.act_control_adapter(action_hidden_states)
            frames = hidden_states.shape[2]
            action_frames = action_features.shape[2]
            if frames > action_frames:
                offset = frames - action_frames
                hidden_states = torch.cat(
                    [hidden_states[:, :, :offset], hidden_states[:, :, offset:] + action_features * action_scale],
                    dim=2,
                )
            else:
                hidden_states = hidden_states + action_features * action_scale

        grid_sizes = torch.tensor(hidden_states.shape[2:], dtype=torch.long).unsqueeze(0)
        batch_size = hidden_states.shape[0]
        frame_seqlen = int(math.prod(hidden_states.shape[3:]))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, temb_proj = self.condition_embedder(timestep, hidden_states.dtype)

        # zero-pad the text embeddings to text_len, matching the reference
        seq = encoder_hidden_states.shape[1]
        if seq < self.config.text_len:
            encoder_hidden_states = torch.cat(
                [
                    encoder_hidden_states,
                    encoder_hidden_states.new_zeros(
                        batch_size, self.config.text_len - seq, encoder_hidden_states.shape[2]
                    ),
                ],
                dim=1,
            )
        encoder_hidden_states = self.condition_embedder.text_embedder(encoder_hidden_states)

        # reference tokens ride along on the first block of a stream and are pinned into the cache
        ref_token_len = kv_cache.ref_token_len
        query_ref_token_len = 0
        ref_rotary_emb = None
        if reference_hidden_states is not None and current_start == 0:
            batch, num_slots, channels, ref_t, ref_h, ref_w = reference_hidden_states.shape
            ref_features = self.patch_embedding(
                reference_hidden_states.reshape(batch * num_slots, channels, ref_t, ref_h, ref_w).to(
                    hidden_states.dtype
                )
            )
            patch_t, patch_h, patch_w = ref_features.shape[2:]
            tokens_per_slot = patch_t * patch_h * patch_w
            ref_tokens = ref_features.flatten(2).transpose(1, 2).reshape(batch, num_slots, tokens_per_slot, -1)
            if reference_mask is None:
                reference_mask = reference_hidden_states.new_ones(batch, num_slots)
            ref_tokens = ref_tokens * reference_mask[:, :, None, None].to(ref_tokens.dtype)
            ref_tokens = ref_tokens.reshape(batch, num_slots * tokens_per_slot, -1)

            query_ref_token_len = num_slots * tokens_per_slot
            if query_ref_token_len != ref_token_len:
                raise ValueError(
                    f"The KV cache was allocated for {ref_token_len} reference tokens but "
                    f"`reference_hidden_states` produced {query_ref_token_len}."
                )
            self._ref_grid = (int(patch_t), int(patch_h), int(patch_w))
            self._ref_num_slots = int(num_slots)
            self._ref_tokens_per_slot = int(tokens_per_slot)

            # token-level modulation: video tokens keep their frame's modulation, reference tokens get timestep 0
            temb_proj = temb_proj.repeat_interleave(frame_seqlen, dim=1)
            ref_timestep = torch.zeros((batch_size, 1), dtype=torch.long, device=hidden_states.device)
            _, ref_temb_proj = self.condition_embedder(ref_timestep, hidden_states.dtype)
            temb_proj = torch.cat([ref_temb_proj.expand(-1, query_ref_token_len, -1, -1), temb_proj], dim=1)
            hidden_states = torch.cat([ref_tokens, hidden_states], dim=1)

        if ref_token_len > 0:
            ref_rotary_emb = reference_rope_freqs(
                freqs, self._ref_num_slots, self._ref_tokens_per_slot, self._ref_grid, hidden_states.device
            )

        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb_proj,
                freqs,
                grid_sizes,
                kv_cache.get(layer_idx),
                current_start,
                query_ref_token_len,
                ref_token_len,
                ref_rotary_emb,
            )

        if query_ref_token_len > 0:
            hidden_states = hidden_states[:, query_ref_token_len:]

        # head: frame-level modulation with the un-projected time embedding
        shift, scale = (
            self.scale_shift_table.unsqueeze(1).float() + temb.unflatten(0, timestep.shape).unsqueeze(2).float()
        ).chunk(2, dim=2)
        num_frames = timestep.shape[1]
        hidden_states = (
            self.norm_out(hidden_states.float()).unflatten(1, (num_frames, frame_seqlen)) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify: (B, F, frame_seqlen, prod(patch) * C) -> (B, C, F*pt, H*ph, W*pw)
        p_t, p_h, p_w = self.config.patch_size
        _, latent_h, latent_w = grid_sizes[0].tolist()
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, latent_h, latent_w, p_t, p_h, p_w, self.config.out_channels
        )
        hidden_states = torch.einsum("bfhwpqrc->bcfphqwr", hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, self.config.out_channels, num_frames * p_t, latent_h * p_h, latent_w * p_w
        )

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)
