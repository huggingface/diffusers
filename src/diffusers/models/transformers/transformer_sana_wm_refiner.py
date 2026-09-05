# Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import apply_lora_scale, logging
from ..attention import AttentionMixin
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from .transformer_ltx2 import (
    LTX2AdaLayerNormSingle,
    LTX2AudioVideoRotaryPosEmbed,
    LTX2VideoTransformerBlock,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ``kv_cache_mode`` values accepted by [`SanaWMLTX2RefinerTransformer3DModel`] and
# [`SanaWMLTX2RefinerTransformerBlock`]. See [`SanaWMRefinerKVCache`] for the AR contract they implement.
KV_CACHE_MODE_INJECT = "inject"
KV_CACHE_MODE_CAPTURE_PRE_ROPE = "capture_pre_rope"
KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE = "inject_and_capture_post_rope"

_KV_CACHE_MODES = (
    KV_CACHE_MODE_INJECT,
    KV_CACHE_MODE_CAPTURE_PRE_ROPE,
    KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE,
)
_KV_CACHE_INJECT_MODES = (KV_CACHE_MODE_INJECT, KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE)


class SanaWMRefinerKVLayerCache:
    r"""
    Per-layer KV cache for the SANA-WM stage-2 chunk-causal AR refiner.

    Holds the two halves of the sliding-window prefix that the refiner's self-attention attends to, plus a slot for
    reading back the K/V that the last forward captured. All tensors are `(batch_size, num_tokens, inner_dim)` (i.e.
    before the per-head unflatten), matching the layout the refiner's self-attention concatenates in.

    * ``sink_k_pre`` / ``sink_v``: **pre**-RoPE K/V of the attention-sink frames, captured once from the raw stage-1
      latents. They are stored pre-RoPE so each AR window can re-apply RoPE at its own shifted sink offset
      (``SanaWMRefinerKVCache.sink_pe``).
    * ``history_k`` / ``history_v``: **post**-RoPE K/V of the already refined recent frames, ready to be concatenated
      as-is.
    * ``captured_k_pre`` / ``captured_v_pre`` and ``captured_k_post`` / ``captured_v_post``: readback slots written by
      the capture ``kv_cache_mode``s.
    """

    def __init__(self):
        self.sink_k_pre: torch.Tensor | None = None
        self.sink_v: torch.Tensor | None = None
        self.history_k: torch.Tensor | None = None
        self.history_v: torch.Tensor | None = None
        self.captured_k_pre: torch.Tensor | None = None
        self.captured_v_pre: torch.Tensor | None = None
        self.captured_k_post: torch.Tensor | None = None
        self.captured_v_post: torch.Tensor | None = None

    def store_sink(self, sink_k_pre: torch.Tensor, sink_v: torch.Tensor) -> None:
        """Store the pre-RoPE sink K/V."""
        self.sink_k_pre = sink_k_pre
        self.sink_v = sink_v

    def get_sink(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the pre-RoPE sink K/V, or `None` if it has not been captured (or is empty)."""
        if self.sink_k_pre is None or self.sink_v is None or self.sink_k_pre.shape[1] == 0:
            return None
        return self.sink_k_pre, self.sink_v

    def store_history(self, history_k: torch.Tensor, history_v: torch.Tensor) -> None:
        """Store the post-RoPE recent-history K/V."""
        self.history_k = history_k
        self.history_v = history_v

    def get_history(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the post-RoPE recent-history K/V, or `None` if empty."""
        if self.history_k is None or self.history_v is None or self.history_k.shape[1] == 0:
            return None
        return self.history_k, self.history_v

    def store_captured_pre_rope(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store the pre-RoPE K/V produced by the current forward."""
        self.captured_k_pre = key
        self.captured_v_pre = value

    def get_captured_pre_rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pop the pre-RoPE K/V captured by the last forward."""
        if self.captured_k_pre is None:
            raise RuntimeError("No pre-RoPE K/V was captured. Run a forward with `kv_cache_mode='capture_pre_rope'`.")
        key, value = self.captured_k_pre, self.captured_v_pre
        # Release the references so the caller owns the only handle.
        self.captured_k_pre = self.captured_v_pre = None
        return key, value

    def store_captured_post_rope(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store the post-RoPE K/V produced by the current forward."""
        self.captured_k_post = key
        self.captured_v_post = value

    def get_captured_post_rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pop the post-RoPE K/V captured by the last forward."""
        if self.captured_k_post is None:
            raise RuntimeError(
                "No post-RoPE K/V was captured. Run a forward with `kv_cache_mode='inject_and_capture_post_rope'`."
            )
        key, value = self.captured_k_post, self.captured_v_post
        self.captured_k_post = self.captured_v_post = None
        return key, value

    def clear(self) -> None:
        self.sink_k_pre = None
        self.sink_v = None
        self.history_k = None
        self.history_v = None
        self.captured_k_pre = None
        self.captured_v_pre = None
        self.captured_k_post = None
        self.captured_v_post = None


class SanaWMRefinerKVCache:
    r"""
    Container holding one [`SanaWMRefinerKVLayerCache`] per transformer block, plus the shared sink RoPE.

    This implements the ``rf_shifted_sink`` KV-cache contract the SANA-WM stage-2 refiner was trained with. Refinement
    is chunk-causal: `block_size` latent frames are denoised at a time while attending to a bounded window of
    `[attention sink + recent history + active block]` K/V.

    * ``sink_pe``: the `(cos, sin)` RoPE tuple for the sink frames, rebuilt per AR window at the sliding
      ``sink_rope_offset`` so the sink sits immediately before the bounded working cache. Shared across layers because
      RoPE does not depend on the layer.

    Args:
        num_layers (`int`):
            Number of transformer blocks to allocate a per-layer cache for.
    """

    def __init__(self, num_layers: int):
        self.layer_caches = [SanaWMRefinerKVLayerCache() for _ in range(num_layers)]
        self.sink_pe: tuple[torch.Tensor, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.layer_caches)

    def get(self, layer_idx: int) -> SanaWMRefinerKVLayerCache:
        return self.layer_caches[layer_idx]

    def clear(self) -> None:
        for layer_cache in self.layer_caches:
            layer_cache.clear()
        self.sink_pe = None


class SanaWMLTX2RefinerTransformerBlock(LTX2VideoTransformerBlock):
    r"""
    Video-only, streaming-attention variant of [`LTX2VideoTransformerBlock`] used by the SANA-WM stage-2 refiner.

    The submodule structure is inherited unchanged from [`LTX2VideoTransformerBlock`] (so LTX-2 checkpoints load
    as-is); only [`~SanaWMLTX2RefinerTransformerBlock.forward`] is overridden. It runs the video stream only (self-attn
    -> prompt cross-attn -> feed-forward), skipping the audio and audio/video cross-attention branches, and routes the
    self-attention through a KV-cached sliding window instead of plain full self-attention.
    """

    def _streaming_self_attention(
        self,
        hidden_states: torch.Tensor,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        sink_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: SanaWMRefinerKVLayerCache | None = None,
        kv_cache_mode: str | None = None,
    ) -> torch.Tensor:
        """LTX-2 self-attention over `[sink + history + current]` K/V.

        The queries always come from the active block only. Depending on `kv_cache_mode`, the layer cache's pre-RoPE
        sink K/V (re-RoPE'd here with `sink_rotary_emb`) and post-RoPE recent-history K/V are prepended to the current
        K/V before a single SDPA call, and/or the current K/V is written back to the cache.
        """
        attn = self.attn1

        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Capture PRE-RoPE (post-norm) K/V so a future window can re-apply RoPE at its shifted sink offset.
        if kv_cache_mode == KV_CACHE_MODE_CAPTURE_PRE_ROPE:
            kv_cache.store_captured_pre_rope(key.detach().clone(), value.detach().clone())

        if attn.rope_type == "interleaved":
            query = apply_interleaved_rotary_emb(query, query_rotary_emb)
            key = apply_interleaved_rotary_emb(key, query_rotary_emb)
        elif attn.rope_type == "split":
            query = apply_split_rotary_emb(query, query_rotary_emb)
            key = apply_split_rotary_emb(key, query_rotary_emb)
        else:
            raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

        # Capture POST-RoPE K/V so the next window can concatenate the recent history directly. Deliberately taken
        # before the prefix is prepended, so only the current block's tokens are recorded.
        if kv_cache_mode == KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE:
            kv_cache.store_captured_post_rope(key.detach().clone(), value.detach().clone())

        if kv_cache_mode in _KV_CACHE_INJECT_MODES:
            prefix_k_parts: list[torch.Tensor] = []
            prefix_v_parts: list[torch.Tensor] = []
            sink_kv = kv_cache.get_sink()
            if sink_kv is not None:
                if sink_rotary_emb is None:
                    raise ValueError("Injecting the attention sink requires the `sink_pe` RoPE tuple on the KV cache.")
                sink_k_pre, sink_v = sink_kv
                sink_k_pre = sink_k_pre.to(key.dtype)
                if attn.rope_type == "interleaved":
                    sink_k = apply_interleaved_rotary_emb(sink_k_pre, sink_rotary_emb)
                else:
                    sink_k = apply_split_rotary_emb(sink_k_pre, sink_rotary_emb)
                prefix_k_parts.append(sink_k)
                prefix_v_parts.append(sink_v.to(value.dtype))
            history_kv = kv_cache.get_history()
            if history_kv is not None:
                prefix_k_parts.append(history_kv[0].to(key.dtype))
                prefix_v_parts.append(history_kv[1].to(value.dtype))
            if prefix_k_parts:
                key = torch.cat([*prefix_k_parts, key], dim=1)
                value = torch.cat([*prefix_v_parts, value], dim=1)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        processor = attn.processor
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(processor, "_attention_backend", None),
            parallel_config=getattr(processor, "_parallel_config", None),
        )

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
        sink_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: SanaWMRefinerKVLayerCache | None = None,
        kv_cache_mode: str | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)

        # 1. Video self-attention over the KV-cached sliding window
        norm_hidden_states = self.norm1(hidden_states)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod_params(
            self.scale_shift_table, temb, batch_size
        )
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self._streaming_self_attention(
            norm_hidden_states,
            query_rotary_emb=video_rotary_emb,
            sink_rotary_emb=sink_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        # 2. Prompt cross-attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp
        return hidden_states


class SanaWMLTX2RefinerTransformer3DModel(
    ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    r"""
    The chunk-causal autoregressive refiner transformer used as SANA-WM stage 2.

    Architecturally identical to [`LTX2VideoTransformer3DModel`] — same config arguments, same submodules, same
    parameter names — so a released LTX-2 checkpoint loads into it unchanged. What differs is the forward pass:

    * only the video stream is run (the audio and audio/video cross-attention branches are skipped),
    * self-attention runs against an explicit sliding-window KV cache ([`SanaWMRefinerKVCache`]) holding the attention
      sink plus recent refined history, so refinement cost is bounded per AR block and scales linearly with video
      length,
    * the caller supplies the video RoPE, which lets each AR window keep every frame's absolute index in the source
      video (see
      [`~SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions`]).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the temporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `4096`):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `48`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
        rope_type (`str`, defaults to `"interleaved"`):
            Which RoPE application to use (`"interleaved"` or `"split"`).

    The remaining arguments mirror [`LTX2VideoTransformer3DModel`] one-for-one. The audio-side arguments and submodules
    are kept purely so the checkpoint's audio weights round-trip; they are not used by the refiner forward.
    """

    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["SanaWMLTX2RefinerTransformerBlock"]
    _skip_keys = ["kv_cache"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,  # Video Arguments
        out_channels: int | None = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        gated_attn: bool = False,
        cross_attn_mod: bool = False,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: int | None = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        audio_gated_attn: bool = False,
        audio_cross_attn_mod: bool = False,
        num_layers: int = 48,  # Shared arguments
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        use_prompt_embeddings=True,
        perturbed_attn: bool = False,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        use_prompt_adaln_single: bool = True,
        use_keyframes_abs_pos_embedding: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        audio_out_channels = audio_out_channels or audio_in_channels
        inner_dim = num_attention_heads * attention_head_dim
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)

        if use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = nn.Parameter(torch.zeros(1, inner_dim))

        # 2. Prompt embeddings
        if use_prompt_embeddings:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=audio_inner_dim
            )

        # 3. Timestep Modulation Params and Embedding
        self.prompt_modulation = cross_attn_mod or audio_cross_attn_mod

        # 3.1. Global Timestep Modulation Parameters (except for cross-attention) and timestep + size embedding
        video_time_emb_mod_params = 9 if cross_attn_mod else 6
        audio_time_emb_mod_params = 9 if audio_cross_attn_mod else 6
        self.time_embed = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=video_time_emb_mod_params, use_additional_conditions=False
        )
        self.audio_time_embed = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=audio_time_emb_mod_params, use_additional_conditions=False
        )

        # 3.2. Global Cross Attention Modulation Parameters
        self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=1, use_additional_conditions=False
        )
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=1, use_additional_conditions=False
        )

        # 3.3. Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)

        # 3.4. Prompt Scale/Shift Modulation parameters (LTX-2.3)
        if self.prompt_modulation and use_prompt_adaln_single:
            self.prompt_adaln = LTX2AdaLayerNormSingle(inner_dim, num_mod_params=2, use_additional_conditions=False)
            self.audio_prompt_adaln = LTX2AdaLayerNormSingle(
                audio_inner_dim, num_mod_params=2, use_additional_conditions=False
            )

        # 4. Rotary Positional Embeddings (RoPE)
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            scale_factors=[audio_scale_factor],
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # Audio-to-Video, Video-to-Audio Cross-Attention
        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SanaWMLTX2RefinerTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=audio_num_attention_heads,
                    audio_attention_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    video_gated_attn=gated_attn,
                    video_cross_attn_adaln=cross_attn_mod,
                    audio_gated_attn=audio_gated_attn,
                    audio_cross_attn_adaln=audio_cross_attn_mod,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                    perturbed_attn=perturbed_attn,
                    ff_bias=ff_bias,
                    audio_ff_bias=audio_ff_bias,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

        self.gradient_checkpointing = False

    def build_rotary_emb_for_absolute_positions(
        self,
        batch_size: int,
        frame_positions: list[int],
        height: int,
        width: int,
        device: torch.device,
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build the video RoPE for an explicit list of absolute latent-frame indices.

        [`LTX2AudioVideoRotaryPosEmbed.prepare_video_coords`] assumes a contiguous `torch.arange(num_frames)`, which is
        fine for bidirectional inference. The sliding-window AR refiner instead needs to keep each frame's absolute
        index in the source video, so RoPE captures the correct temporal phase across the
        `[sink + recent + active]` window.

        Args:
            batch_size (`int`):
                Batch size to broadcast the coordinates to.
            frame_positions (`list[int]`):
                Absolute latent-frame indices covered by this window.
            height (`int`), width (`int`):
                Latent spatial resolution.
            device (`torch.device`):
                Device to build the coordinates on.
            fps (`float`):
                Video frame rate, which drives LTX-2's temporal RoPE scaling.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(cos, sin)` RoPE tuple.
        """
        rope = self.rope
        patch_size_t = int(rope.patch_size_t)
        patch_size = int(rope.patch_size)
        f_positions = torch.tensor(frame_positions, dtype=torch.float32, device=device)
        if patch_size_t > 1:
            # Each patch covers ``patch_size_t`` latent frames; pick the start of each patch.
            f_positions = f_positions[::patch_size_t]
        grid_h = torch.arange(start=0, end=height, step=patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=width, step=patch_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(f_positions, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)
        latent_coords = torch.stack([grid, patch_ends], dim=-1)
        latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        scale_tensor = torch.tensor(rope.scale_factors, device=device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
        return rope(pixel_coords, device=device)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
        kv_cache: SanaWMRefinerKVCache | None = None,
        kv_cache_mode: str | None = None,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ):
        r"""
        Video-only forward pass over a single AR block.

        Args:
            hidden_states (`torch.Tensor`):
                Patchified video latents of the active block, of shape `(batch_size, num_video_tokens, in_channels)`.
            encoder_hidden_states (`torch.Tensor`):
                Text embeddings of shape `(batch_size, text_seq_len, caption_channels)`.
            timestep (`torch.Tensor`):
                Timestep of shape `(batch_size, num_video_tokens)`, already scaled by
                `self.config.timestep_scale_multiplier`.
            video_rotary_emb (`tuple[torch.Tensor, torch.Tensor]`):
                The `(cos, sin)` RoPE for the active block's absolute frame positions, as returned by
                [`~SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions`].
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Multiplicative text attention mask of shape `(batch_size, text_seq_len)`.
            kv_cache (`SanaWMRefinerKVCache`, *optional*):
                Sliding-window KV cache holding the per-layer attention sink and recent refined history, plus the
                shared `sink_pe` RoPE. Required whenever `kv_cache_mode` is set.
            kv_cache_mode (`str`, *optional*):
                One of:

                - `"inject"`: attend to `[sink + history + current]` K/V (the denoising steps).
                - `"capture_pre_rope"`: no prefix; record the pre-RoPE K/V of this forward into the cache (used once to
                  seed the attention sink from the raw stage-1 latents).
                - `"inject_and_capture_post_rope"`: attend to `[sink + history + current]` K/V and record this block's
                  post-RoPE K/V into the cache so it can be appended to the history.

                When `None`, the block runs plain full self-attention over the current tokens only.
            attention_kwargs (`dict`, *optional*):
                Optional kwargs forwarded to the LoRA scale handling.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`: the predicted velocity for the active
            block, of shape `(batch_size, num_video_tokens, out_channels)`.
        """
        if kv_cache_mode is not None:
            if kv_cache_mode not in _KV_CACHE_MODES:
                raise ValueError(f"`kv_cache_mode` must be one of {_KV_CACHE_MODES} or `None`, got {kv_cache_mode!r}.")
            if kv_cache is None:
                raise ValueError(f"`kv_cache_mode={kv_cache_mode!r}` requires a `SanaWMRefinerKVCache`.")
            if len(kv_cache) != len(self.transformer_blocks):
                raise ValueError(
                    f"`kv_cache` holds {len(kv_cache)} layer caches but the model has "
                    f"{len(self.transformer_blocks)} transformer blocks."
                )

        batch_size = hidden_states.size(0)

        # Convert encoder_attention_mask to an additive bias.
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Patchification input projection
        hidden_states = self.proj_in(hidden_states)

        # 2. Timestep embedding and modulation parameters
        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        # 3. Prompt embeddings
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        # 4. Transformer blocks
        sink_rotary_emb = kv_cache.sink_pe if kv_cache is not None else None
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                sink_rotary_emb=sink_rotary_emb,
                kv_cache=kv_cache.get(i) if kv_cache is not None else None,
                kv_cache_mode=kv_cache_mode,
            )

        # 5. Output norm and projection
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
