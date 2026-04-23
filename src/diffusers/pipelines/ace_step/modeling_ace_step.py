# Copyright 2025 The ACE-Step Team and The HuggingFace Team. All rights reserved.
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

"""Pipeline-specific models for ACE-Step 1.5.

Holds the condition encoder (lyric + timbre + text packing), the encoder layer
(``AceStepEncoderLayer`` — not used by the DiT itself, hence kept here), and the
``_pack_sequences`` helper. The DiT uses the RoPE helper, ``AceStepAttention``,
and ``_create_4d_mask`` from ``diffusers/models/transformers/ace_step_transformer.py``.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ...models.normalization import RMSNorm
from ...models.transformers.ace_step_transformer import (
    AceStepAttention,
    AceStepMLP,
    _ace_step_rotary_freqs,
    _create_4d_mask,
)
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# --------------------------------------------------------------------------- #
#                        helpers used only by condition encoder                #
# --------------------------------------------------------------------------- #


def _pack_sequences(
    hidden1: torch.Tensor, hidden2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack two masked sequences into one with all valid tokens first.

    Concatenates ``hidden1`` + ``hidden2`` along the sequence dim, then stably sorts
    each batch so mask=1 tokens come before mask=0 tokens. Returns the packed
    hidden states plus a fresh contiguous mask.
    """
    hidden_cat = torch.cat([hidden1, hidden2], dim=1)
    mask_cat = torch.cat([mask1, mask2], dim=1)

    B, L, D = hidden_cat.shape
    sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)
    hidden_left = torch.gather(hidden_cat, 1, sort_idx.unsqueeze(-1).expand(B, L, D))
    lengths = mask_cat.sum(dim=1)
    new_mask = torch.arange(L, dtype=torch.long, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(1)
    return hidden_left, new_mask


class AceStepEncoderLayer(nn.Module):
    """Pre-LN transformer block used by the lyric and timbre encoders.

    Kept inside the pipeline package because the main DiT does not consume it
    (reviewer comment on PR #13095).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.self_attn = AceStepAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            bias=attention_bias,
            dropout=attention_dropout,
            eps=rms_norm_eps,
            is_cross_attention=False,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = AceStepMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            image_rotary_emb=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _run_encoder_layers(
    layers: nn.ModuleList,
    layer_types: list,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    full_attn_mask: torch.Tensor,
    sliding_attn_mask: Optional[torch.Tensor],
    gradient_checkpointing: bool,
    gradient_checkpointing_func,
) -> torch.Tensor:
    """Shared encoder-layer loop used by the lyric + timbre encoders.

    Picks the per-layer mask based on ``layer_types`` and optionally wraps each
    call in ``gradient_checkpointing_func`` (matches the original ACE-Step
    encoders that enable gradient checkpointing during training — reviewer
    comment on PR #13095).
    """
    for i, layer_module in enumerate(layers):
        mask = (
            sliding_attn_mask
            if layer_types[i] == "sliding_attention" and sliding_attn_mask is not None
            else full_attn_mask
        )
        if gradient_checkpointing and torch.is_grad_enabled():
            hidden_states = gradient_checkpointing_func(layer_module, hidden_states, position_embeddings, mask)
        else:
            hidden_states = layer_module(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )
    return hidden_states


# --------------------------------------------------------------------------- #
#                                  encoders                                    #
# --------------------------------------------------------------------------- #


class AceStepLyricEncoder(ModelMixin, ConfigMixin):
    """Lyric encoder: projects Qwen3 lyric embeddings and runs a small transformer.

    Output feeds the DiT cross-attention (after packing with text + timbre).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        text_hidden_dim: int = 1024,
        num_lyric_encoder_hidden_layers: int = 8,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 128,
        layer_types: list = None,
    ):
        super().__init__()

        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(num_lyric_encoder_hidden_layers)
            ]

        self.embed_tokens = nn.Linear(text_hidden_dim, hidden_size)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.layers = nn.ModuleList(
            [
                AceStepEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=sliding_window if layer_types[i] == "sliding_attention" else None,
                )
                for i in range(num_lyric_encoder_hidden_layers)
            ]
        )

        self._layer_types = layer_types
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(inputs_embeds)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        cos, sin = _ace_step_rotary_freqs(seq_len, self.head_dim, self.rope_theta, device, dtype)
        position_embeddings = (cos, sin)

        full_attn_mask = _create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device, attention_mask=attention_mask, is_causal=False
        )
        sliding_attn_mask = _create_4d_mask(
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            attention_mask=attention_mask,
            sliding_window=self.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )

        hidden_states = _run_encoder_layers(
            self.layers,
            self._layer_types,
            inputs_embeds,
            position_embeddings,
            full_attn_mask,
            sliding_attn_mask,
            self.gradient_checkpointing,
            self._gradient_checkpointing_func if hasattr(self, "_gradient_checkpointing_func") else None,
        )
        return self.norm(hidden_states)


class AceStepTimbreEncoder(ModelMixin, ConfigMixin):
    """Timbre encoder: consumes VAE-encoded reference-audio latents and returns a
    pooled per-batch timbre embedding (plus a presence mask).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        timbre_hidden_dim: int = 64,
        num_timbre_encoder_hidden_layers: int = 4,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 128,
        layer_types: list = None,
    ):
        super().__init__()

        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(num_timbre_encoder_hidden_layers)
            ]

        self.embed_tokens = nn.Linear(timbre_hidden_dim, hidden_size)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.special_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.layers = nn.ModuleList(
            [
                AceStepEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=sliding_window if layer_types[i] == "sliding_attention" else None,
                )
                for i in range(num_timbre_encoder_hidden_layers)
            ]
        )

        self._layer_types = layer_types
        self.gradient_checkpointing = False

    @staticmethod
    def unpack_timbre_embeddings(
        timbre_embs_packed: torch.Tensor, refer_audio_order_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, d = timbre_embs_packed.shape
        device = timbre_embs_packed.device
        dtype = timbre_embs_packed.dtype

        B = int(refer_audio_order_mask.max().item() + 1)
        counts = torch.bincount(refer_audio_order_mask, minlength=B)
        max_count = counts.max().item()

        sorted_indices = torch.argsort(refer_audio_order_mask * N + torch.arange(N, device=device), stable=True)
        sorted_batch_ids = refer_audio_order_mask[sorted_indices]

        positions = torch.arange(N, device=device)
        batch_starts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])
        positions_in_sorted = positions - batch_starts[sorted_batch_ids]

        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(N, device=device)
        positions_in_batch = positions_in_sorted[inverse_indices]

        indices_2d = refer_audio_order_mask * max_count + positions_in_batch
        one_hot = F.one_hot(indices_2d, num_classes=B * max_count).to(dtype)

        timbre_embs_flat = one_hot.t() @ timbre_embs_packed
        timbre_embs_unpack = timbre_embs_flat.reshape(B, max_count, d)

        mask_flat = (one_hot.sum(dim=0) > 0).long()
        new_mask = mask_flat.reshape(B, max_count)
        return timbre_embs_unpack, new_mask

    def forward(
        self,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.embed_tokens(refer_audio_acoustic_hidden_states_packed)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        cos, sin = _ace_step_rotary_freqs(seq_len, self.head_dim, self.rope_theta, device, dtype)
        position_embeddings = (cos, sin)

        full_attn_mask = _create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device, attention_mask=None, is_causal=False
        )
        sliding_attn_mask = _create_4d_mask(
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            attention_mask=None,
            sliding_window=self.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )

        hidden_states = _run_encoder_layers(
            self.layers,
            self._layer_types,
            inputs_embeds,
            position_embeddings,
            full_attn_mask,
            sliding_attn_mask,
            self.gradient_checkpointing,
            self._gradient_checkpointing_func if hasattr(self, "_gradient_checkpointing_func") else None,
        )

        hidden_states = self.norm(hidden_states)
        # CLS-like pooling: first-token embedding per packed sequence.
        hidden_states = hidden_states[:, 0, :]
        timbre_embs_unpack, timbre_embs_mask = self.unpack_timbre_embeddings(hidden_states, refer_audio_order_mask)
        return timbre_embs_unpack, timbre_embs_mask


# --------------------------------------------------------------------------- #
#                               condition encoder                              #
# --------------------------------------------------------------------------- #


class AceStepConditionEncoder(ModelMixin, ConfigMixin):
    """Fuses text + lyric + timbre conditioning into the packed sequence used by
    the DiT's cross-attention.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        text_hidden_dim: int = 1024,
        timbre_hidden_dim: int = 64,
        num_lyric_encoder_hidden_layers: int = 8,
        num_timbre_encoder_hidden_layers: int = 4,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 128,
        layer_types: list = None,
    ):
        super().__init__()

        self.text_projector = nn.Linear(text_hidden_dim, hidden_size, bias=False)

        self.lyric_encoder = AceStepLyricEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            text_hidden_dim=text_hidden_dim,
            num_lyric_encoder_hidden_layers=num_lyric_encoder_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window,
            layer_types=layer_types,
        )

        self.timbre_encoder = AceStepTimbreEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            timbre_hidden_dim=timbre_hidden_dim,
            num_timbre_encoder_hidden_layers=num_timbre_encoder_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window,
        )

        # Learned null-condition embedding for classifier-free guidance, trained with
        # `cfg_ratio=0.15` in the original model. Broadcast along the sequence dim when used.
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Silence latent — VAE-encoded audio-silence, stored as (1, T_long, timbre_hidden_dim).
        # When no reference audio is provided, the pipeline slices `silence_latent[:, :timbre_fix_frame, :]`
        # and feeds that to the timbre encoder. Passing literal zeros puts the timbre encoder
        # OOD and produces drone-like audio (observed on all text2music outputs before this fix).
        # The placeholder here is overwritten by the converter with the real encoded silence,
        # so its shape just needs to match the timbre-encoder input: last dim is
        # `timbre_hidden_dim` (so smaller test configs with `timbre_hidden_dim != 64` also load).
        self.register_buffer(
            "silence_latent",
            torch.zeros(1, 15000, timbre_hidden_dim),
            persistent=True,
        )

    def forward(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_hidden_states = self.text_projector(text_hidden_states)

        lyric_hidden_states = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states, attention_mask=lyric_attention_mask
        )

        timbre_embs_unpack, timbre_embs_mask = self.timbre_encoder(
            refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask
        )

        encoder_hidden_states, encoder_attention_mask = _pack_sequences(
            lyric_hidden_states, timbre_embs_unpack, lyric_attention_mask, timbre_embs_mask
        )
        encoder_hidden_states, encoder_attention_mask = _pack_sequences(
            encoder_hidden_states, text_hidden_states, encoder_attention_mask, text_attention_mask
        )

        return encoder_hidden_states, encoder_attention_mask
