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

"""
Pipeline-specific models for ACE-Step: ConditionEncoder, LyricEncoder, TimbreEncoder, AudioTokenizer, and
AudioTokenDetokenizer.

These models are used within the AceStepPipeline to encode conditioning inputs (text, lyrics, timbre) for
cross-attention in the DiT model.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ...models.transformers.ace_step_transformer import (
    AceStepEncoderLayer,
    AceStepRMSNorm,
    AceStepRotaryEmbedding,
    _create_4d_mask,
    _pack_sequences,
)
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AceStepLyricEncoder(ModelMixin, ConfigMixin):
    """
    Encoder for processing lyric text embeddings in the ACE-Step pipeline.

    Encodes lyric text hidden states using a transformer encoder architecture with bidirectional attention. Projects
    text embeddings to model hidden size and processes them through multiple encoder layers.

    Parameters:
        hidden_size (`int`, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, defaults to 6144):
            Dimension of the MLP representations.
        text_hidden_dim (`int`, defaults to 1024):
            Dimension of the input text embeddings from the text encoder.
        num_lyric_encoder_hidden_layers (`int`, defaults to 8):
            Number of transformer encoder layers.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, defaults to 8):
            Number of key/value heads for grouped query attention.
        head_dim (`int`, defaults to 128):
            Dimension of each attention head.
        max_position_embeddings (`int`, defaults to 32768):
            Maximum sequence length for rotary embeddings.
        rope_theta (`float`, defaults to 1000000.0):
            Base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, defaults to 0.0):
            Dropout probability for attention weights.
        rms_norm_eps (`float`, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_sliding_window (`bool`, defaults to `True`):
            Whether to use sliding window attention.
        sliding_window (`int`, defaults to 128):
            Sliding window size.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
    """

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
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_sliding_window: bool = True,
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
        self.norm = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = AceStepRotaryEmbedding(
            dim=head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
        )

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
        self._use_sliding_window = use_sliding_window
        self._sliding_window = sliding_window

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, text_hidden_dim)`):
                Lyric text embeddings from the text encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Attention mask for padding (1 for valid, 0 for padding).

        Returns:
            `torch.Tensor`: Encoded lyric hidden states of shape `(batch_size, seq_len, hidden_size)`.
        """
        inputs_embeds = self.embed_tokens(inputs_embeds)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Build attention masks
        full_attn_mask = _create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device, attention_mask=attention_mask, is_causal=False
        )
        sliding_attn_mask = None
        if self._use_sliding_window:
            sliding_attn_mask = _create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,
                sliding_window=self._sliding_window,
                is_sliding_window=True,
                is_causal=False,
            )

        hidden_states = inputs_embeds
        for i, layer_module in enumerate(self.layers):
            layer_type = self._layer_types[i]
            if layer_type == "sliding_attention" and sliding_attn_mask is not None:
                mask = sliding_attn_mask
            else:
                mask = full_attn_mask

            hidden_states = layer_module(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class AceStepTimbreEncoder(ModelMixin, ConfigMixin):
    """
    Encoder for extracting timbre embeddings from reference audio in the ACE-Step pipeline.

    Processes packed reference audio acoustic features to extract timbre representations. Outputs are unpacked back to
    batch format for use in conditioning.

    Parameters:
        hidden_size (`int`, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, defaults to 6144):
            Dimension of the MLP representations.
        timbre_hidden_dim (`int`, defaults to 64):
            Dimension of the input acoustic features.
        num_timbre_encoder_hidden_layers (`int`, defaults to 4):
            Number of transformer encoder layers.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, defaults to 8):
            Number of key/value heads.
        head_dim (`int`, defaults to 128):
            Dimension of each attention head.
        max_position_embeddings (`int`, defaults to 32768):
            Maximum sequence length for rotary embeddings.
        rope_theta (`float`, defaults to 1000000.0):
            Base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, defaults to 0.0):
            Dropout probability for attention weights.
        rms_norm_eps (`float`, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_sliding_window (`bool`, defaults to `True`):
            Whether to use sliding window attention.
        sliding_window (`int`, defaults to 128):
            Sliding window size.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
    """

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
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_sliding_window: bool = True,
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
        self.norm = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = AceStepRotaryEmbedding(
            dim=head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
        )
        self.special_token = nn.Parameter(torch.randn(1, 1, hidden_size))

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
        self._use_sliding_window = use_sliding_window
        self._sliding_window = sliding_window

    @staticmethod
    def unpack_timbre_embeddings(
        timbre_embs_packed: torch.Tensor, refer_audio_order_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack packed timbre embeddings into batch format.

        Args:
            timbre_embs_packed (`torch.Tensor` of shape `(N, d)`):
                Packed timbre embeddings.
            refer_audio_order_mask (`torch.Tensor` of shape `(N,)`):
                Order mask indicating batch assignment.

        Returns:
            Tuple of `(unpacked_embeddings, mask)`:
            - `unpacked_embeddings` of shape `(B, max_count, d)`
            - `mask` of shape `(B, max_count)`
        """
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
        """
        Args:
            refer_audio_acoustic_hidden_states_packed (`torch.FloatTensor` of shape `(N, T, timbre_hidden_dim)`):
                Packed reference audio acoustic features.
            refer_audio_order_mask (`torch.LongTensor` of shape `(N,)`):
                Order mask indicating which batch element each packed sequence belongs to.

        Returns:
            Tuple of `(timbre_embeddings, timbre_mask)`:
            - `timbre_embeddings` of shape `(B, max_refs, hidden_size)`
            - `timbre_mask` of shape `(B, max_refs)`
        """
        inputs_embeds = self.embed_tokens(refer_audio_acoustic_hidden_states_packed)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Build attention masks
        full_attn_mask = _create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device, attention_mask=None, is_causal=False
        )
        sliding_attn_mask = None
        if self._use_sliding_window:
            sliding_attn_mask = _create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=None,
                sliding_window=self._sliding_window,
                is_sliding_window=True,
                is_causal=False,
            )

        hidden_states = inputs_embeds
        for i, layer_module in enumerate(self.layers):
            layer_type = self._layer_types[i]
            if layer_type == "sliding_attention" and sliding_attn_mask is not None:
                mask = sliding_attn_mask
            else:
                mask = full_attn_mask

            hidden_states = layer_module(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )

        hidden_states = self.norm(hidden_states)
        # Extract first token (CLS-like) as timbre embedding
        hidden_states = hidden_states[:, 0, :]
        timbre_embs_unpack, timbre_embs_mask = self.unpack_timbre_embeddings(hidden_states, refer_audio_order_mask)
        return timbre_embs_unpack, timbre_embs_mask


class AceStepConditionEncoder(ModelMixin, ConfigMixin):
    """
    Condition encoder for the ACE-Step pipeline.

    Encodes multiple conditioning inputs (text, lyrics, timbre) and packs them into a single sequence for
    cross-attention in the DiT model. This model handles projection, encoding, and sequence packing.

    Parameters:
        hidden_size (`int`, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, defaults to 6144):
            Dimension of the MLP representations.
        text_hidden_dim (`int`, defaults to 1024):
            Dimension of the input text embeddings.
        timbre_hidden_dim (`int`, defaults to 64):
            Dimension of the input acoustic features.
        num_lyric_encoder_hidden_layers (`int`, defaults to 8):
            Number of lyric encoder layers.
        num_timbre_encoder_hidden_layers (`int`, defaults to 4):
            Number of timbre encoder layers.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, defaults to 8):
            Number of key/value heads.
        head_dim (`int`, defaults to 128):
            Dimension of each attention head.
        max_position_embeddings (`int`, defaults to 32768):
            Maximum sequence length for rotary embeddings.
        rope_theta (`float`, defaults to 1000000.0):
            Base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, defaults to 0.0):
            Dropout probability for attention weights.
        rms_norm_eps (`float`, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_sliding_window (`bool`, defaults to `True`):
            Whether to use sliding window attention.
        sliding_window (`int`, defaults to 128):
            Sliding window size.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
    """

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
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_sliding_window: bool = True,
        sliding_window: int = 128,
        layer_types: list = None,
    ):
        super().__init__()

        # Text projector
        self.text_projector = nn.Linear(text_hidden_dim, hidden_size, bias=False)

        # Lyric encoder
        self.lyric_encoder = AceStepLyricEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            text_hidden_dim=text_hidden_dim,
            num_lyric_encoder_hidden_layers=num_lyric_encoder_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            layer_types=layer_types,
        )

        # Timbre encoder
        self.timbre_encoder = AceStepTimbreEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            timbre_hidden_dim=timbre_hidden_dim,
            num_timbre_encoder_hidden_layers=num_timbre_encoder_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
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
        """
        Encode text, lyrics, and timbre into a single packed conditioning sequence.

        Args:
            text_hidden_states (`torch.FloatTensor` of shape `(batch_size, text_seq_len, text_hidden_dim)`):
                Text embeddings from the text encoder.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`):
                Attention mask for text.
            lyric_hidden_states (`torch.FloatTensor` of shape `(batch_size, lyric_seq_len, text_hidden_dim)`):
                Lyric embeddings from the text encoder.
            lyric_attention_mask (`torch.Tensor` of shape `(batch_size, lyric_seq_len)`):
                Attention mask for lyrics.
            refer_audio_acoustic_hidden_states_packed (`torch.FloatTensor` of shape `(N, T, timbre_hidden_dim)`):
                Packed reference audio acoustic features.
            refer_audio_order_mask (`torch.LongTensor` of shape `(N,)`):
                Order mask for reference audio packing.

        Returns:
            Tuple of `(encoder_hidden_states, encoder_attention_mask)`:
            - `encoder_hidden_states` of shape `(batch_size, total_seq_len, hidden_size)`
            - `encoder_attention_mask` of shape `(batch_size, total_seq_len)`
        """
        # Project text
        text_hidden_states = self.text_projector(text_hidden_states)

        # Encode lyrics
        lyric_hidden_states = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states,
            attention_mask=lyric_attention_mask,
        )

        # Encode timbre
        timbre_embs_unpack, timbre_embs_mask = self.timbre_encoder(
            refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask
        )

        # Pack sequences: lyrics + timbre, then + text
        encoder_hidden_states, encoder_attention_mask = _pack_sequences(
            lyric_hidden_states, timbre_embs_unpack, lyric_attention_mask, timbre_embs_mask
        )
        encoder_hidden_states, encoder_attention_mask = _pack_sequences(
            encoder_hidden_states, text_hidden_states, encoder_attention_mask, text_attention_mask
        )

        return encoder_hidden_states, encoder_attention_mask
