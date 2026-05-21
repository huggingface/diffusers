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


"""ACE-Step audio tokenizer / detokenizer.

Converts between the VAE's 25 Hz acoustic latents and the 5 Hz semantic audio tokens used by the cover-conditioning
path. The internal ``AceStepEncoderLayer`` block is duplicated here (and in
:mod:`diffusers.models.condition_embedders.condition_encoder_ace_step`) to keep the two model files independent.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm
from ..transformers.ace_step_transformer import (
    AceStepAttention,
    AceStepMLP,
    _ace_step_rotary_freqs,
    _create_4d_mask,
    _is_flash_attention_backend,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AceStepEncoderLayer(nn.Module):
    """Pre-LN transformer block used by the lyric and timbre encoders."""

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
            sliding_window=sliding_window,
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


class _AceStepResidualFSQ(nn.Module):
    """Minimal ResidualFSQ compatible with ACE-Step's saved tokenizer weights."""

    def __init__(
        self,
        dim: int = 2048,
        levels: Optional[list] = None,
        num_quantizers: int = 1,
    ):
        super().__init__()

        if levels is None:
            levels = [8, 8, 8, 5, 5, 5]

        self.levels = levels
        self.num_quantizers = num_quantizers
        self.codebook_dim = len(levels)

        self.project_in = nn.Linear(dim, self.codebook_dim)
        self.project_out = nn.Linear(self.codebook_dim, dim)

        levels_tensor = torch.tensor(levels, dtype=torch.long)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)
        scales = torch.stack([levels_tensor.float() ** -i for i in range(num_quantizers)])
        self.register_buffer("_levels", levels_tensor, persistent=False)
        self.register_buffer("_basis", basis, persistent=False)
        self.register_buffer("scales", scales, persistent=False)

    @property
    def codebook_size(self) -> int:
        return int(torch.prod(self._levels).item())

    def _indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        levels = self._levels.to(device=indices.device)
        basis = self._basis.to(device=indices.device)
        level_indices = (indices.long().unsqueeze(-1) // basis) % levels
        scale = 2.0 / (levels.to(dtype=torch.float32) - 1.0)
        return level_indices.to(dtype=torch.float32) * scale - 1.0

    def _codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        levels = self._levels.to(device=codes.device, dtype=codes.dtype)
        basis = self._basis.to(device=codes.device, dtype=codes.dtype)
        level_indices = (codes + 1.0) / (2.0 / (levels - 1.0))
        return (level_indices * basis).sum(dim=-1).round().to(torch.long)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        levels = self._levels.to(device=x.device, dtype=x.dtype)
        levels_minus_one = levels - 1.0
        step = 2.0 / levels_minus_one
        bracket = levels_minus_one * (x.clamp(-1.0, 1.0) + 1.0) / 2.0 + 0.5
        return step * torch.floor(bracket) - 1.0

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 2:
            indices = indices.unsqueeze(-1)
        if indices.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"Expected audio code indices with last dimension {self.num_quantizers}, got {indices.shape[-1]}."
            )

        codes = []
        for quantizer_idx in range(self.num_quantizers):
            code = self._indices_to_codes(indices[..., quantizer_idx])
            scale = self.scales[quantizer_idx].to(device=code.device, dtype=code.dtype)
            codes.append(code * scale)
        return torch.stack(codes, dim=0)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_indices(indices).sum(dim=0)
        weight = self.project_out.weight.float()
        bias = self.project_out.bias.float() if self.project_out.bias is not None else None
        output = F.linear(codes.float(), weight, bias)
        return output.to(dtype=self.project_out.weight.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dtype = hidden_states.dtype
        weight = self.project_in.weight.float()
        bias = self.project_in.bias.float() if self.project_in.bias is not None else None
        hidden_states = F.linear(hidden_states.float(), weight, bias)

        levels = self._levels.to(device=hidden_states.device, dtype=hidden_states.dtype)
        soft_clamp = 1.0 + (1.0 / (levels - 1.0))
        hidden_states = (hidden_states / soft_clamp).tanh() * soft_clamp

        quantized_out = torch.zeros_like(hidden_states)
        residual = hidden_states
        all_indices = []
        for scale in self.scales.to(device=hidden_states.device, dtype=hidden_states.dtype):
            quantized = self._quantize(residual / scale) * scale
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(self._codes_to_indices(quantized / scale))

        weight = self.project_out.weight.float()
        bias = self.project_out.bias.float() if self.project_out.bias is not None else None
        quantized_out = F.linear(quantized_out.float(), weight, bias).to(dtype=input_dtype)
        all_indices = torch.stack(all_indices, dim=-1)
        return quantized_out, all_indices


class AceStepAttentionPooler(nn.Module):
    """Attention pooler used by the ACE-Step audio tokenizer."""

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_attention_pooler_hidden_layers: int = 2,
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
                for i in range(num_attention_pooler_hidden_layers)
            ]

        self.embed_tokens = nn.Linear(hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.special_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
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
                for i in range(num_attention_pooler_hidden_layers)
            ]
        )
        self._layer_types = layer_types

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, patch_size, _ = hidden_states.shape
        hidden_states = self.embed_tokens(hidden_states)
        special_token = self.special_token.to(device=hidden_states.device, dtype=hidden_states.dtype)
        special_token = special_token.expand(batch_size, num_patches, -1, -1)
        hidden_states = torch.cat([special_token, hidden_states], dim=2)
        hidden_states = hidden_states.reshape(batch_size * num_patches, patch_size + 1, -1)

        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device
        position_embeddings = _ace_step_rotary_freqs(seq_len, self.head_dim, self.rope_theta, device, dtype)
        sliding_attn_mask = None
        if not _is_flash_attention_backend(self.layers[0].self_attn.processor):
            sliding_attn_mask = _create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=None,
                sliding_window=self.sliding_window,
                is_sliding_window=True,
                is_causal=False,
            )

        for i, layer_module in enumerate(self.layers):
            mask = sliding_attn_mask if self._layer_types[i] == "sliding_attention" else None
            hidden_states = layer_module(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states[:, 0, :]
        return hidden_states.reshape(batch_size, num_patches, -1)


class AceStepAudioTokenDetokenizer(ModelMixin, ConfigMixin):
    """Expands ACE-Step 5 Hz audio tokens back to 25 Hz acoustic conditioning."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        audio_acoustic_hidden_dim: int = 64,
        pool_window_size: int = 5,
        num_attention_pooler_hidden_layers: int = 2,
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
                for i in range(num_attention_pooler_hidden_layers)
            ]

        self.embed_tokens = nn.Linear(hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.special_tokens = nn.Parameter(torch.randn(1, pool_window_size, hidden_size) * 0.02)
        self.proj_out = nn.Linear(hidden_size, audio_acoustic_hidden_dim)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.pool_window_size = pool_window_size
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
                for i in range(num_attention_pooler_hidden_layers)
            ]
        )
        self._layer_types = layer_types
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input audio tokens of shape `(batch_size, num_tokens, hidden_size)` to be unpooled back to the 25 Hz
                acoustic-latent rate.
        """
        batch_size, num_tokens, _ = hidden_states.shape
        hidden_states = self.embed_tokens(hidden_states)
        hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, self.pool_window_size, -1)
        special_tokens = self.special_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + special_tokens.unsqueeze(0)
        hidden_states = hidden_states.reshape(batch_size * num_tokens, self.pool_window_size, -1)

        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device
        position_embeddings = _ace_step_rotary_freqs(seq_len, self.head_dim, self.rope_theta, device, dtype)
        sliding_attn_mask = None
        if not _is_flash_attention_backend(self.layers[0].self_attn.processor):
            sliding_attn_mask = _create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=None,
                sliding_window=self.sliding_window,
                is_sliding_window=True,
                is_causal=False,
            )

        for i, layer_module in enumerate(self.layers):
            mask = sliding_attn_mask if self._layer_types[i] == "sliding_attention" else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module, hidden_states, position_embeddings, mask
                )
            else:
                hidden_states = layer_module(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=mask,
                )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.reshape(batch_size, num_tokens * self.pool_window_size, -1)


class AceStepAudioTokenizer(ModelMixin, ConfigMixin):
    """Converts 25 Hz acoustic latents to ACE-Step 5 Hz audio tokens."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        audio_acoustic_hidden_dim: int = 64,
        pool_window_size: int = 5,
        fsq_dim: int = 2048,
        fsq_input_levels: list = None,
        fsq_input_num_quantizers: int = 1,
        num_attention_pooler_hidden_layers: int = 2,
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

        if fsq_input_levels is None:
            fsq_input_levels = [8, 8, 8, 5, 5, 5]

        self.audio_acoustic_proj = nn.Linear(audio_acoustic_hidden_dim, hidden_size)
        self.attention_pooler = AceStepAttentionPooler(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_pooler_hidden_layers=num_attention_pooler_hidden_layers,
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
        self.quantizer = _AceStepResidualFSQ(
            dim=fsq_dim,
            levels=fsq_input_levels,
            num_quantizers=fsq_input_num_quantizers,
        )
        self.pool_window_size = pool_window_size

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input acoustic latents of shape `(batch_size, latent_length, audio_acoustic_hidden_dim)` to be
                quantized into ACE-Step 5 Hz audio tokens.
        """
        input_dtype = hidden_states.dtype
        hidden_states = self.audio_acoustic_proj(hidden_states)
        hidden_states = self.attention_pooler(hidden_states)
        quantized, indices = self.quantizer(hidden_states)
        return quantized.to(dtype=input_dtype), indices

    def tokenize(
        self,
        hidden_states: torch.Tensor,
        silence_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, latent_length, acoustic_dim = hidden_states.shape
        pad_len = (-latent_length) % self.pool_window_size
        if pad_len:
            if silence_latent is not None and silence_latent.shape[-1] == acoustic_dim:
                pad = silence_latent[:, :pad_len, :].to(device=hidden_states.device, dtype=hidden_states.dtype)
                pad = pad.expand(batch_size, -1, -1)
            else:
                pad = torch.zeros(
                    batch_size, pad_len, acoustic_dim, device=hidden_states.device, dtype=hidden_states.dtype
                )
            hidden_states = torch.cat([hidden_states, pad], dim=1)

        num_patches = hidden_states.shape[1] // self.pool_window_size
        hidden_states = hidden_states.reshape(batch_size, num_patches, self.pool_window_size, acoustic_dim)
        return self(hidden_states)
