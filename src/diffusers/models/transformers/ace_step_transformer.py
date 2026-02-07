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

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Create a 4D attention mask compatible with SDPA and eager attention.

    Supports causal/bidirectional attention with optional sliding window.

    Returns:
        Tensor of shape `[batch, 1, seq_len, seq_len]` with `0.0` for visible positions and `-inf` for masked ones.
    """
    indices = torch.arange(seq_len, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    valid_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)

    if is_causal:
        valid_mask = valid_mask & (diff >= 0)

    if is_sliding_window and sliding_window is not None:
        if is_causal:
            valid_mask = valid_mask & (diff <= sliding_window)
        else:
            valid_mask = valid_mask & (torch.abs(diff) <= sliding_window)

    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is not None:
        padding_mask_4d = attention_mask.view(attention_mask.shape[0], 1, 1, seq_len).to(torch.bool)
        valid_mask = valid_mask & padding_mask_4d

    min_dtype = torch.finfo(dtype).min
    mask_tensor = torch.full(valid_mask.shape, min_dtype, dtype=dtype, device=device)
    mask_tensor.masked_fill_(valid_mask, 0.0)
    return mask_tensor


def _pack_sequences(
    hidden1: torch.Tensor, hidden2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack two sequences by concatenating and sorting valid tokens first.

    Args:
        hidden1: First hidden states `[B, L1, D]`.
        hidden2: Second hidden states `[B, L2, D]`.
        mask1: Mask for first sequence `[B, L1]`.
        mask2: Mask for second sequence `[B, L2]`.

    Returns:
        Tuple of `(packed_hidden_states, new_mask)` with valid tokens sorted first.
    """
    hidden_cat = torch.cat([hidden1, hidden2], dim=1)
    mask_cat = torch.cat([mask1, mask2], dim=1)

    B, L, D = hidden_cat.shape
    sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)
    hidden_left = torch.gather(hidden_cat, 1, sort_idx.unsqueeze(-1).expand(B, L, D))
    lengths = mask_cat.sum(dim=1)
    new_mask = torch.arange(L, dtype=torch.long, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(1)
    return hidden_left, new_mask


class AceStepRMSNorm(nn.Module):
    """RMS Normalization used throughout the ACE-Step model."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class AceStepRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for ACE-Step attention layers."""

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class AceStepMLP(nn.Module):
    """MLP (SwiGLU) used in ACE-Step transformer layers."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AceStepTimestepEmbedding(nn.Module):
    """
    Timestep embedding module for the ACE-Step diffusion model.

    Converts scalar timestep values into high-dimensional embeddings using sinusoidal positional encoding followed by
    MLP layers. Also produces scale-shift parameters for adaptive layer normalization.
    """

    def __init__(self, in_channels: int, time_embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.in_channels = in_channels
        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6)
        self.scale = scale

    def _timestep_embedding(self, t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        t = t * self.scale
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_freq = self._timestep_embedding(t, self.in_channels)
        temb = self.linear_1(t_freq.to(t.dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).unflatten(1, (6, -1))
        return temb, timestep_proj


class AceStepAttention(nn.Module):
    """
    Multi-headed attention module for the ACE-Step model.

    Supports self-attention and cross-attention with RMSNorm on query/key and optional sliding window attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        is_cross_attention: bool = False,
        is_causal: bool = False,
        sliding_window: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        self.attention_dropout = attention_dropout
        if is_cross_attention:
            is_causal = False
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm = AceStepRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = AceStepRMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(-3, -2)

        is_cross_attention = self.is_cross_attention and encoder_hidden_states is not None

        if is_cross_attention:
            kv_input = encoder_hidden_states
            kv_shape = (*encoder_hidden_states.shape[:-1], self.num_key_value_heads, self.head_dim)
        else:
            kv_input = hidden_states
            kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        key_states = self.k_norm(self.k_proj(kv_input).view(kv_shape)).transpose(-3, -2)
        value_states = self.v_proj(kv_input).view(kv_shape).transpose(-3, -2)

        if not is_cross_attention and position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads for grouped query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=-3)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=-3)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(-3, -2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class AceStepEncoderLayer(nn.Module):
    """
    Encoder layer for the ACE-Step model.

    Consists of self-attention and MLP (feed-forward) sub-layers with residual connections.
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
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            is_cross_attention=False,
            is_causal=False,
            sliding_window=sliding_window,
            rms_norm_eps=rms_norm_eps,
        )
        self.input_layernorm = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
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
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AceStepDiTLayer(nn.Module):
    """
    DiT (Diffusion Transformer) layer for the ACE-Step model.

    Implements a transformer layer with:
    1. Self-attention with adaptive layer norm (AdaLN)
    2. Cross-attention for conditioning on encoder outputs
    3. Feed-forward MLP with adaptive layer norm

    Uses scale-shift modulation from timestep embeddings for adaptive normalization.
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
        use_cross_attention: bool = True,
    ):
        super().__init__()
        # Self-attention
        self.self_attn_norm = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = AceStepAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            is_cross_attention=False,
            is_causal=False,
            sliding_window=sliding_window,
            rms_norm_eps=rms_norm_eps,
        )

        # Cross-attention
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
            self.cross_attn = AceStepAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                is_cross_attention=True,
                rms_norm_eps=rms_norm_eps,
            )

        # Feed-forward MLP
        self.mlp_norm = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = AceStepMLP(hidden_size, intermediate_size)

        # Scale-shift table for adaptive layer norm (6 values)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table + temb).chunk(
            6, dim=1
        )

        # Self-attention with AdaLN
        norm_hidden_states = (self.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.self_attn(
            hidden_states=norm_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # Cross-attention
        if self.use_cross_attention and encoder_hidden_states is not None:
            norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(hidden_states)
            attn_output = self.cross_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # Feed-forward MLP with AdaLN
        norm_hidden_states = (self.mlp_norm(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.mlp(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        return hidden_states


class AceStepDiTModel(ModelMixin, ConfigMixin):
    """
    The Diffusion Transformer (DiT) model for ACE-Step music generation.

    This model generates audio latents conditioned on text, lyrics, and timbre. It uses patch-based processing with
    transformer layers, timestep conditioning via AdaLN, and cross-attention to encoder outputs.

    Parameters:
        hidden_size (`int`, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, defaults to 6144):
            Dimension of the MLP intermediate representations.
        num_hidden_layers (`int`, defaults to 24):
            Number of DiT transformer layers.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads for query states.
        num_key_value_heads (`int`, defaults to 8):
            Number of attention heads for key and value states (for grouped query attention).
        head_dim (`int`, defaults to 128):
            Dimension of each attention head.
        in_channels (`int`, defaults to 192):
            Number of input channels (context_latents + hidden_states concatenated).
        audio_acoustic_hidden_dim (`int`, defaults to 64):
            Output dimension of the model (acoustic latent dimension).
        patch_size (`int`, defaults to 2):
            Patch size for input patchification.
        max_position_embeddings (`int`, defaults to 32768):
            Maximum sequence length for rotary embeddings.
        rope_theta (`float`, defaults to 1000000.0):
            Base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in attention projection layers.
        attention_dropout (`float`, defaults to 0.0):
            Dropout probability for attention weights.
        rms_norm_eps (`float`, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_sliding_window (`bool`, defaults to `True`):
            Whether to use sliding window attention for alternating layers.
        sliding_window (`int`, defaults to 128):
            Sliding window size for local attention layers.
        layer_types (`List[str]`, *optional*):
            Attention pattern for each layer. Defaults to alternating `"sliding_attention"` and `"full_attention"`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        in_channels: int = 192,
        audio_acoustic_hidden_dim: int = 64,
        patch_size: int = 2,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_sliding_window: bool = True,
        sliding_window: int = 128,
        layer_types: Optional[List[str]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Determine layer types
        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(num_hidden_layers)
            ]

        # Rotary position embeddings
        self.rotary_emb = AceStepRotaryEmbedding(
            dim=head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
        )

        # DiT transformer layers
        self.layers = nn.ModuleList(
            [
                AceStepDiTLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=sliding_window if layer_types[i] == "sliding_attention" else None,
                    use_cross_attention=True,
                )
                for i in range(num_hidden_layers)
            ]
        )

        # Store layer types for mask selection
        self._layer_types = layer_types

        # Input projection (patchify)
        self.proj_in_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Timestep embeddings
        self.time_embed = AceStepTimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.time_embed_r = AceStepTimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)

        # Condition projection
        self.condition_embedder = nn.Linear(hidden_size, hidden_size, bias=True)

        # Output (de-patchify)
        self.norm_out = AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj_out_conv = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=audio_acoustic_hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, hidden_size) / hidden_size**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`AceStepDiTModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, channels)`):
                Noisy latent input for the diffusion process.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Current diffusion timestep `t`.
            timestep_r (`torch.Tensor` of shape `(batch_size,)`):
                Reference timestep `r` (set equal to `t` for standard inference).
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, encoder_seq_len, hidden_size)`):
                Conditioning embeddings from the condition encoder (text + lyrics + timbre).
            context_latents (`torch.Tensor` of shape `(batch_size, seq_len, context_dim)`):
                Context latents (source latents concatenated with chunk masks).
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the hidden states sequence.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the encoder hidden states.
            return_dict (`bool`, defaults to `True`):
                Whether to return a `Transformer2DModelOutput` or a plain tuple.

        Returns:
            `Transformer2DModelOutput` or `tuple`: The predicted velocity field for flow matching.
        """
        # Compute timestep embeddings
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concatenate context latents with hidden states
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden_states.shape[1]

        # Pad if sequence length is not divisible by patch_size
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode="constant", value=0)

        # Patchify: [B, T, C] -> [B, C, T] -> conv -> [B, C', T'] -> [B, T', C']
        hidden_states = self.proj_in_conv(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Project encoder hidden states
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        # Position embeddings
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build attention masks
        dtype = hidden_states.dtype
        device = hidden_states.device
        encoder_seq_len = encoder_hidden_states.shape[1]

        full_attn_mask = _create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device, attention_mask=None, is_causal=False
        )
        encoder_4d_mask = _create_4d_mask(
            seq_len=max(seq_len, encoder_seq_len), dtype=dtype, device=device, attention_mask=None, is_causal=False
        )
        encoder_4d_mask = encoder_4d_mask[:, :, :seq_len, :encoder_seq_len]

        sliding_attn_mask = None
        if self.config.use_sliding_window:
            sliding_attn_mask = _create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=None,
                sliding_window=self.config.sliding_window,
                is_sliding_window=True,
                is_causal=False,
            )

        # Process through transformer layers
        for i, layer_module in enumerate(self.layers):
            layer_type = self._layer_types[i]
            if layer_type == "sliding_attention" and sliding_attn_mask is not None:
                layer_attn_mask = sliding_attn_mask
            else:
                layer_attn_mask = full_attn_mask

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    layer_attn_mask,
                    encoder_hidden_states,
                    encoder_4d_mask,
                )
            else:
                hidden_states = layer_module(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    temb=timestep_proj,
                    attention_mask=layer_attn_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_4d_mask,
                )

        # Adaptive output normalization
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)

        # De-patchify: [B, T', C'] -> [B, C', T'] -> deconv -> [B, C, T] -> [B, T, C]
        hidden_states = self.proj_out_conv(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Crop to original sequence length
        hidden_states = hidden_states[:, :original_seq_len, :]

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
