# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class StableAudio3DiTModelOutput(BaseOutput):
    """
    The output of [`StableAudio3DiTModel`].

    Args:
        sample (`torch.Tensor`):
            The predicted velocity field, of the same shape as the input `hidden_states`.
    """

    sample: torch.Tensor


class StableAudio3RMSNorm(nn.Module):
    """RMS normalization with a learnable scale (`gamma`). Matches the reference SA3 DiT norm.

    The reference uses `norm_type="rms_norm"` with `force_fp32=True`, so the normalization is computed in float32
    regardless of the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6, force_fp32: bool = True):
        super().__init__()
        self.eps = eps
        self.force_fp32 = force_fp32
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        compute_dtype = torch.float32 if self.force_fp32 else input_dtype
        hidden_states = hidden_states.to(compute_dtype)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states * self.gamma.to(compute_dtype)
        return hidden_states.to(input_dtype)


class StableAudio3ExpoFourierFeatures(nn.Module):
    """Exponentially-spaced Fourier features for the timestep (`timestep_features_type="expo"`).

    Frequencies are spaced log-uniformly between `min_freq` and `max_freq`. The output is the concatenation of the
    cosine and sine halves, of shape `(batch, dim)`.
    """

    def __init__(self, dim: int, min_freq: float = 0.5, max_freq: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1)
        half = self.dim // 2
        ramp = torch.linspace(0.0, 1.0, half, device=timestep.device, dtype=torch.float32)
        freqs = torch.exp(ramp * (math.log(self.max_freq) - math.log(self.min_freq)) + math.log(self.min_freq))
        args = timestep.float() * freqs * 2.0 * math.pi
        out = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return out.to(timestep.dtype)


class StableAudio3RotaryEmbedding(nn.Module):
    """Shared rotary positional embedding. Stable Audio 3 applies the RoPE embeddings partially over only the first
    `2 * (dim // 2)` head channels).
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # `persistent=True` so it appears in the state dict, matching the reference checkpoint key
        # `transformer.rotary_pos_emb.inv_freq`.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)  # (seq_len, rot_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (2, -1))
    x1, x2 = x.unbind(-2)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_partial(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Partial RoPE — rotate the first `rot_dim` channels, leave the rest.

    `t` has shape `(batch, seq_len, heads, head_dim)`; `freqs` has shape `(seq_len, rot_dim)`.
    """
    rot_dim = freqs.shape[-1]
    out_dtype = t.dtype
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot.float()
    freqs = freqs[-t_rot.shape[-3] :].float().unsqueeze(1)  # (seq_len, 1, rot_dim) broadcasts over heads
    t_rot = t_rot * freqs.cos() + _rotate_half(t_rot) * freqs.sin()
    return torch.cat((t_rot.to(out_dtype), t_pass), dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: `x * (1 + scale) + shift` (scale/shift broadcast over the sequence axis)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class StableAudio3FeedForward(nn.Module):
    """SwiGLU feed-forward. Reference keys: `ff.ff.0.proj` (gated in-proj) and `ff.ff.2` (out-proj)."""

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner * 2, bias=True)
        self.proj_out = nn.Linear(inner, dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj_in(hidden_states).chunk(2, dim=-1)
        return self.proj_out(hidden_states * F.silu(gate))


class StableAudio3SelfAttnProcessor:
    """Differential self-attention with RMS QK-norm and partial RoPE."""

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "StableAudio3Attention", hidden_states: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.unflatten(-1, (attn.heads, attn.dim_heads))

        if attn.use_differential:
            q1, q2, k1, k2, v = attn.to_qkv(hidden_states).chunk(5, dim=-1)
            q1, q2, k1, k2, v = map(heads, (q1, q2, k1, k2, v))
            q1, q2 = attn.q_norm(q1), attn.q_norm(q2)
            k1, k2 = attn.k_norm(k1), attn.k_norm(k2)
            q1, q2 = _apply_rotary_partial(q1, rope), _apply_rotary_partial(q2, rope)
            k1, k2 = _apply_rotary_partial(k1, rope), _apply_rotary_partial(k2, rope)
            out = dispatch_attention_fn(
                q1, k1, v, backend=self._attention_backend, parallel_config=self._parallel_config
            ) - dispatch_attention_fn(
                q2, k2, v, backend=self._attention_backend, parallel_config=self._parallel_config
            )
        else:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            q, k, v = map(heads, (q, k, v))
            q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = _apply_rotary_partial(q, rope), _apply_rotary_partial(k, rope)
            out = dispatch_attention_fn(
                q, k, v, backend=self._attention_backend, parallel_config=self._parallel_config
            )

        return attn.to_out(out.flatten(2, 3))


class StableAudio3CrossAttnProcessor:
    """Differential cross-attention from audio latents to the text/duration context (see [`StableAudio3Attention`]).

    No RoPE — context tokens have no positional order relative to the audio latents.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "StableAudio3Attention",
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.unflatten(-1, (attn.heads, attn.dim_heads))

        # Boolean mask (B, T_ctx), True = valid. SDPA accepts a boolean mask directly.
        attn_mask = None if context_mask is None else context_mask[:, None, None, :].bool()

        if attn.use_differential:
            q1, q2 = attn.to_q(hidden_states).chunk(2, dim=-1)
            k1, k2, v = attn.to_kv(context).chunk(3, dim=-1)
            q1, q2, k1, k2, v = map(heads, (q1, q2, k1, k2, v))
            q1, q2 = attn.q_norm(q1), attn.q_norm(q2)
            k1, k2 = attn.k_norm(k1), attn.k_norm(k2)
            out = dispatch_attention_fn(
                q1, k1, v, attn_mask=attn_mask, backend=self._attention_backend, parallel_config=self._parallel_config
            ) - dispatch_attention_fn(
                q2, k2, v, attn_mask=attn_mask, backend=self._attention_backend, parallel_config=self._parallel_config
            )
        else:
            q = heads(attn.to_q(hidden_states))
            k, v = map(heads, attn.to_kv(context).chunk(2, dim=-1))
            q, k = attn.q_norm(q), attn.k_norm(k)
            out = dispatch_attention_fn(
                q, k, v, attn_mask=attn_mask, backend=self._attention_backend, parallel_config=self._parallel_config
            )

        return attn.to_out(out.flatten(2, 3))


class StableAudio3Attention(nn.Module, AttentionModuleMixin):
    """Shared self-/cross-attention module for the SA3 DiT, built on the `AttentionModuleMixin` +
    `AttentionProcessor` pattern (see [`StableAudio3SelfAttnProcessor`] / [`StableAudio3CrossAttnProcessor`]).

    Self-attention (`context_dim=None`): the fused `to_qkv` projection produces `[q1 | q2 | k1 | k2 | v]`.
    Cross-attention (`context_dim` set): `to_q` produces `[q1 | q2]` and `to_kv` produces `[k1 | k2 | v]`. When
    `use_differential` is `True` (SA3 default) the attention is differential: `Attn(Q1, K1, V) - Attn(Q2, K2, V)`.
    """

    _available_processors = [StableAudio3SelfAttnProcessor, StableAudio3CrossAttnProcessor]
    # The QKV projections are fused (self-attn) or split into differential q/kv groups (cross-attn), neither of
    # which matches the plain to_q/to_k/to_v shape `fuse_projections`/`unfuse_projections` assume.
    _supports_qkv_fusion = False

    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        use_differential: bool = True,
        context_dim: Optional[int] = None,
        processor=None,
    ):
        super().__init__()
        self.dim_heads = dim_heads
        self.heads = dim // dim_heads
        self.use_differential = use_differential
        self.is_cross_attention = context_dim is not None

        if self.is_cross_attention:
            n_q = 2 if use_differential else 1
            n_kv = 3 if use_differential else 2
            self.to_q = nn.Linear(dim, dim * n_q, bias=False)
            self.to_kv = nn.Linear(context_dim, dim * n_kv, bias=False)
        else:
            n_proj = 5 if use_differential else 3
            self.to_qkv = nn.Linear(dim, dim * n_proj, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        self.q_norm = StableAudio3RMSNorm(dim_heads)
        self.k_norm = StableAudio3RMSNorm(dim_heads)

        if processor is None:
            processor = (
                StableAudio3CrossAttnProcessor() if self.is_cross_attention else StableAudio3SelfAttnProcessor()
            )
        self.set_processor(processor)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


class StableAudio3DiTBlock(nn.Module):
    """Single SA3 DiT transformer block.

    Order of operations:
      1. AdaLN-modulated self-attention (partial RoPE, RMS QK-norm)
      2. Cross-attention to the text/duration context (plain RMS pre-norm)
      3. AdaLN-modulated SwiGLU feed-forward

    The AdaLN modulation is `to_scale_shift_gate + global_modulation`, split into six chunks `(scale_attn, shift_attn,
    gate_attn, scale_ff, shift_ff, gate_ff)`. Each gated branch is scaled by `sigmoid(1 - gate)`. Cross-attention is
    *not* AdaLN-modulated, matching the reference (`cross_attend_norm` is a plain RMS norm).

    When `local_seq` is provided (inpainting), it is projected per-block by `to_local_embed` and added to the audio
    positions of the residual stream after cross-attention (and before the feed-forward), matching the reference.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        dim_heads: int = 64,
        use_differential: bool = True,
        ff_mult: float = 4.0,
        local_add_cond_dim: int = 257,
    ):
        super().__init__()
        self.pre_norm = StableAudio3RMSNorm(dim)
        self.self_attn = StableAudio3Attention(dim, dim_heads=dim_heads, use_differential=use_differential)

        self.cross_attend_norm = StableAudio3RMSNorm(dim)
        self.cross_attn = StableAudio3Attention(
            dim, context_dim=context_dim, dim_heads=dim_heads, use_differential=use_differential
        )

        self.ff_norm = StableAudio3RMSNorm(dim)
        self.ff = StableAudio3FeedForward(dim, mult=ff_mult)

        # Per-block AdaLN bias added to the shared global modulation (6 * dim chunks).
        self.to_scale_shift_gate = nn.Parameter(torch.randn(6 * dim) / dim**0.5)

        # Local-additive (inpaint) conditioning projection (zero-init output → no-op until trained).
        self.to_local_embed = nn.Sequential(
            nn.Linear(local_add_cond_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        nn.init.zeros_(self.to_local_embed[-1].weight)
        nn.init.zeros_(self.to_local_embed[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        global_modulation: torch.Tensor,
        rope: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        local_seq: Optional[torch.Tensor] = None,
        num_memory_tokens: int = 0,
    ) -> torch.Tensor:
        modulation = self.to_scale_shift_gate + global_modulation
        scale_attn, shift_attn, gate_attn, scale_ff, shift_ff, gate_ff = modulation.chunk(6, dim=-1)

        # Self-attention with AdaLN.
        residual = hidden_states
        norm_hidden = _modulate(self.pre_norm(hidden_states), shift_attn, scale_attn)
        attn_out = self.self_attn(norm_hidden, rope=rope)
        hidden_states = residual + attn_out * torch.sigmoid(1 - gate_attn).unsqueeze(1)

        # Cross-attention (no AdaLN modulation).
        hidden_states = hidden_states + self.cross_attn(
            self.cross_attend_norm(hidden_states), context=context, context_mask=context_mask
        )

        # Local-additive conditioning (inpaint) — only the audio positions, not the memory tokens.
        if local_seq is not None:
            local = self.to_local_embed(local_seq)
            audio = hidden_states[:, num_memory_tokens:] + local
            hidden_states = torch.cat([hidden_states[:, :num_memory_tokens], audio], dim=1)

        # Feed-forward with AdaLN.
        residual = hidden_states
        norm_hidden = _modulate(self.ff_norm(hidden_states), shift_ff, scale_ff)
        ff_out = self.ff(norm_hidden)
        hidden_states = residual + ff_out * torch.sigmoid(1 - gate_ff).unsqueeze(1)
        return hidden_states


class StableAudio3DiTModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    The Diffusion Transformer (DiT) backbone of [Stable Audio 3](https://stability.ai/news/stable-audio-3).

    The model takes a batch of noisy audio latents, a scalar timestep, a cross-attention context (projected text and
    duration tokens), and a global duration embedding, and predicts the velocity field (rectified-flow objective).

    Conditioning:
      - Cross-attention context (`encoder_hidden_states`) is projected by `to_cond_embed`.
      - The global duration embedding (`global_hidden_states`) is projected by `to_global_embed`, summed with the
        timestep embedding, then expanded by `global_cond_embedder` into the per-block AdaLN modulation.
      - `local_add_cond` (inpainting) is projected per-block by `to_local_embed`.

    `num_memory_tokens` learnable tokens are prepended to the audio sequence inside the transformer and removed before
    the output projection.

    Parameters:
        io_channels (`int`, defaults to 256): Number of latent channels.
        patch_size (`int`, defaults to 1): Temporal patch size applied before the transformer.
        embed_dim (`int`, defaults to 1536): Transformer hidden dimension.
        depth (`int`, defaults to 24): Number of [`StableAudio3DiTBlock`] layers.
        num_heads (`int`, defaults to 24): Number of attention heads.
        cond_token_dim (`int`, defaults to 768): Dimension of the cross-attention context tokens.
        global_cond_dim (`int`, defaults to 768): Dimension of the global duration embedding.
        local_add_cond_dim (`int`, defaults to 257): Channels of the local-additive (inpaint) tensor.
        timestep_features_dim (`int`, defaults to 256): Output dimension of the Fourier timestep features.
        ff_mult (`float`, defaults to 4.0): SwiGLU feed-forward expansion factor.
        num_memory_tokens (`int`, defaults to 64): Number of learnable memory tokens.
        use_differential_attention (`bool`, defaults to `True`): Enable differential self/cross attention.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        io_channels: int = 256,
        patch_size: int = 1,
        embed_dim: int = 1536,
        depth: int = 24,
        num_heads: int = 24,
        cond_token_dim: int = 768,
        global_cond_dim: int = 768,
        local_add_cond_dim: int = 257,
        timestep_features_dim: int = 256,
        ff_mult: float = 4.0,
        num_memory_tokens: int = 64,
        use_differential_attention: bool = True,
    ):
        super().__init__()

        dim_heads = embed_dim // num_heads

        # Timestep embedding.
        self.timestep_features = StableAudio3ExpoFourierFeatures(timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        # Learned embedding that replaces padded text positions in the cross-attention context.
        self.prompt_padding_embedding = nn.Parameter(torch.zeros(cond_token_dim))

        # Cross-attention context projection (text + duration tokens).
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

        # Global (duration) projection.
        self.to_global_embed = nn.Sequential(
            nn.Linear(global_cond_dim, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

        # AdaLN: expand the global conditioning into the 6 * embed_dim modulation signal.
        self.global_cond_embedder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True),
        )

        # Audio in/out: zero-initialised residual convs around the transformer.
        dim_in = io_channels * patch_size
        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

        self.proj_in = nn.Linear(dim_in, embed_dim, bias=False)
        self.proj_out = nn.Linear(embed_dim, dim_in, bias=False)

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, embed_dim) * 0.02)

        self.rotary_pos_emb = StableAudio3RotaryEmbedding(dim_heads // 2)

        self.transformer_blocks = nn.ModuleList(
            [
                StableAudio3DiTBlock(
                    dim=embed_dim,
                    context_dim=embed_dim,
                    dim_heads=dim_heads,
                    use_differential=use_differential_attention,
                    ff_mult=ff_mult,
                    local_add_cond_dim=local_add_cond_dim,
                )
                for _ in range(depth)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        local_add_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[StableAudio3DiTModelOutput, tuple]:
        """
        Args:
            hidden_states (`torch.Tensor`): Noisy latent audio `(batch, io_channels, T)`.
            timestep (`torch.Tensor`): Diffusion timestep `(batch,)` in `[0, 1]`.
            encoder_hidden_states (`torch.Tensor`): Cross-attention context `(batch, T_ctx, cond_token_dim)`.
            global_hidden_states (`torch.Tensor`): Global duration embedding `(batch, global_cond_dim)`.
            encoder_attention_mask (`torch.Tensor`, *optional*): Bool mask `(batch, T_ctx)`, `True` = valid.
            local_add_cond (`torch.Tensor`, *optional*): Local-additive (inpaint) conditioning
                `(batch, local_add_cond_dim, T)`.
            return_dict (`bool`, defaults to `True`): Whether to return a [`StableAudio3DiTModelOutput`].

        Returns:
            [`StableAudio3DiTModelOutput`] or `tuple`: the predicted velocity field, same shape as `hidden_states`.
        """
        batch_size = hidden_states.shape[0]

        # Replace padded text positions with the learned padding embedding and then attend to the
        # full context (the reference SA3 DiT disables the cross-attention mask). This must happen
        # in the raw `cond_token_dim` space, before `to_cond_embed`, to match the reference
        # conditioner which applies its learned padding embedding prior to the DiT projection.
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.bool().unsqueeze(-1)
            pad = self.prompt_padding_embedding.to(encoder_hidden_states.dtype).view(1, 1, -1)
            encoder_hidden_states = torch.where(mask, encoder_hidden_states, pad)
            encoder_attention_mask = None

        # Conditioning projections.
        context = self.to_cond_embed(encoder_hidden_states)

        timestep_embed = self.to_timestep_embed(
            self.timestep_features(timestep.float()[:, None]).to(hidden_states.dtype)
        )
        global_embed = self.to_global_embed(global_hidden_states) + timestep_embed
        global_modulation = self.global_cond_embedder(global_embed)  # (batch, 6 * embed_dim)

        # Preprocess (zero-init residual conv).
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states

        # Patch + project into the transformer dim.
        x = hidden_states.transpose(1, 2)  # (batch, T, io_channels)
        if self.config.patch_size > 1:
            B, T, C = x.shape
            x = x.reshape(B, T // self.config.patch_size, C * self.config.patch_size)
        x = self.proj_in(x)  # (batch, T, embed_dim)

        # Local-additive (inpaint) sequence, projected per-block inside the blocks.
        local_seq = None
        if local_add_cond is not None:
            if self.config.patch_size > 1:
                raise ValueError(
                    f"`local_add_cond` is not supported with `patch_size > 1` "
                    f"(got patch_size={self.config.patch_size}). Use patch_size=1 for inpainting."
                )
            local_seq = local_add_cond.transpose(1, 2)  # (batch, T, local_add_cond_dim)

        # Prepend memory tokens.
        if self.config.num_memory_tokens > 0:
            memory = self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([memory, x], dim=1)

        rope = self.rotary_pos_emb(x.shape[1], x.device)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(
                    block,
                    x,
                    context,
                    global_modulation,
                    rope,
                    encoder_attention_mask,
                    local_seq,
                    self.config.num_memory_tokens,
                )
            else:
                x = block(
                    x,
                    context=context,
                    global_modulation=global_modulation,
                    rope=rope,
                    context_mask=encoder_attention_mask,
                    local_seq=local_seq,
                    num_memory_tokens=self.config.num_memory_tokens,
                )

        # Remove memory tokens.
        if self.config.num_memory_tokens > 0:
            x = x[:, self.config.num_memory_tokens :]

        x = self.proj_out(x)

        # Unpatch.
        if self.config.patch_size > 1:
            B, T_p, CP = x.shape
            x = x.reshape(B, T_p * self.config.patch_size, CP // self.config.patch_size)

        output = x.transpose(1, 2)  # (batch, io_channels, T)
        output = self.postprocess_conv(output) + output

        if not return_dict:
            return (output,)
        return StableAudio3DiTModelOutput(sample=output)
