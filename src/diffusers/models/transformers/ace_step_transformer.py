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
"""Diffusion Transformer (DiT) for ACE-Step 1.5 music generation."""

import inspect
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    _maybe_download_kernel_for_backend,
    dispatch_attention_fn,
)
from ..cache_utils import CacheMixin
from ..embeddings import Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_FLASH_ATTENTION_BACKEND_TO_VARLEN = {
    AttentionBackendName.FLASH: AttentionBackendName.FLASH_VARLEN,
    AttentionBackendName.FLASH_HUB: AttentionBackendName.FLASH_VARLEN_HUB,
}


def _get_current_attention_backend(processor: Optional["AceStepAttnProcessor2_0"] = None) -> AttentionBackendName:
    backend = getattr(processor, "_attention_backend", None)
    if backend is None:
        backend, _ = _AttentionBackendRegistry.get_active_backend()
    return AttentionBackendName(backend)


def _is_flash_attention_backend(processor: Optional["AceStepAttnProcessor2_0"] = None) -> bool:
    return _get_current_attention_backend(processor) in _FLASH_ATTENTION_BACKEND_TO_VARLEN


# --------------------------------------------------------------------------- #
#                                attention-mask                                #
# --------------------------------------------------------------------------- #


def _create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> torch.Tensor:
    """Build a `[B, 1, seq_len, seq_len]` additive mask (0.0 kept, -inf masked).

    Mirrors the mask construction in ``acestep/models/turbo/modeling_acestep_v15_turbo.py::create_4d_mask`` so the DiT
    sees identical attention coverage regardless of whether SDPA, eager or flash attention is selected downstream.
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


# --------------------------------------------------------------------------- #
#                                 RoPE helpers                                 #
# --------------------------------------------------------------------------- #


def _ace_step_rotary_freqs(
    seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) freqs for ACE-Step RoPE using ``get_1d_rotary_pos_embed``.

    The original ACE-Step DiT reuses Qwen3's rotary layout: ``freqs = cat([freq_half, freq_half], dim=-1)`` (not
    interleaved), and the rotate-half convention splits the last dim in two halves rather than unbinding pairs. That
    matches ``get_1d_rotary_pos_embed(..., use_real=True, repeat_interleave_real=False)`` + ``apply_rotary_emb(...,
    use_real_unbind_dim=-2)``.
    """
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    cos, sin = get_1d_rotary_pos_embed(head_dim, positions, theta=theta, use_real=True, repeat_interleave_real=False)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


# --------------------------------------------------------------------------- #
#                                building blocks                               #
# --------------------------------------------------------------------------- #


class AceStepMLP(nn.Module):
    """SwiGLU MLP used in ACE-Step transformer blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AceStepTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP + 6-way AdaLN scale/shift projection.

    Matches the original ACE-Step checkpoint layout exactly (``linear_1``, ``linear_2``, ``time_proj``) so the
    converter maps keys 1:1. The sinusoid itself is the shared ``Timesteps`` module (``flip_sin_to_cos=True`` for
    ACE-Step's ``cat([cos, sin])`` convention).
    """

    def __init__(self, in_channels: int = 256, time_embed_dim: int = 2048, scale: float = 1000.0):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.time_sinusoid = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_freq = self.time_sinusoid(t * self.scale)
        temb = self.linear_1(t_freq.to(t.dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).unflatten(1, (6, -1))
        return temb, timestep_proj


class AceStepAttnProcessor2_0:
    """Attention processor for ACE-Step GQA attention.

    Dispatches the actual attention call through ``dispatch_attention_fn`` so users can pick flash / sage / native
    backends via ``model.set_attention_backend(...)`` or the ``attention_backend`` context manager. Uses the ``(B, L,
    H, D)`` tensor layout that the diffusers attention backends consume directly.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AceStepAttnProcessor2_0 requires PyTorch 2.0. Please upgrade your pytorch version.")

    @staticmethod
    def _padding_mask_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.ndim == 2:
            return attention_mask.to(torch.bool)
        if attention_mask.ndim == 4:
            keep_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
            return keep_mask.any(dim=(1, 2))
        raise ValueError(f"Unsupported ACE-Step attention mask shape for flash attention: {attention_mask.shape}")

    def __call__(
        self,
        attn: "AceStepAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        is_cross = attn.is_cross_attention and encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states

        # Project to (B, L, H, D). Q uses ``heads``; K/V use ``kv_heads`` (GQA).
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))
        key = attn.to_k(kv_input).unflatten(-1, (attn.kv_heads, attn.head_dim))
        value = attn.to_v(kv_input).unflatten(-1, (attn.kv_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # RoPE on self-attention only. Matches Qwen3 layout:
        # freqs = cat([freq_half, freq_half], dim=-1); rotate-half splits last dim.
        if not is_cross and image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2, sequence_dim=1)

        attention_kwargs = None
        backend = _get_current_attention_backend(self)
        dispatch_backend = self._attention_backend
        sliding_window = getattr(attn, "sliding_window", None)

        if backend in _FLASH_ATTENTION_BACKEND_TO_VARLEN:
            if attention_mask is not None:
                padding_mask = self._padding_mask_from_attention_mask(attention_mask)
                has_padding = not torch.all(padding_mask).item()
                attention_mask = None
                if has_padding:
                    dispatch_backend = _FLASH_ATTENTION_BACKEND_TO_VARLEN[backend]
                    _maybe_download_kernel_for_backend(dispatch_backend)
                    attention_mask = padding_mask

            if not is_cross and sliding_window is not None and key.shape[1] > sliding_window:
                # ACE-Step's dense mask keeps `abs(i - j) <= sliding_window`; flash-attn uses the same inclusive
                # left/right window convention, so pass the configured value through directly.
                attention_kwargs = {"window_size": (sliding_window, sliding_window)}

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=attn.dropout if attn.training else 0.0,
            scale=attn.scaling,
            enable_gqa=attn.heads != attn.kv_heads,
            attention_kwargs=attention_kwargs,
            backend=dispatch_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AceStepAttention(torch.nn.Module, AttentionModuleMixin):
    """GQA attention with RMSNorm on query/key for ACE-Step 1.5.

    Uses the diffusers ``Attention`` + ``AttnProcessor`` split: this module holds the projections and Q/K norm; the
    processor runs the attention dispatch. Self-attention applies RoPE on query/key; cross-attention reads K/V from
    ``encoder_hidden_states`` and does not apply RoPE.

    GQA means Q has ``heads * head_dim`` output while K/V have ``kv_heads * head_dim`` — QKV fusion is therefore
    disabled (``_supports_qkv_fusion = False``).
    """

    _default_processor_cls = AceStepAttnProcessor2_0
    _available_processors = [AceStepAttnProcessor2_0]
    _supports_qkv_fusion = False

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
        eps: float = 1e-6,
        sliding_window: Optional[int] = None,
        is_cross_attention: bool = False,
        processor: Optional[AceStepAttnProcessor2_0] = None,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scaling = head_dim**-0.5
        self.sliding_window = sliding_window
        self.is_cross_attention = is_cross_attention

        self.to_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.to_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.to_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias), nn.Dropout(0.0)]
        )
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in attn_parameters}
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


class AceStepTransformerBlock(nn.Module):
    """ACE-Step DiT transformer block: self-attn (AdaLN) → cross-attn → MLP (AdaLN).

    AdaLN parameters come from the shared ``scale_shift_table + timestep_proj`` chunked into 6 (3 for self-attn + 3 for
    MLP).
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
        self.self_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
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

        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.cross_attn = AceStepAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                bias=attention_bias,
                dropout=attention_dropout,
                eps=rms_norm_eps,
                is_cross_attention=True,
            )

        self.mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = AceStepMLP(hidden_size, intermediate_size)

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

        # Self-attention with AdaLN.
        norm_hidden_states = (self.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.self_attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        if self.use_cross_attention and encoder_hidden_states is not None:
            norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(hidden_states)
            attn_output = self.cross_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.mlp_norm(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.mlp(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
#                                 main DiT model                               #
# --------------------------------------------------------------------------- #


class AceStepTransformer1DModel(ModelMixin, ConfigMixin, AttentionMixin, CacheMixin):
    """Diffusion Transformer for ACE-Step 1.5 music generation.

    Generates audio latents conditioned on text, lyrics, and timbre. Uses 1D patch embedding (`Conv1d` with stride
    `patch_size`) followed by a stack of `AceStepTransformerBlock`s with alternating sliding-window / full attention on
    the self-attention branch. Cross-attention consumes the packed `encoder_hidden_states` produced by
    `AceStepConditionEncoder`.
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
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 128,
        layer_types: Optional[List[str]] = None,
        # Dim of the condition encoder's output. Equal to `hidden_size` on the
        # non-XL turbo / base models, but the XL turbo has a smaller condition
        # encoder (`encoder_hidden_size=2048`) feeding a wider DiT
        # (`hidden_size=2560`), so `condition_embedder` needs to project it up.
        encoder_hidden_size: Optional[int] = None,
        # Variant metadata. Turbo models have guidance distilled into the weights and
        # should run without CFG; base/SFT models require CFG with the learned
        # `AceStepConditionEncoder.null_condition_emb`. The pipeline reads these to
        # pick default `guidance_scale`, `shift`, and `num_inference_steps`.
        is_turbo: bool = False,
        model_version: Optional[str] = None,
    ):
        super().__init__()
        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size
        self.patch_size = patch_size
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(num_hidden_layers)
            ]
        self.layer_types = list(layer_types)

        self.layers = nn.ModuleList(
            [
                AceStepTransformerBlock(
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

        # Patchify: concat(src_latents, chunk_mask) on the channel dim then Conv1d with
        # stride=patch_size lifts (B, T, in_channels) -> (B, T/patch_size, hidden_size).
        self.proj_in_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Dual-timestep conditioning: one path for `t`, one for `(t - r)` (mean-flow).
        self.time_embed = AceStepTimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.time_embed_r = AceStepTimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)

        self.condition_embedder = nn.Linear(encoder_hidden_size, hidden_size, bias=True)

        self.norm_out = RMSNorm(hidden_size, eps=rms_norm_eps)
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
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """The [`AceStepTransformer1DModel`] forward method.

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
                Context latents (source latents concatenated with chunk masks) — fed to the patchify conv alongside
                `hidden_states`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a `Transformer2DModelOutput` or a plain tuple.

        Returns:
            `Transformer2DModelOutput` or `tuple`: The predicted velocity field.
        """
        # Dual timestep embedding: t and (t - r). Sum both paths' AdaLN projections.
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Context concatenation + padding to patch_size boundary + patchify.
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden_states.shape[1]
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode="constant", value=0)
        hidden_states = self.proj_in_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device

        cos, sin = _ace_step_rotary_freqs(seq_len, self.head_dim, self.rope_theta, device, dtype)
        position_embeddings = (cos, sin)

        sliding_attn_mask = None
        if not _is_flash_attention_backend():
            sliding_attn_mask = _create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                sliding_window=self.config.sliding_window,
                is_sliding_window=True,
                is_causal=False,
            )

        for i, layer_module in enumerate(self.layers):
            # Full-attention layers see no mask; only the sliding-attention layers
            # need the banded mask. Cross-attention uses no padding mask.
            layer_attn_mask = sliding_attn_mask if self.layer_types[i] == "sliding_attention" else None

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module,
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    layer_attn_mask,
                    encoder_hidden_states,
                    None,
                )
            else:
                hidden_states = layer_module(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    temb=timestep_proj,
                    attention_mask=layer_attn_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                )

        # Adaptive output normalization + de-patchify.
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states[:, :original_seq_len, :]

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)
