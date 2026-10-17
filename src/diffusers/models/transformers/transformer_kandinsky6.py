# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""Kandinsky 6 Diffusers transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import BaseOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin, get_parameter_dtype


@dataclass
class Kandinsky6TransformerOutput(BaseOutput):
    r"""
    The output of [`Kandinsky6Transformer3DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_frames, height, width, out_visual_dim)`):
            The predicted video velocity.
        audio_sample (`torch.Tensor` of shape `(batch_size, audio_length, out_audio_dim)`, *optional*):
            The predicted audio velocity, `None` when the model was called without `audio_hidden_states`.
    """

    sample: torch.Tensor
    audio_sample: torch.Tensor | None = None


def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    """Return inverse frequencies for rotary position embeddings.

    Args:
        dim (`int`): Number of frequency values to generate.
        max_period (`float`, *optional*, defaults to 10000.0): Maximum period
            used by the frequency schedule.

    Returns:
        `torch.Tensor`: Frequency values in float32.
    """
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def apply_scale_shift(normed: Tensor, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply an AdaLN-style scale/shift affine to an already-normalized tensor, in fp32, cast back to ``x.dtype``.

    Callers compute ``normed`` themselves (e.g. ``self.some_norm(x.float())``) so the norm-layer call stays visible in
    `forward` instead of being hidden inside this helper.
    """
    if x.ndim > 2 and scale.ndim == 2:
        shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[-1],)
        scale, shift = scale.reshape(shape), shift.reshape(shape)
    return (normed * (scale.float() + 1.0) + shift.float()).to(dtype=x.dtype)


def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor) -> Tensor:
    """Residual gate in fp32, cast back to ``x.dtype``."""
    if x.ndim > 2 and gate.ndim == 2:
        gate = gate.reshape((gate.shape[0],) + (1,) * (x.ndim - 2) + (gate.shape[-1],))
    return (x.float() + gate.float() * out.float()).to(dtype=x.dtype)


def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """RoPE apply in fp32 (rope tables are fp32), cast back to ``x.dtype``."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope.float() * x_).sum(dim=-1).reshape(*x.shape).to(dtype=x.dtype)


class Kandinsky6RoPE1D(nn.Module):
    """1-D Rotary Position Embedding — used for text and audio sequences."""

    def __init__(
        self,
        dim: int,
        max_pos: int = 2048,
        max_period: float = 10000.0,
        freqs_scaling: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.max_period = max_period
        self.freqs_scaling = freqs_scaling
        freq = get_freqs(dim // 2, max_period) * freqs_scaling
        self.register_buffer("angles", torch.outer(torch.arange(max_pos, dtype=freq.dtype), freq), persistent=False)

    def forward(self, pos: Tensor) -> Tensor:
        # RoPE tables are fp32; keep trig in fp32.
        angles = self.angles[pos]  # (seq_len, dim//2)
        rope = torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)], dim=-1)
        return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)


class Kandinsky6RoPE3D(nn.Module):
    """3-D Rotary Position Embedding — used for video spatial-temporal tokens (T, H, W)."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period
        for i, (d, mp) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(d // 2, max_period)
            self.register_buffer(
                f"angles_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype), freq), persistent=False
            )

    def forward(
        self,
        pos: tuple[Tensor, Tensor, Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        # `pos` holds one index tensor per axis; the grid size is implied by their lengths.
        num_frames, height, width = (int(axis_pos.shape[0]) for axis_pos in pos)
        angles_t = self.angles_0[pos[0]] / scale_factor[0]  # (T, d//2)
        angles_h = self.angles_1[pos[1]] / scale_factor[1]  # (H, d//2)
        angles_w = self.angles_2[pos[2]] / scale_factor[2]  # (W, d//2)

        angles = torch.cat(
            [
                angles_t.view(num_frames, 1, 1, -1).expand(num_frames, height, width, -1),
                angles_h.view(1, height, 1, -1).expand(num_frames, height, width, -1),
                angles_w.view(1, 1, width, -1).expand(num_frames, height, width, -1),
            ],
            dim=-1,
        )
        cos, sin = torch.cos(angles), torch.sin(angles)
        rope = torch.stack([cos, -sin, sin, cos], dim=-1)  # (T, H, W, total_dim, 4)
        rope = rope.view(*rope.shape[:-1], 2, 2)  # (T, H, W, total_dim, 2, 2)
        return rope.unsqueeze(-4)  # (T, H, W, 1, total_dim, 2, 2)


class Kandinsky6AttnProcessor:
    """Diffusers attention processor used by the TI2VA transformer."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, attention_backend=None, parallel_config=None):
        self._attention_backend = attention_backend
        self._parallel_config = parallel_config

    def __call__(
        self,
        attn: Any,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        rotary_emb: Tensor | None = None,
        rotary_emb_kv: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        query = attn.to_query(hidden_states)
        if encoder_hidden_states is None:
            key = attn.to_key(hidden_states)
            value = attn.to_value(hidden_states)
        else:
            key = attn.to_key(encoder_hidden_states)
            value = attn.to_value(encoder_hidden_states)

        query = query.reshape(*query.shape[:-1], attn.num_heads, -1)
        key = key.reshape(*key.shape[:-1], attn.num_heads, -1)
        value = value.reshape(*value.shape[:-1], attn.num_heads, -1)
        query = attn.query_norm(query)
        key = attn.key_norm(key)

        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb).to(dtype=query.dtype)
        if rotary_emb_kv is not None:
            key = apply_rotary(key, rotary_emb_kv).to(dtype=key.dtype)

        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        return attn.out_layer(output.flatten(-2, -1))


class Kandinsky6TimeEmbeddings(nn.Module):
    """Sinusoidal timestep embedding with a K6-compatible parameter layout."""

    def __init__(self, model_dim: int, time_dim: int):
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even")
        self.time_proj = Timesteps(model_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(model_dim, time_dim, act_fn="silu")

    def forward(self, timestep: Tensor) -> Tensor:
        # The sinusoidal embedding is float32; `_keep_in_fp32_modules` keeps these layers float32 under
        # `from_pretrained(torch_dtype=...)`, and the cast aligns the input with whatever dtype they hold.
        embed = self.time_proj(timestep)
        embed = embed.to(get_parameter_dtype(self.timestep_embedder))
        return self.timestep_embedder(embed)


class Kandinsky6TextEmbeddings(nn.Module):
    """Text projection and normalization used by K6 text branches."""

    def __init__(self, text_dim: int, model_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.in_layer(x))


class Kandinsky6VisualEmbeddings(nn.Module):
    """Patch projection for ``[B,T,H,W,C]`` visual tokens."""

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, duration, height, width, channels = x.shape
        p_t, p_h, p_w = self.patch_size
        x = (
            x.view(batch, duration // p_t, p_t, height // p_h, p_h, width // p_w, p_w, channels)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


class Kandinsky6Modulation(nn.Module):
    """AdaLN modulation projection."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(x.to(get_parameter_dtype(self.out_layer))))


class Kandinsky6Attention(nn.Module, AttentionModuleMixin):
    """K6 attention with the Diffusers ``set_processor`` contract."""

    _default_processor_cls = Kandinsky6AttnProcessor
    _available_processors = [Kandinsky6AttnProcessor]

    def __init__(
        self,
        num_channels: int,
        head_dim: int,
        kv_dim: int | None = None,
        processor: Kandinsky6AttnProcessor | None = None,
    ):
        super().__init__()
        if num_channels % head_dim:
            raise ValueError("num_channels must be divisible by head_dim")
        kv_dim = kv_dim or num_channels
        self.num_heads = num_channels // head_dim
        self.to_query = nn.Linear(num_channels, num_channels)
        self.to_key = nn.Linear(kv_dim, num_channels)
        self.to_value = nn.Linear(kv_dim, num_channels)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels)
        self.set_processor(processor or self._default_processor_cls())

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rotary_emb: Tensor | None = None,
        rope_q: Tensor | None = None,
        rope_kv: Tensor | None = None,
    ) -> Tensor:
        # Native K6 blocks pass ``(hidden, rope, mask)`` for self-attention. Keep that call shape while exposing
        # Diffusers' encoder_hidden_states/rotary_emb keyword boundary.
        rotary_emb = rope_q if rope_q is not None else rotary_emb
        rotary_emb_kv = rope_kv if rope_kv is not None else rotary_emb
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=rotary_emb,
            rotary_emb_kv=rotary_emb_kv,
            attn_mask=attn_mask,
        )


class Kandinsky6OutLayer(nn.Module):
    """Projects visual hidden states back to packed latent patches."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim)

    def forward(self, visual_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        condition_shape = (scale.shape[0],) + (1,) * (visual_embed.ndim - 2) + (scale.shape[-1],)
        x = apply_scale_shift(
            self.norm(visual_embed.float()),
            visual_embed,
            scale.reshape(condition_shape),
            shift.reshape(condition_shape),
        )
        x = self.out_layer(x)

        batch, duration, height, width = x.shape[:4]
        p_t, p_h, p_w = self.patch_size
        return (
            x.view(batch, duration, height, width, -1, p_t, p_h, p_w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )


class Kandinsky6OutLayerAudio(nn.Module):
    """Projects audio hidden states back to audio latent channels."""

    def __init__(self, model_dim: int, time_dim: int, audio_dim: int):
        super().__init__()
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, audio_dim)

    def forward(self, audio_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        x = apply_scale_shift(self.norm(audio_embed.float()), audio_embed, scale, shift)
        # The reference audio head normalizes a second time after the AdaLN affine; the released weights were
        # trained this way, so the double norm is kept for parity even though the video head has none.
        x = self.norm(x)
        return self.out_layer(x)


class Kandinsky6TransformerEncoderBlock(nn.Module):
    """Text self-attention + feed-forward block in Diffusers style."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
    ):
        super().__init__()
        self.text_modulation = Kandinsky6Modulation(time_dim, model_dim, 6)
        self.attn_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.attn = Kandinsky6Attention(model_dim, head_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, inner_dim=ff_dim, activation_fn="gelu", bias=False)

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        sa_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(sa_params, 3, dim=-1)
        x = apply_gate_sum(
            x,
            self.attn(
                apply_scale_shift(self.attn_norm(x.float()), x, scale, shift),
                rotary_emb=rope,
                attn_mask=attn_mask,
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        return apply_gate_sum(
            x, self.feed_forward(apply_scale_shift(self.feed_forward_norm(x.float()), x, scale, shift)), gate
        )


class Kandinsky6TransformerDecoderBlock(nn.Module):
    """Visual self-attention, text cross-attention, and feed-forward submodules.

    `Kandinsky6FusedTransformerDecoderBlock` uses this class only as a named submodule container
    (`self_attention`/`cross_attention`/`feed_forward` and their norms/modulation), calling those submodules directly
    rather than this class's own `forward`.
    """

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
    ):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky6Attention(model_dim, head_dim)
        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            kv_dim=model_dim,
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, inner_dim=ff_dim, activation_fn="gelu", bias=False)


class Kandinsky6FusedTransformerDecoderBlock(nn.Module):
    """Fused K6 video/audio block with cross-modal attention."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        model_dim_a: int,
        time_dim_a: int,
        ff_dim_a: int,
        head_dim_a: int,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
    ):
        super().__init__()
        self.video_dec_block = Kandinsky6TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
        self.audio_dec_block = Kandinsky6TransformerDecoderBlock(model_dim_a, time_dim_a, ff_dim_a, head_dim_a)
        self.va_cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            kv_dim=model_dim_a,
        )
        self.av_cross_attention = Kandinsky6Attention(
            model_dim_a,
            head_dim_a,
            kv_dim=model_dim,
        )
        self.va_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim if not cross_gates else model_dim * 2 + model_dim_a,
            1 if cross_gates else 3,
        )
        self.av_modulation = Kandinsky6Modulation(
            time_dim_a,
            model_dim_a if not cross_gates else model_dim_a * 2 + model_dim,
            1 if cross_gates else 3,
        )
        self.va_normalization = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.av_normalization = nn.LayerNorm(model_dim_a, elementwise_affine=False)
        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def forward(
        self,
        vis: Tensor | None,
        aud: Tensor | None,
        text_v: Tensor,
        text_a: Tensor,
        time_embed: tuple[Tensor, Tensor],
        vis_rope: Tensor | None,
        aud_rope: Tensor | None,
        attn_mask=None,
    ) -> tuple[Tensor | None, Tensor | None]:
        t_v, t_a = time_embed
        if vis is not None:
            sa_p, ca_p, ff_p = torch.chunk(self.video_dec_block.visual_modulation(t_v), 3, dim=-1)
            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            vis = apply_gate_sum(
                vis,
                self.video_dec_block.self_attention(
                    apply_scale_shift(self.video_dec_block.self_attention_norm(vis.float()), vis, scale, shift),
                    rotary_emb=vis_rope,
                ),
                gate,
            ).type_as(vis)
            shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
            vis_pre_ca = apply_scale_shift(self.video_dec_block.cross_attention_norm(vis.float()), vis, scale, shift)
            vis_out_t = self.video_dec_block.cross_attention(
                vis_pre_ca,
                encoder_hidden_states=text_v,
                attn_mask=attn_mask,
            )

        if aud is not None:
            sa_p, ca_p, ff_p_a = torch.chunk(self.audio_dec_block.visual_modulation(t_a), 3, dim=-1)
            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            aud = apply_gate_sum(
                aud,
                self.audio_dec_block.self_attention(
                    apply_scale_shift(self.audio_dec_block.self_attention_norm(aud.float()), aud, scale, shift),
                    rotary_emb=aud_rope,
                ),
                gate,
            ).type_as(aud)
            shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
            aud_pre_ca = apply_scale_shift(self.audio_dec_block.cross_attention_norm(aud.float()), aud, scale, shift)
            aud_out_t = self.audio_dec_block.cross_attention(
                aud_pre_ca,
                encoder_hidden_states=text_a,
                attn_mask=attn_mask,
            )
            aud = apply_gate_sum(aud, aud_out_t, gate_a).type_as(aud)

            if vis is not None:
                t_va_mod = t_a if not self.fix_modulation else t_v
                t_av_mod = t_v if not self.fix_modulation else t_a
                va_params = self.va_modulation(t_va_mod)
                av_params = self.av_modulation(t_av_mod)
                if self.cross_gates:
                    va_shift, va_scale, va_gate = torch.split(
                        va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1
                    )
                    av_shift, av_scale, av_gate = torch.split(
                        av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1
                    )
                else:
                    va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
                    av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)
                vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)
                vis_for_va = apply_scale_shift(self.va_normalization(vis.float()), vis, va_scale, va_shift)
                aud_for_av = apply_scale_shift(self.av_normalization(aud.float()), aud, av_scale, av_shift)
                rq_v = vis_rope if self.ca_rope else None
                rk_a = aud_rope if self.ca_rope else None
                vis_from_aud = self.va_cross_attention(
                    vis_for_va,
                    encoder_hidden_states=aud_pre_ca,
                    rope_q=rq_v,
                    rope_kv=rk_a,
                )
                aud_from_vis = self.av_cross_attention(
                    aud_for_av,
                    encoder_hidden_states=vis_pre_ca,
                    rope_q=rk_a,
                    rope_kv=rq_v,
                )
                vis = apply_gate_sum(vis, vis_from_aud, va_gate if not self.cross_gates else av_gate).type_as(vis)
                aud = apply_gate_sum(aud, aud_from_vis, av_gate if not self.cross_gates else va_gate).type_as(aud)
        elif vis is not None:
            vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)

        if vis is not None:
            shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
            vis = apply_gate_sum(
                vis,
                self.video_dec_block.feed_forward(
                    apply_scale_shift(self.video_dec_block.feed_forward_norm(vis.float()), vis, scale, shift)
                ),
                gate,
            ).type_as(vis)
        if aud is not None:
            shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
            aud = apply_gate_sum(
                aud,
                self.audio_dec_block.feed_forward(
                    apply_scale_shift(self.audio_dec_block.feed_forward_norm(aud.float()), aud, scale, shift)
                ),
                gate,
            ).type_as(aud)
        return vis, aud


class Kandinsky6Transformer3DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    CacheMixin,
    AttentionMixin,
):
    """Kandinsky 6 multimodal transformer for text/image-to-video-and-audio generation.

    Video and audio are denoised together through fused, cross-modal transformer blocks, each conditioned on its own
    text branch (Qwen2.5-VL tokens + CLIP pooled embedding). Passing no `audio_hidden_states` denoises video alone
    while still running the fused block's video self-attention, cross-attention, and feed-forward stages. Rotary
    embeddings are computed inside `forward` from the token grid.

    Args:
        in_visual_dim (`int`, *optional*, defaults to 16): Number of input video latent channels.
        out_visual_dim (`int`, *optional*, defaults to 16): Number of output video latent channels.
        in_text_dim (`int`, *optional*, defaults to 3584): Text token embedding dimension.
        in_text_dim2 (`int`, *optional*, defaults to 768): Pooled text embedding dimension.
        time_dim (`int`, *optional*, defaults to 1024): Time embedding dimension.
        patch_size (`tuple[int, int, int]`, *optional*, defaults to ``(1, 2, 2)``): Video patch size.
        model_dim (`int`, *optional*, defaults to 4096): Video transformer hidden dimension.
        ff_dim (`int`, *optional*, defaults to 16384): Video feed-forward hidden dimension.
        num_text_blocks (`int`, *optional*, defaults to 4): Number of text blocks per modality.
        num_visual_blocks (`int`, *optional*, defaults to 60): Number of fused video/audio blocks.
        axes_dims (`tuple[int, int, int]`, *optional*, defaults to ``(32, 48, 48)``): RoPE dimensions for video.
        visual_cond (`bool`, *optional*, defaults to True): Whether video conditioning channels are present.
        in_audio_dim (`int`, *optional*, defaults to 20): Number of input audio latent channels.
        out_audio_dim (`int`, *optional*, defaults to 20): Number of output audio latent channels.
        model_dim_a (`int`, *optional*): Audio transformer hidden dimension. Defaults to `model_dim`.
        time_dim_a (`int`, *optional*): Audio time embedding dimension. Defaults to `time_dim`.
        ff_dim_a (`int`, *optional*): Audio feed-forward hidden dimension. Defaults to `ff_dim`.
        axes_dims_a (`tuple[int, int, int]`, *optional*): Audio RoPE dimensions. Defaults to `axes_dims`.
        audio_freqs_scaling (`float`, *optional*, defaults to 1.0): Audio RoPE frequency scaling.
        scale_factor (`tuple[float, float, float]`, *optional*, defaults to `(1.0, 2.0, 2.0)`): Per-axis
            `(t, h, w)` RoPE frequency scaling applied to the video positions.
        text_token_padding (`bool`, *optional*, defaults to False): Checkpoint metadata recording whether the
            reference model's text sequences are padded. `forward` always accepts an optional `encoder_attention_mask`
            regardless of this flag; whether one is actually passed is entirely up to the caller.
        ca_rope (`bool`, *optional*, defaults to False): Whether to use cross-modal audio RoPE.
        cross_gates (`bool`, *optional*, defaults to False): Whether to use cross-modal residual gates.
        fix_modulation (`bool`, *optional*, defaults to False): Whether to use the fixed modulation variant.
        visual_token_type_num_embeddings (`int`, *optional*, defaults to 0): Number of visual token type embeddings.

    Released checkpoints set `text_token_padding`, `ca_rope`, `cross_gates`, and `fix_modulation` to `True` (see each
    checkpoint's `transformer/config.json`); the `False` defaults only describe an architecture variant this repo does
    not ship weights for.
    """

    _repeated_blocks = [
        "Kandinsky6TransformerEncoderBlock",
        "Kandinsky6FusedTransformerDecoderBlock",
    ]
    _no_split_modules = _repeated_blocks
    _skip_layerwise_casting_patterns = ["norm"]
    # Timestep embeddings and every AdaLN modulation projection (`*_modulation`, `out_layer.modulation`) are
    # precision-sensitive and stay float32 under `from_pretrained(torch_dtype=...)`.
    _keep_in_fp32_modules = ["time_embeddings", "modulation"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 1024,
        patch_size: tuple = (1, 2, 2),
        model_dim: int = 4096,
        ff_dim: int = 16384,
        num_text_blocks: int = 4,
        num_visual_blocks: int = 60,
        axes_dims: tuple = (32, 48, 48),
        visual_cond: bool = True,
        in_audio_dim: int = 20,
        out_audio_dim: int = 20,
        model_dim_a: int | None = None,
        time_dim_a: int | None = None,
        ff_dim_a: int | None = None,
        axes_dims_a: tuple | None = None,
        audio_freqs_scaling: float = 1.0,
        scale_factor: tuple | list[float] = (1.0, 2.0, 2.0),
        text_token_padding: bool = False,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        visual_token_type_num_embeddings: int = 0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.in_visual_dim = in_visual_dim
        self.in_audio_dim = in_audio_dim
        self.text_token_padding = text_token_padding
        self.scale_factor = tuple(float(value) for value in scale_factor)
        self.visual_token_type_num_embeddings = int(visual_token_type_num_embeddings or 0)
        head_dim = sum(axes_dims)
        model_dim_a = model_dim_a or model_dim
        time_dim_a = time_dim_a or time_dim
        ff_dim_a = ff_dim_a or ff_dim
        axes_dims_a = axes_dims_a or axes_dims
        head_dim_a = sum(axes_dims_a)

        vis_in_dim = (2 * in_visual_dim + 1) if visual_cond else in_visual_dim
        self.visual_embeddings = Kandinsky6VisualEmbeddings(vis_in_dim, model_dim, patch_size)
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(self.visual_token_type_num_embeddings, model_dim)
        self.visual_rope_embeddings = Kandinsky6RoPE3D(axes_dims)
        self.out_layer = Kandinsky6OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

        self.audio_embeddings = Kandinsky6TextEmbeddings(in_audio_dim, model_dim_a)
        self.audio_rope_embeddings = Kandinsky6RoPE1D(head_dim_a, freqs_scaling=audio_freqs_scaling)
        self.audio_out_layer = Kandinsky6OutLayerAudio(model_dim_a, time_dim_a, out_audio_dim)
        for prefix, md, td, fd, hd in (
            ("video", model_dim, time_dim, ff_dim, head_dim),
            ("audio", model_dim_a, time_dim_a, ff_dim_a, head_dim_a),
        ):
            setattr(self, f"{prefix}_time_embeddings", Kandinsky6TimeEmbeddings(md, td))
            setattr(self, f"{prefix}_text_embeddings", Kandinsky6TextEmbeddings(in_text_dim, md))
            setattr(self, f"{prefix}_pooled_text_embeddings", Kandinsky6TextEmbeddings(in_text_dim2, td))
            setattr(self, f"{prefix}_text_rope_embeddings", Kandinsky6RoPE1D(hd))
            setattr(
                self,
                f"{prefix}_text_transformer_blocks",
                nn.ModuleList([Kandinsky6TransformerEncoderBlock(md, td, fd, hd) for _ in range(num_text_blocks)]),
            )
        self.visual_transformer_blocks = nn.ModuleList(
            [
                Kandinsky6FusedTransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    model_dim_a,
                    time_dim_a,
                    ff_dim_a,
                    head_dim_a,
                    ca_rope,
                    cross_gates,
                    fix_modulation,
                )
                for _ in range(num_visual_blocks)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        audio_hidden_states: Tensor | None = None,
        visual_rope_pos: tuple[Tensor, Tensor, Tensor] | None = None,
        encoder_attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
        return_dict: bool = True,
    ) -> Kandinsky6TransformerOutput | tuple[Tensor, ...]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_frames, height, width, in_channels)`):
                Video latents in the `(B, T, H, W, C)` layout. With `visual_cond=True`, `in_channels` is `2 *
                in_visual_dim + 1`: the noisy latent, the conditioning latent and a conditioning mask.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, in_text_dim)`):
                Text token embeddings, shared by the video and audio text branches.
            pooled_projections (`torch.Tensor` of shape `(batch_size, in_text_dim2)`):
                Pooled text embedding added to the timestep embedding of both branches.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion timestep on the `[0, num_train_timesteps]` scale, shared by both modalities.
            audio_hidden_states (`torch.Tensor` of shape `(batch_size, audio_length, in_audio_dim)`, *optional*):
                Audio latents. When omitted the fused blocks run their video path only.
            visual_rope_pos (`tuple[torch.Tensor, torch.Tensor, torch.Tensor]`, *optional*):
                Per-axis `(t, h, w)` rotary position indices of the patchified video tokens. Defaults to `arange` over
                each axis. The pipeline passes an explicit temporal index when it appends a reference frame that reuses
                position `0`.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Boolean padding mask for `encoder_hidden_states`. Pass `None` when no prompt is padded.
            visual_token_type_ids (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
                Per-frame token type ids, embedded through `visual_token_type_embeddings`. Requires
                `visual_token_type_num_embeddings > 0`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`Kandinsky6TransformerOutput`] instead of a plain tuple.

        Returns:
            [`Kandinsky6TransformerOutput`] or `tuple`:
                The predicted video velocity in the `(B, T, H, W, out_visual_dim)` layout and, when
                `audio_hidden_states` was given, the predicted audio velocity of shape `(B, audio_length,
                out_audio_dim)`.
        """
        if visual_token_type_ids is not None and self.visual_token_type_num_embeddings == 0:
            raise ValueError("`visual_token_type_ids` requires `visual_token_type_num_embeddings > 0`")
        checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing
        device = hidden_states.device

        # 1. Rotary embeddings for the two text branches
        text_pos = torch.arange(encoder_hidden_states.shape[1], device=device)
        video_text_rope = self.video_text_rope_embeddings(text_pos)
        audio_text_rope = self.audio_text_rope_embeddings(text_pos)

        # 2. Encode the text tokens through the video and audio text branches
        video_text_embed = self.video_text_embeddings(encoder_hidden_states)
        video_temb = self.video_time_embeddings(timestep) + self.video_pooled_text_embeddings(pooled_projections)
        for block in self.video_text_transformer_blocks:
            args = (video_text_embed, video_temb, video_text_rope, encoder_attention_mask)
            video_text_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

        audio_text_embed = self.audio_text_embeddings(encoder_hidden_states)
        audio_temb = self.audio_time_embeddings(timestep) + self.audio_pooled_text_embeddings(pooled_projections)
        for block in self.audio_text_transformer_blocks:
            args = (audio_text_embed, audio_temb, audio_text_rope, encoder_attention_mask)
            audio_text_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

        # 3. Patchify the video latents and build their rotary embeddings
        visual_embed = self.visual_embeddings(hidden_states)
        if visual_token_type_ids is not None:
            token_types = self.visual_token_type_embeddings(visual_token_type_ids)
            visual_embed = visual_embed + token_types[:, :, None, None, :]
        visual_shape = visual_embed.shape[1:4]
        if visual_rope_pos is None:
            visual_rope_pos = tuple(torch.arange(size, device=device) for size in visual_shape)
        visual_rope = self.visual_rope_embeddings(visual_rope_pos, self.scale_factor)
        visual_embed = visual_embed.flatten(1, 3)
        visual_rope = visual_rope.flatten(0, 2)

        # 4. Embed the audio latents and build their rotary embeddings
        audio_embed = audio_rope = None
        if audio_hidden_states is not None:
            audio_embed = self.audio_embeddings(audio_hidden_states)
            audio_rope = self.audio_rope_embeddings(torch.arange(audio_hidden_states.shape[1], device=device))

        # 5. Run the fused video/audio transformer blocks (a block skips the audio path when it is absent)
        for block in self.visual_transformer_blocks:
            args = (
                visual_embed,
                audio_embed,
                video_text_embed,
                audio_text_embed,
                (video_temb, audio_temb),
                visual_rope,
                audio_rope,
                encoder_attention_mask,
            )
            visual_embed, audio_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

        # 6. Project back to the latent space
        visual_embed = visual_embed.reshape(-1, *visual_shape, visual_embed.shape[-1])
        video_out = self.out_layer(visual_embed, video_temb)
        audio_out = self.audio_out_layer(audio_embed, audio_temb) if audio_embed is not None else None

        if not return_dict:
            return (video_out,) if audio_out is None else (video_out, audio_out)
        return Kandinsky6TransformerOutput(sample=video_out, audio_sample=audio_out)
