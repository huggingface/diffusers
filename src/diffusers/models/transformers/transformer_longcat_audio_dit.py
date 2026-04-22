# Copyright 2026 MeiTuan LongCat-AudioDiT Team and The HuggingFace Team. All rights reserved.
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

# Adapted from the LongCat-AudioDiT reference implementation:
# https://github.com/meituan-longcat/LongCat-AudioDiT

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.torch_utils import lru_cache_unless_export, maybe_allow_in_graph
from ..attention import AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


@dataclass
class LongCatAudioDiTTransformerOutput(BaseOutput):
    sample: torch.Tensor


class AudioDiTSinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device).float() * -exponent)
        embeddings = scale * timesteps.unsqueeze(1) * embeddings.unsqueeze(0)
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class AudioDiTTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = AudioDiTSinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        hidden_states = self.time_embed(timestep)
        return self.time_mlp(hidden_states.to(timestep.dtype))


class AudioDiTRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 100000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    @lru_cache_unless_export(maxsize=128)
    def _build(self, seq_len: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        if device is not None:
            inv_freq = inv_freq.to(device)
        steps = torch.arange(seq_len, dtype=torch.int64, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.outer(steps, inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        return embeddings.cos().contiguous(), embeddings.sin().contiguous()

    def forward(self, hidden_states: torch.Tensor, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = hidden_states.shape[1] if seq_len is None else seq_len
        cos, sin = self._build(max(seq_len, self.max_position_embeddings), hidden_states.device)
        return cos[:seq_len].to(dtype=hidden_states.dtype), sin[:seq_len].to(dtype=hidden_states.dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary_emb(hidden_states: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rope
    cos = cos[None, :, None].to(hidden_states.device)
    sin = sin[None, :, None].to(hidden_states.device)
    return (hidden_states.float() * cos + _rotate_half(hidden_states).float() * sin).to(hidden_states.dtype)


class AudioDiTGRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(hidden_states, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (hidden_states * nx) + self.beta + hidden_states


class AudioDiTConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        kernel_size: int = 7,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, dilation=dilation, bias=bias
        )
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, intermediate_dim, bias=bias)
        self.act = nn.SiLU()
        self.grn = AudioDiTGRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.grn(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        return residual + hidden_states


class AudioDiTEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, hidden_states: torch.Tensor, mask: torch.BoolTensor | None = None) -> torch.Tensor:
        if mask is not None:
            hidden_states = hidden_states.masked_fill(mask.logical_not().unsqueeze(-1), 0.0)
        hidden_states = self.proj(hidden_states)
        if mask is not None:
            hidden_states = hidden_states.masked_fill(mask.logical_not().unsqueeze(-1), 0.0)
        return hidden_states


class AudioDiTAdaLNMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, out_dim, bias=bias))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class AudioDiTAdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, hidden_states: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        embedding = self.linear(self.silu(embedding))
        scale, shift = torch.chunk(embedding, 2, dim=-1)
        hidden_states = self.norm(hidden_states.float()).type_as(hidden_states)
        if scale.ndim == 2:
            hidden_states = hidden_states * (1 + scale)[:, None, :] + shift[:, None, :]
        else:
            hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class AudioDiTSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "AudioDiTAttention",
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
        audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if attn.qk_norm:
            query = attn.q_norm(query)
            key = attn.k_norm(key)

        head_dim = attn.inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if audio_rotary_emb is not None:
            query = _apply_rotary_emb(query, audio_rotary_emb)
            key = _apply_rotary_emb(key, audio_rotary_emb)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None, None].to(hidden_states.dtype)

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AudioDiTAttention(nn.Module, AttentionModuleMixin):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int | None,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
        processor: AttentionModuleMixin | None = None,
    ):
        super().__init__()
        kv_dim = q_dim if kv_dim is None else kv_dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(q_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.inner_dim, eps=eps)
            self.k_norm = RMSNorm(self.inner_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, q_dim, bias=bias), nn.Dropout(dropout)])
        self.set_processor(processor or AudioDiTSelfAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        post_attention_mask: torch.BoolTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        prompt_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            return self.processor(
                self,
                hidden_states,
                attention_mask=attention_mask,
                audio_rotary_emb=audio_rotary_emb,
            )
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            post_attention_mask=post_attention_mask,
            attention_mask=attention_mask,
            audio_rotary_emb=audio_rotary_emb,
            prompt_rotary_emb=prompt_rotary_emb,
        )


class AudioDiTCrossAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "AudioDiTAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        post_attention_mask: torch.BoolTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        prompt_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.qk_norm:
            query = attn.q_norm(query)
            key = attn.k_norm(key)

        head_dim = attn.inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if audio_rotary_emb is not None:
            query = _apply_rotary_emb(query, audio_rotary_emb)
        if prompt_rotary_emb is not None:
            key = _apply_rotary_emb(key, prompt_rotary_emb)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        if post_attention_mask is not None:
            hidden_states = hidden_states * post_attention_mask[:, :, None, None].to(hidden_states.dtype)

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AudioDiTFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


@maybe_allow_in_graph
class AudioDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
        cross_attn: bool = True,
        cross_attn_norm: bool = False,
        adaln_type: str = "global",
        adaln_use_text_cond: bool = True,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        self.adaln_type = adaln_type
        self.adaln_use_text_cond = adaln_use_text_cond
        if adaln_type == "local":
            self.adaln_mlp = AudioDiTAdaLNMLP(dim, dim * 6, bias=True)
        elif adaln_type == "global":
            self.adaln_scale_shift = nn.Parameter(torch.randn(dim * 6) / dim**0.5)

        self.self_attn = AudioDiTAttention(
            dim, None, heads, dim_head, dropout=dropout, bias=bias, qk_norm=qk_norm, eps=eps
        )

        self.use_cross_attn = cross_attn
        if cross_attn:
            self.cross_attn = AudioDiTAttention(
                dim,
                cond_dim,
                heads,
                dim_head,
                dropout=dropout,
                bias=bias,
                qk_norm=qk_norm,
                eps=eps,
                processor=AudioDiTCrossAttnProcessor(),
            )
            self.cross_attn_norm = (
                nn.LayerNorm(dim, elementwise_affine=True, eps=eps) if cross_attn_norm else nn.Identity()
            )
            self.cross_attn_norm_c = (
                nn.LayerNorm(cond_dim, elementwise_affine=True, eps=eps) if cross_attn_norm else nn.Identity()
            )
        self.ffn = AudioDiTFeedForward(dim=dim, mult=ff_mult, dropout=dropout, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        rope: tuple | None = None,
        cond_rope: tuple | None = None,
        adaln_global_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.adaln_type == "local" and adaln_global_out is None:
            if self.adaln_use_text_cond:
                denom = cond_mask.sum(1, keepdim=True).clamp(min=1).to(cond.dtype)
                cond_mean = cond.sum(1) / denom
                norm_cond = timestep_embed + cond_mean
            else:
                norm_cond = timestep_embed
            adaln_out = self.adaln_mlp(norm_cond)
            gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(adaln_out, 6, dim=-1)
        else:
            adaln_out = adaln_global_out + self.adaln_scale_shift.unsqueeze(0)
            gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(adaln_out, 6, dim=-1)

        norm_hidden_states = F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],), eps=1e-6).type_as(
            hidden_states
        )
        norm_hidden_states = norm_hidden_states * (1 + scale_sa[:, None]) + shift_sa[:, None]
        attn_output = self.self_attn(
            norm_hidden_states,
            attention_mask=mask,
            audio_rotary_emb=rope,
        )
        hidden_states = hidden_states + gate_sa.unsqueeze(1) * attn_output

        if self.use_cross_attn:
            cross_output = self.cross_attn(
                hidden_states=self.cross_attn_norm(hidden_states),
                encoder_hidden_states=self.cross_attn_norm_c(cond),
                post_attention_mask=mask,
                attention_mask=cond_mask,
                audio_rotary_emb=rope,
                prompt_rotary_emb=cond_rope,
            )
            hidden_states = hidden_states + cross_output

        norm_hidden_states = F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],), eps=1e-6).type_as(
            hidden_states
        )
        norm_hidden_states = norm_hidden_states * (1 + scale_ffn[:, None]) + shift_ffn[:, None]
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + gate_ffn.unsqueeze(1) * ff_output
        return hidden_states


class LongCatAudioDiTTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False
    _repeated_blocks = ["AudioDiTBlock"]

    @register_to_config
    def __init__(
        self,
        dit_dim: int = 1536,
        dit_depth: int = 24,
        dit_heads: int = 24,
        dit_text_dim: int = 768,
        latent_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attn: bool = True,
        adaln_type: str = "global",
        adaln_use_text_cond: bool = True,
        long_skip: bool = True,
        text_conv: bool = True,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        use_latent_condition: bool = True,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        dim = dit_dim
        dim_head = dim // dit_heads
        self.time_embed = AudioDiTTimestepEmbedding(dim)
        self.input_embed = AudioDiTEmbedder(latent_dim, dim)
        self.text_embed = AudioDiTEmbedder(dit_text_dim, dim)
        self.rotary_embed = AudioDiTRotaryEmbedding(dim_head, 2048, base=100000.0)
        self.blocks = nn.ModuleList(
            [
                AudioDiTBlock(
                    dim=dim,
                    cond_dim=dim,
                    heads=dit_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    bias=bias,
                    qk_norm=qk_norm,
                    eps=eps,
                    cross_attn=cross_attn,
                    cross_attn_norm=cross_attn_norm,
                    adaln_type=adaln_type,
                    adaln_use_text_cond=adaln_use_text_cond,
                    ff_mult=ff_mult,
                )
                for _ in range(dit_depth)
            ]
        )
        self.norm_out = AudioDiTAdaLayerNormZeroFinal(dim, bias=bias, eps=eps)
        self.proj_out = nn.Linear(dim, latent_dim)
        if adaln_type == "global":
            self.adaln_global_mlp = AudioDiTAdaLNMLP(dim, dim * 6, bias=True)
        self.text_conv = text_conv
        if text_conv:
            self.text_conv_layer = nn.Sequential(
                *[AudioDiTConvNeXtV2Block(dim, dim * 2, bias=bias, eps=eps) for _ in range(4)]
            )
        self.use_latent_condition = use_latent_condition
        if use_latent_condition:
            self.latent_embed = AudioDiTEmbedder(latent_dim, dim)
            self.latent_cond_embedder = AudioDiTEmbedder(dim * 2, dim)
        self._initialize_weights(bias=bias)

    def _initialize_weights(self, bias: bool = True):
        if self.config.adaln_type == "local":
            for block in self.blocks:
                nn.init.constant_(block.adaln_mlp.mlp[-1].weight, 0)
                if bias:
                    nn.init.constant_(block.adaln_mlp.mlp[-1].bias, 0)
        elif self.config.adaln_type == "global":
            nn.init.constant_(self.adaln_global_mlp.mlp[-1].weight, 0)
            if bias:
                nn.init.constant_(self.adaln_global_mlp.mlp[-1].bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        if bias:
            nn.init.constant_(self.norm_out.linear.bias, 0)
            nn.init.constant_(self.proj_out.bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.BoolTensor,
        timestep: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
        latent_cond: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> LongCatAudioDiTTransformerOutput | tuple[torch.Tensor]:
        dtype = hidden_states.dtype
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        batch_size = hidden_states.shape[0]
        if timestep.ndim == 0:
            timestep = timestep.repeat(batch_size)
        timestep_embed = self.time_embed(timestep)
        text_mask = encoder_attention_mask.bool()
        encoder_hidden_states = self.text_embed(encoder_hidden_states, text_mask)
        if self.text_conv:
            encoder_hidden_states = self.text_conv_layer(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.masked_fill(text_mask.logical_not().unsqueeze(-1), 0.0)
        hidden_states = self.input_embed(hidden_states, attention_mask)
        if self.use_latent_condition and latent_cond is not None:
            latent_cond = self.latent_embed(latent_cond.to(hidden_states.dtype), attention_mask)
            hidden_states = self.latent_cond_embedder(torch.cat([hidden_states, latent_cond], dim=-1))
        residual = hidden_states.clone() if self.config.long_skip else None
        rope = self.rotary_embed(hidden_states, hidden_states.shape[1])
        cond_rope = self.rotary_embed(encoder_hidden_states, encoder_hidden_states.shape[1])
        if self.config.adaln_type == "global":
            if self.config.adaln_use_text_cond:
                text_len = text_mask.sum(1).clamp(min=1).to(encoder_hidden_states.dtype)
                text_mean = encoder_hidden_states.sum(1) / text_len.unsqueeze(1)
                norm_cond = timestep_embed + text_mean
            else:
                norm_cond = timestep_embed
            adaln_global_out = self.adaln_global_mlp(norm_cond)
            for block in self.blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    timestep_embed=timestep_embed,
                    cond=encoder_hidden_states,
                    mask=attention_mask,
                    cond_mask=text_mask,
                    rope=rope,
                    cond_rope=cond_rope,
                    adaln_global_out=adaln_global_out,
                )
        else:
            norm_cond = timestep_embed
            for block in self.blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    timestep_embed=timestep_embed,
                    cond=encoder_hidden_states,
                    mask=attention_mask,
                    cond_mask=text_mask,
                    rope=rope,
                    cond_rope=cond_rope,
                )
        if self.config.long_skip:
            hidden_states = hidden_states + residual
        hidden_states = self.norm_out(hidden_states, norm_cond)
        hidden_states = self.proj_out(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        if not return_dict:
            return (hidden_states,)
        return LongCatAudioDiTTransformerOutput(sample=hidden_states)
