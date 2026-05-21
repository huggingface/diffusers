# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

from typing import Tuple

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..attention import AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin


class CosmosAttnProcessor3_0:
    """Dual-pathway attention processor for Cosmos3.

    Projects, normalizes, applies rotary position embeddings, then runs separate
    causal (understanding) and full (generation) attention pathways. The generation
    pathway cross-attends to both und and gen keys/values.
    """

    def __call__(
        self,
        attn: "PackedAttentionMoT",
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-pathway projections
        q_und = attn.q_proj(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.k_proj(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.v_proj(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.q_proj_moe_gen(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.k_proj_moe_gen(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.v_proj_moe_gen(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.q_norm(q_und)
        k_und = attn.k_norm(k_und)
        q_gen = attn.q_norm_moe_gen(q_gen)
        k_gen = attn.k_norm_moe_gen(k_gen)

        # Apply rotary position embeddings per pathway
        cos_und, sin_und, cos_gen, sin_gen = position_embeddings
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        # Causal pathway (understanding): und tokens self-attend with causal masking.
        causal_out = (
            dispatch_attention_fn(
                q_und.unsqueeze(0),
                k_und.unsqueeze(0),
                v_und.unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
            )
            .squeeze(0)
            .flatten(-2, -1)
        )

        # Full pathway (generation): gen tokens cross-attend to all (und + gen) keys/values.
        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = (
            dispatch_attention_fn(
                q_gen.unsqueeze(0),
                all_k.unsqueeze(0),
                all_v.unsqueeze(0),
                is_causal=False,
                enable_gqa=True,
            )
            .squeeze(0)
            .flatten(-2, -1)
        )

        # Per-pathway output projection
        und_out = attn.o_proj(causal_out)
        gen_out = attn.o_proj_moe_gen(full_out)
        return und_out, gen_out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float, rope_scaling: dict | None = None):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = (
            rope_scaling.get("mrope_section", [24, 20, 20]) if rope_scaling is not None else [24, 20, 20]
        )

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Reorganize chunked [TTT...HHH...WWW] frequency layout into interleaved
        [THTHWHTHW...TT], preserving frequency continuity across the 3 grids."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )  # [3,B,head_dim//2,1]
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3,B,1,N]
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)  # [3,B,N,head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)  # [B,N,head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,N,head_dim]
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)  # each: [B,N,head_dim]


class Cosmos3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Cosmos3VLTextMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PackedAttentionMoT(nn.Module, AttentionModuleMixin):
    """Dual-pathway packed attention for Qwen3VL MoT — separate projections for
    understanding (causal) and generation (full) token streams."""

    _default_processor_cls = CosmosAttnProcessor3_0
    _available_processors = [CosmosAttnProcessor3_0]

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        attention_dropout: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.is_causal = True

        # Understanding pathway. q_norm / k_norm are applied per-head (only on
        # head_dim), so no reshape is needed after them.
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm = Cosmos3VLTextRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Cosmos3VLTextRMSNorm(head_dim, eps=rms_norm_eps)

        # Generation pathway
        self.q_proj_moe_gen = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj_moe_gen = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj_moe_gen = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj_moe_gen = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm_moe_gen = Cosmos3VLTextRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm_moe_gen = Cosmos3VLTextRMSNorm(head_dim, eps=rms_norm_eps)

        self.set_processor(CosmosAttnProcessor3_0())

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, und_seq, gen_seq, position_embeddings)


class Cosmos3VLTextMoTDecoderLayer(nn.Module):
    """
    Qwen3VL text MoT (Mixture of Tokens) decoder layer.
    Features dual-pathway attention for understanding vs generation.

    This is used for both Dense and MoE models.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool,
        attention_dropout: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = PackedAttentionMoT(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
        )

        self.mlp = Cosmos3VLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.mlp_moe_gen = Cosmos3VLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

        self.input_layernorm = Cosmos3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm_moe_gen = Cosmos3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Cosmos3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Cosmos3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, position_embeddings)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))

        return residual_und + mlp_out_und, residual_gen + mlp_out_gen


class Cosmos3VLTextModel(nn.Module):
    """Inner transformer stack — mirrors Cosmos3OmniTextModel.

    Holds ``embed_tokens``, ``layers``, dual final norms, and ``rotary_emb``
    under the same attribute names so that weight keys match the checkpoint.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        num_hidden_layers: int,
        attention_bias: bool,
        attention_dropout: float,
        rms_norm_eps: float,
        rope_theta: float,
        rope_scaling: dict | None,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Cosmos3VLTextMoTDecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = Cosmos3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_moe_gen = Cosmos3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Cosmos3VLTextRotaryEmbedding(
            head_dim=head_dim, rope_theta=rope_theta, rope_scaling=rope_scaling
        )

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create position embeddings (Qwen3 style). The dummy tensor only carries
        # dtype/device — rotary_emb needs its dtype to cast cos/sin back.
        _meta_tensor = torch.tensor([], dtype=und_seq.dtype, device=und_seq.device)
        cos, sin = self.rotary_emb(
            _meta_tensor,
            position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
        )  # if ndim == 2, then the mrope position_ids is (3, seq_len), we need to put batch dimension in the middle to make it compatible with the rotary_emb
        # cos, sin: [1,N,head_dim] (1D pos_ids) or [3,1,N,head_dim] (mrope pos_ids)
        cos = cos.squeeze(0)  # [N,head_dim] or [3,N,head_dim]
        sin = sin.squeeze(0)  # [N,head_dim] or [3,N,head_dim]

        # The joint sequence layout is [und... | gen...] contiguously (per pack_input_sequence),
        # so the und/gen position embeddings are just the leading / trailing slices.
        und_len = und_seq.shape[0]
        position_embeddings = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])

        for decoder_layer in self.layers:
            und_seq, gen_seq = decoder_layer(und_seq, gen_seq, position_embeddings)

        return self.norm(und_seq), self.norm_moe_gen(gen_seq)


class Cosmos3OmniTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        dtype: str = "bfloat16",  # accepted by ConfigMixin loader (configuration_utils.py:288); not read directly
        head_dim: int = 128,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        base_fps: int = 24,
        enable_fps_modulation: bool = True,
        joint_attn_implementation: str = "two_way",
        latent_channel: int = 48,
        max_action_dim: int = 32,
        position_embedding_type: str = "unified_3d_mrope",
        unified_3d_mrope_reset_spatial_ids: bool = True,
        unified_3d_mrope_temporal_modality_margin: int = 15000,
        video_temporal_causal: bool = False,
        latent_patch_size: int = 2,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 36,
        num_key_value_heads: int = 8,
        patch_latent_dim: int = 192,
        rms_norm_eps: float = 1e-6,
        rope_scaling: dict | None = None,
        rope_theta: float = 5000000.0,
        sound_dim: int | None = None,
        sound_gen: bool = False,
        sound_latent_fps: float = 25.0,
        timestep_scale: float = 0.001,
        use_moe: bool = True,
        vocab_size: int = 151936,
    ):
        super().__init__()

        if rope_scaling is None:
            rope_scaling = {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"}
            self.register_to_config(rope_scaling=rope_scaling)

        self.model = Cosmos3VLTextModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vae2llm = nn.Linear(patch_latent_dim, hidden_size, bias=True)
        self.llm2vae = nn.Linear(hidden_size, patch_latent_dim, bias=True)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        if sound_gen:
            if sound_dim is None:
                raise ValueError("`sound_dim` must be provided when `sound_gen=True`.")
            self.sound2llm = nn.Linear(sound_dim, hidden_size, bias=True)
            self.llm2sound = nn.Linear(hidden_size, sound_dim, bias=True)
            self.sound_modality_embed = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(und_seq=und_seq, gen_seq=gen_seq, position_ids=position_ids)
