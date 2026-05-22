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

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin


class CosmosAttnProcessor3_0:
    """Dual-pathway attention processor for Cosmos3.

    Projects, normalizes, applies rotary position embeddings, then runs separate
    causal (understanding) and full (generation) attention pathways. The generation
    pathway cross-attends to both und and gen keys/values.
    """

    _attention_backend = None
    _parallel_config = None

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
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
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
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
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


class Cosmos3OmniTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["Cosmos3VLTextMoTDecoderLayer"]
    _repeated_blocks = ["Cosmos3VLTextMoTDecoderLayer"]
    _skip_layerwise_casting_patterns = ["embed_tokens", "time_embedder", "norm"]
    _keep_in_fp32_modules = ["time_embedder"]
    # `dtype` is injected into init_dict by ModelMixin.from_pretrained (configuration_utils.py:289),
    # so __init__ must accept it. Excluding it here keeps save_pretrained from writing it into
    # config.json — the value is a load-time runtime hint, not part of the model architecture.
    ignore_for_config = ["dtype"]

    @register_to_config
    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        dtype: str = "bfloat16",  # required by the loader (see `ignore_for_config` above); not read here
        head_dim: int = 128,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        base_fps: int = 24,
        enable_fps_modulation: bool = True,
        joint_attn_implementation: str = "two_way",
        latent_channel: int = 48,
        position_embedding_type: str = "unified_3d_mrope",
        unified_3d_mrope_reset_spatial_ids: bool = True,
        unified_3d_mrope_temporal_modality_margin: int = 15000,
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

        # Text-model layers live directly on the transformer (flat layout). The published
        # checkpoint must be re-keyed with the leading `model.` prefix stripped — see
        # scripts/build_flat_layout_repo.py for the rewrite.
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

        # Modality projection heads + timestep embedding.
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

        self.gradient_checkpointing = False

    # -------------------------------------------------------------------------
    # Pure-tensor packing/unpacking helpers (no layer state).
    # -------------------------------------------------------------------------

    def _apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: List[torch.Tensor],
        token_shapes: List[Tuple[int, ...]],
    ) -> torch.Tensor:
        start_noisy_index = 0
        flattened_noisy_frame_indexes: List[torch.Tensor] = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            noisy_indexes_i = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
            noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())
            start_noisy_index += math.prod(token_shape_i)
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _patchify_and_pack_latents(
        self,
        tokens_vision: List[torch.Tensor],
        token_shapes_vision: List[Tuple[int, int, int]],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        packed_latent: List[torch.Tensor] = []
        original_latent_shapes: List[Tuple[int, int, int]] = []
        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)  # [C, T, H, W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def _unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: List[torch.Tensor],
        original_latent_shapes: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[torch.Tensor]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        unpatchified_latents: List[torch.Tensor] = []
        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            if original_latent_shapes is not None:
                _, h_orig, w_orig = original_latent_shapes[i]
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                h_orig, w_orig = h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c
            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                output_tensor[:, noisy_frame_indexes] = latent
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def _pack_sound_latents(
        self,
        tokens_sound: List[torch.Tensor],
        token_shapes_sound: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """List of ``[C, T]`` tensors → packed ``[total_T, C]`` tensor."""
        packed: List[torch.Tensor] = []
        for sound, shape in zip(tokens_sound, token_shapes_sound):
            T = shape[0]
            packed.append(sound[:, :T].permute(1, 0))  # [C, T] → [T, C]
        return torch.cat(packed, dim=0)

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: List[Tuple[int, int, int]],
        noisy_frame_indexes_sound: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Packed ``[total_noisy_T, C]`` predictions → list of ``[C, T]`` tensors (zeros at conditioned positions)."""
        sound_dim = self.config.sound_dim
        unpacked: List[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            output = torch.zeros((sound_dim, T), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[:, noisy_idxs] = packed_preds[start_idx : start_idx + t_n].T
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    # -------------------------------------------------------------------------
    # forward: full per-step pass — encode text/vision/sound → run layers →
    # decode vision/sound. Pipeline calls this once per CFG pass.
    # -------------------------------------------------------------------------

    def forward(self, packed_seq: Any) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Run a full denoising-step forward pass.

        Args:
            packed_seq: ``Cosmos3PackedSequence`` from ``Cosmos3OmniDiffusersPipeline.pack_input_sequence``
                — carries text/vision/sound
                token data + position_ids + the und/gen split point.

        Returns:
            ``(preds_vision, preds_sound)`` — list of per-modality latents (``preds_sound`` is
            ``None`` when the model has no sound branch or the packed sequence has no sound tokens).
        """
        vision = packed_seq.vision
        sound = packed_seq.sound
        has_sound = sound is not None and sound.tokens is not None

        # Embed text tokens into the joint hidden_states buffer at their sequence positions.
        packed_text_embedding = self.embed_tokens(packed_seq.text_ids)
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(packed_seq.sequence_length, self.config.hidden_size))
        hidden_states[packed_seq.text_indexes] = packed_text_embedding

        # Patchify + project vision latents, then add timestep embeddings to noisy frames.
        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(
            vision.tokens, vision.token_shapes
        )
        packed_tokens_vision = self.vae2llm(packed_tokens_vision)
        timesteps_vision = vision.timesteps * self.config.timestep_scale
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision))
        packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds_vision,
            noisy_frame_indexes=vision.noisy_frame_indexes,
            token_shapes=vision.token_shapes,
        )
        hidden_states[vision.sequence_indexes] = packed_tokens_vision

        # Pack + project sound latents (when present); all sound frames are noisy.
        if has_sound:
            packed_tokens_sound = self._pack_sound_latents(sound.tokens, sound.token_shapes).to(target_dtype)
            packed_tokens_sound = self.sound2llm(packed_tokens_sound) + self.sound_modality_embed
            timesteps_sound = sound.timesteps * self.config.timestep_scale
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_sound = self.time_embedder(self.time_proj(timesteps_sound))
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
            packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound.noisy_frame_indexes,
                token_shapes=sound.token_shapes,
            )
            hidden_states[sound.sequence_indexes] = packed_tokens_sound

        # Compute rotary embeddings once for the joint sequence, then slice into und/gen halves.
        _meta_tensor = torch.tensor([], dtype=hidden_states.dtype, device=hidden_states.device)
        position_ids = packed_seq.position_ids
        cos, sin = self.rotary_emb(
            _meta_tensor,
            position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
        )
        # cos, sin: [1, N, head_dim] (1-D pos_ids) or [3, 1, N, head_dim] (mrope pos_ids)
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        und_len = packed_seq.und_len
        und_seq = hidden_states[:und_len]
        gen_seq = hidden_states[und_len:]
        position_embeddings = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
        for decoder_layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                und_seq, gen_seq = self._gradient_checkpointing_func(
                    decoder_layer.__call__, und_seq, gen_seq, position_embeddings
                )
            else:
                und_seq, gen_seq = decoder_layer(und_seq, gen_seq, position_embeddings)
        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)
        last_hidden_state = torch.cat([und_out, gen_out], dim=0)

        # Decode vision predictions from the joint hidden state.
        preds_vision_packed = self.llm2vae(last_hidden_state[vision.mse_loss_indexes])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision.token_shapes,
            noisy_frame_indexes_vision=vision.noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

        preds_sound: Optional[List[torch.Tensor]] = None
        if has_sound:
            preds_sound_packed = self.llm2sound(last_hidden_state[sound.mse_loss_indexes])
            preds_sound = self._unpack_sound_latents(preds_sound_packed, sound.token_shapes, sound.noisy_frame_indexes)

        return preds_vision, preds_sound
