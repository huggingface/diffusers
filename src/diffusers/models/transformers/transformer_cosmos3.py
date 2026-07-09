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

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


class Cosmos3AttnProcessor:
    """Dual-pathway attention processor for Cosmos3.

    Projects, normalizes, applies rotary position embeddings, then runs separate causal (understanding) and full
    (generation) attention pathways. The generation pathway cross-attends to both und and gen keys/values.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "Cosmos3PackedMoTAttention",
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-pathway projections
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        # Apply rotary position embeddings per pathway
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        # Causal pathway (understanding): und tokens self-attend with causal masking.
        causal_out = dispatch_attention_fn(
            q_und.unsqueeze(0),
            k_und.unsqueeze(0),
            v_und.unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        causal_out = causal_out.squeeze(0).flatten(-2, -1)

        # Full pathway (generation): gen tokens cross-attend to all (und + gen) keys/values.
        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = dispatch_attention_fn(
            q_gen.unsqueeze(0),
            all_k.unsqueeze(0),
            all_v.unsqueeze(0),
            is_causal=False,
            enable_gqa=True,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        full_out = full_out.squeeze(0).flatten(-2, -1)

        # Per-pathway output projection
        und_out = attn.to_out(causal_out)
        gen_out = attn.to_add_out(full_out)
        return und_out, gen_out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float, rope_axes_dim: tuple[int, int, int]):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rope_axes_dim = rope_axes_dim

    def apply_interleaved_mrope(self, freqs, rope_axes_dim):
        """Reorganize chunked [TTT...HHH...WWW] frequency layout into interleaved
        [THTHWHTHW...TT], preserving frequency continuity across the 3 grids."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = rope_axes_dim[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids, device, dtype):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(device)
        )  # [3,B,head_dim//2,1]
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3,B,1,N]
        # Disable autocast so the position-id matmul runs in float32: under an ambient autocast it would run in
        # bfloat16, which cannot represent consecutive integers past 256, collapsing positions onto the same
        # frequency and degrading the rotary embedding.
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = inv_freq_expanded @ position_ids_expanded
        freqs = freqs.transpose(2, 3)  # [3,B,N,head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs, self.rope_axes_dim)  # [B,N,head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,N,head_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)  # each: [B,N,head_dim]


class Cosmos3VLTextMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DomainAwareLinear(nn.Module):
    """Linear projection with one weight/bias pair per embodiment domain."""

    def __init__(self, input_size: int, output_size: int, num_domains: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(self.num_domains, self.output_size * self.input_size)
        self.bias = nn.Embedding(self.num_domains, self.output_size)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        if x.shape[0] != domain_id.shape[0]:
            raise ValueError(
                "Cosmos3 action domain_id batch size must match action tokens: "
                f"tokens={x.shape[0]}, domain_id={domain_id.shape[0]}."
            )
        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
            raise ValueError(f"Cosmos3 action domain_id must be in [0, {self.num_domains}), got {domain_id.tolist()}.")
        weight = self.fc(domain_id).view(domain_id.shape[0], self.input_size, self.output_size)
        bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        if x.ndim == 3:
            return torch.bmm(x, weight) + bias.unsqueeze(1)
        raise ValueError(f"Cosmos3 DomainAwareLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")


class Cosmos3PackedMoTAttention(nn.Module, AttentionModuleMixin):
    """Dual-pathway packed attention for Qwen3VL not — separate projections for
    understanding (causal) and generation (full) token streams."""

    _default_processor_cls = Cosmos3AttnProcessor
    _available_processors = [Cosmos3AttnProcessor]

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
        processor=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Understanding pathway. norm_q / norm_k are applied per-head (only on
        # head_dim), so no reshape is needed after them.
        self.to_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_q = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_k = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)

        # Generation pathway
        self.add_q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.add_k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.add_v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_add_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_added_q = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_added_k = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, und_seq, gen_seq, rotary_emb)


class Cosmos3VLTextMoTDecoderLayer(nn.Module):
    """
    Qwen3VL text not (Mixture of Tokens) decoder layer. Features dual-pathway attention for understanding vs
    generation.

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
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = Cosmos3PackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )

        self.mlp = Cosmos3VLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.mlp_moe_gen = Cosmos3VLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.input_layernorm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.post_attention_layernorm_moe_gen = RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False
        )

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
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
        latent_channel: int = 48,
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
        action_dim: int | None = None,
        action_gen: bool = False,
        num_embodiment_domains: int = 32,
        sound_dim: int | None = None,
        sound_gen: bool = False,
        sound_latent_fps: float = 25.0,
        timestep_scale: float = 0.001,
        vocab_size: int = 151936,
    ):
        super().__init__()

        rope_axes_dim = rope_scaling.get("mrope_section", [24, 20, 20]) if rope_scaling is not None else [24, 20, 20]
        self.register_to_config(rope_axes_dim=rope_axes_dim)

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
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.rotary_emb = Cosmos3VLTextRotaryEmbedding(
            head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes_dim
        )

        # Modality projection heads + timestep embedding.
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.proj_in = nn.Linear(patch_latent_dim, hidden_size, bias=True)
        self.proj_out = nn.Linear(hidden_size, patch_latent_dim, bias=True)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.action_gen = action_gen
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        if action_gen:
            if self.action_dim is None:
                raise ValueError("`action_dim` must be provided when `action_gen=True`.")
            self.action_proj_in = DomainAwareLinear(self.action_dim, hidden_size, self.num_embodiment_domains)
            self.action_proj_out = DomainAwareLinear(hidden_size, self.action_dim, self.num_embodiment_domains)
            self.action_modality_embed = nn.Parameter(torch.zeros(hidden_size))
        if sound_gen:
            if sound_dim is None:
                raise ValueError("`sound_dim` must be provided when `sound_gen=True`.")
            self.audio_proj_in = nn.Linear(sound_dim, hidden_size, bias=True)
            self.audio_proj_out = nn.Linear(hidden_size, sound_dim, bias=True)
            self.audio_modality_embed = nn.Parameter(torch.zeros(hidden_size))

        self.gradient_checkpointing = False

    # -------------------------------------------------------------------------
    # Pure-tensor packing/unpacking helpers (no layer state).
    # -------------------------------------------------------------------------

    def _apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: list[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        start_noisy_index = 0
        flattened_noisy_frame_indexes: list[torch.Tensor] = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            # Broadcast [N, 1] + [spatial_numel_i] → [N, spatial_numel_i]
            frame_offsets = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1) + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(frame_offsets.flatten())
            start_noisy_index += token_shape_i[0] * spatial_numel_i
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _patchify_and_pack_latents(
        self,
        tokens_vision: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        packed_latent: list[torch.Tensor] = []
        original_latent_shapes: list[tuple[int, int, int]] = []
        for latent in tokens_vision:
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
        token_shapes_vision: list[tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: list[tuple[int, int, int]],
    ) -> list[torch.Tensor]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        unpatchified_latents: list[torch.Tensor] = []
        start_idx = 0
        for token_shape, noisy_frame_indexes, original_shape in zip(
            token_shapes_vision, noisy_frame_indexes_vision, original_latent_shapes
        ):
            t_c = token_shape[0]
            _, h_orig, w_orig = original_shape
            h_padded = ((h_orig + p - 1) // p) * p
            w_padded = ((w_orig + p - 1) // p) * p
            h_patches = h_padded // p
            w_patches = w_padded // p
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
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """List of ``[C, T]`` tensors → packed ``[total_T, C]`` tensor."""
        return torch.cat(
            [sound[:, : shape[0]].permute(1, 0) for sound, shape in zip(tokens_sound, token_shapes_sound)],
            dim=0,
        )

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, C]`` predictions → list of ``[C, T]`` tensors (zeros at conditioned positions)."""
        sound_dim = self.config.sound_dim
        unpacked: list[torch.Tensor] = []
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

    def _pack_action_latents(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, int, int]],
        domain_ids_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """List of ``[T, D]`` tensors → packed ``[total_T, D]`` plus per-token domain ids."""
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for action, shape, domain_id in zip(tokens_action, token_shapes_action, domain_ids_action):
            token_count = shape[0]
            packed.append(action[:token_count])
            domain_ids.append(domain_id.reshape(1).expand(token_count))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def _unpack_action_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, int, int]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, D]`` predictions → list of ``[T, D]`` tensors."""
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_action, noisy_frame_indexes_action):
            T = shape[0]
            output = torch.zeros((T, self.action_dim), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[noisy_idxs] = packed_preds[start_idx : start_idx + t_n]
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    # -------------------------------------------------------------------------
    # forward: full per-step pass — encode text/vision/sound/action → run layers →
    # decode vision/sound/action. Pipeline calls this once per CFG pass.
    # -------------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        und_len: int,
        sequence_length: int,
        vision_tokens: list[torch.Tensor],
        vision_token_shapes: list[tuple[int, int, int]],
        vision_sequence_indexes: torch.Tensor,
        vision_mse_loss_indexes: torch.Tensor,
        vision_timesteps: torch.Tensor,
        vision_noisy_frame_indexes: list[torch.Tensor],
        sound_tokens: list[torch.Tensor] | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_sequence_indexes: torch.Tensor | None = None,
        sound_mse_loss_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_tokens: list[torch.Tensor] | None = None,
        action_token_shapes: list[tuple[int, int, int]] | None = None,
        action_sequence_indexes: torch.Tensor | None = None,
        action_mse_loss_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_domain_ids: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Run a full denoising-step forward pass.

        Args:
            input_ids: Text token IDs placed at ``text_indexes`` in the joint sequence.
            text_indexes: Indices of text tokens in the joint sequence.
            position_ids: ``[3, sequence_length]`` mRoPE position IDs for the full joint sequence.
            und_len: Length of the causal text (understanding) prefix; generation tokens follow.
            sequence_length: Total length of the joint packed sequence.
            vision_tokens: Per-item vision latent tensors before patchify.
            vision_token_shapes: Patch grid shapes ``(T, H, W)`` per vision item.
            vision_sequence_indexes: Indices of vision tokens in the joint sequence.
            vision_mse_loss_indexes: Indices used to read vision predictions after the backbone.
            vision_timesteps: Per-patch diffusion timesteps for vision tokens.
            vision_noisy_frame_indexes: Noisy frame indices per vision item.
            sound_tokens: Optional sound latent tensors before packing.
            sound_token_shapes: Optional patch grid shapes for sound items.
            sound_sequence_indexes: Optional indices of sound tokens in the joint sequence.
            sound_mse_loss_indexes: Optional indices used to read sound predictions.
            sound_timesteps: Optional per-token diffusion timesteps for sound.
            sound_noisy_frame_indexes: Optional noisy frame indices per sound item.
            action_tokens: Optional action latent tensors before packing.
            action_token_shapes: Optional patch grid shapes ``(T, H, W)`` per action item.
            action_sequence_indexes: Optional indices of action tokens in the joint sequence.
            action_mse_loss_indexes: Optional indices used to read action predictions after the backbone.
            action_timesteps: Optional per-token diffusion timesteps for action tokens.
            action_noisy_frame_indexes: Optional noisy frame indices per action item.
            action_domain_ids: Optional per-item domain IDs selecting the action head weights.

        Returns:
            ``(preds_vision, preds_sound, preds_action)`` — lists of per-modality predictions. Optional modalities
            return ``None`` when their inputs are omitted.
        """
        has_sound = sound_tokens is not None and sound_sequence_indexes is not None
        has_action = action_tokens is not None and action_sequence_indexes is not None

        # Embed text tokens into the joint hidden_states buffer at their sequence positions.
        packed_text_embedding = self.embed_tokens(input_ids)
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.config.hidden_size))
        hidden_states[text_indexes] = packed_text_embedding

        # Patchify + project vision latents, then add timestep embeddings to noisy frames.
        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(vision_tokens)
        packed_tokens_vision = self.proj_in(packed_tokens_vision)
        timesteps_vision = vision_timesteps * self.config.timestep_scale
        packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision))
        packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds_vision,
            noisy_frame_indexes=vision_noisy_frame_indexes,
            token_shapes=vision_token_shapes,
        )
        hidden_states[vision_sequence_indexes] = packed_tokens_vision

        # Pack + project sound latents (when present); all sound frames are noisy.
        if has_sound:
            packed_tokens_sound = self._pack_sound_latents(sound_tokens, sound_token_shapes).to(target_dtype)
            packed_tokens_sound = self.audio_proj_in(packed_tokens_sound) + self.audio_modality_embed
            timesteps_sound = sound_timesteps * self.config.timestep_scale
            packed_timestep_embeds_sound = self.time_embedder(self.time_proj(timesteps_sound))
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
            packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound_noisy_frame_indexes,
                token_shapes=sound_token_shapes,
            )
            hidden_states[sound_sequence_indexes] = packed_tokens_sound

        # Pack + project action latents (when present). Domain ids select the action head weights.
        if has_action:
            packed_tokens_action, per_token_domain_ids = self._pack_action_latents(
                action_tokens, action_token_shapes, action_domain_ids
            )
            packed_tokens_action = packed_tokens_action.to(target_dtype)
            per_token_domain_ids = per_token_domain_ids.to(device=packed_tokens_action.device)
            packed_tokens_action = self.action_proj_in(packed_tokens_action, per_token_domain_ids)
            packed_tokens_action = packed_tokens_action + self.action_modality_embed
            if action_mse_loss_indexes.numel() > 0:
                timesteps_action = action_timesteps * self.config.timestep_scale
                packed_timestep_embeds_action = self.time_embedder(self.time_proj(timesteps_action))
                packed_timestep_embeds_action = packed_timestep_embeds_action.to(target_dtype)
                packed_tokens_action = self._apply_timestep_embeds_to_noisy_tokens(
                    packed_tokens=packed_tokens_action,
                    packed_timestep_embeds=packed_timestep_embeds_action,
                    noisy_frame_indexes=action_noisy_frame_indexes,
                    token_shapes=action_token_shapes,
                )
            hidden_states[action_sequence_indexes] = packed_tokens_action

        # Compute rotary embeddings once for the joint sequence, then slice into und/gen halves.
        _meta_tensor = torch.tensor([], dtype=hidden_states.dtype, device=hidden_states.device)
        cos, sin = self.rotary_emb(
            position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # cos, sin: [1, N, head_dim] (1-D pos_ids) or [3, 1, N, head_dim] (mrope pos_ids)
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        und_seq = hidden_states[:und_len]
        gen_seq = hidden_states[und_len:]
        rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
        for decoder_layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                und_seq, gen_seq = self._gradient_checkpointing_func(
                    decoder_layer.__call__, und_seq, gen_seq, rotary_emb
                )
            else:
                und_seq, gen_seq = decoder_layer(und_seq, gen_seq, rotary_emb)
        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)
        last_hidden_state = torch.cat([und_out, gen_out], dim=0)

        # Decode vision predictions from the joint hidden state.
        preds_vision_packed = self.proj_out(last_hidden_state[vision_mse_loss_indexes])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

        preds_sound: list[torch.Tensor] | None = None
        if has_sound:
            preds_sound_packed = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
            preds_sound = self._unpack_sound_latents(preds_sound_packed, sound_token_shapes, sound_noisy_frame_indexes)

        preds_action: list[torch.Tensor] | None = None
        if has_action:
            per_noisy_domain_ids = [
                domain_id.reshape(1).expand(len(noisy_idxs))
                for domain_id, noisy_idxs in zip(action_domain_ids, action_noisy_frame_indexes)
            ]
            per_noisy_domain_ids = torch.cat(per_noisy_domain_ids, dim=0).to(device=last_hidden_state.device)
            preds_action_packed = self.action_proj_out(
                last_hidden_state[action_mse_loss_indexes], per_noisy_domain_ids
            )
            preds_action = self._unpack_action_latents(
                preds_action_packed, action_token_shapes, action_noisy_frame_indexes
            )

        return preds_vision, preds_sound, preds_action
