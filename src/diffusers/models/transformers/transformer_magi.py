# Copyright 2025 SandAI and The HuggingFace Team. All rights reserved.
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

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import BaseOutput, apply_lora_scale, is_flash_attn_available
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin


if is_flash_attn_available():
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb


@dataclass
class MagiTransformer3DModelOutput(BaseOutput):
    """Predicted flow velocities and optional per-layer self-attention keys and values."""

    sample: torch.Tensor
    kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None


class MagiFP32Linear(nn.Linear):
    def forward(self, hidden_states):
        bias = self.bias.float() if self.bias is not None else None
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return F.linear(hidden_states.float(), self.weight.float(), bias)


class MagiPatchEmbedding(nn.Conv3d):
    def forward(self, hidden_states):
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return F.conv3d(hidden_states.float(), self.weight.float(), stride=self.stride)


class MagiLayerNorm(nn.Module):
    def __init__(self, dim, eps, zero_centered_gamma=True, upcast=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim) if zero_centered_gamma else torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.upcast = upcast

    def forward(self, hidden_states):
        if self.upcast:
            hidden_states = hidden_states.float()
        weight = self.weight.to(hidden_states.dtype)
        weight = weight + 1 if self.zero_centered_gamma else weight
        return F.layer_norm(hidden_states, self.weight.shape, weight, self.bias.to(hidden_states.dtype), self.eps)


class MagiTimestepEmbedding(nn.Module):
    def __init__(self, dim, frequency_embedding_size):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(MagiFP32Linear(frequency_embedding_size, dim), nn.SiLU(), MagiFP32Linear(dim, dim))

    def forward(self, timestep, hidden_dtype):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device="cpu").float() / half).to(timestep.device)
        angles = timestep.flatten()[:, None].float() * frequencies[None] * 1000
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(hidden_dtype))


class MagiCaptionEmbedding(nn.Module):
    def __init__(self, caption_channels, caption_max_length, dim, condition_dim):
        super().__init__()
        self.null_caption_embedding = nn.Parameter(torch.zeros(caption_max_length, caption_channels))
        self.y_proj_xattn = nn.Sequential(MagiFP32Linear(caption_channels, dim), nn.SiLU())
        self.y_proj_adaln = nn.Sequential(MagiFP32Linear(caption_channels, condition_dim))

    def forward(self, encoder_hidden_states, caption_dropout_mask):
        encoder_hidden_states = self.y_proj_xattn(encoder_hidden_states.contiguous())
        caption = torch.where(
            caption_dropout_mask[:, None], self.null_caption_embedding[-1], self.null_caption_embedding[-2]
        )
        condition = self.y_proj_adaln(caption)
        return encoder_hidden_states, condition


class MagiConditionEmbedding(nn.Module):
    def __init__(self, caption_channels, caption_max_length, dim, condition_dim, frequency_embedding_size):
        super().__init__()
        self.t_embedder = MagiTimestepEmbedding(condition_dim, frequency_embedding_size)
        self.y_embedder = MagiCaptionEmbedding(caption_channels, caption_max_length, dim, condition_dim)

    def forward(self, timestep, encoder_hidden_states, caption_dropout_mask, timestep_delta, hidden_dtype):
        condition = self.t_embedder(timestep, hidden_dtype)
        if timestep_delta is not None:
            condition = condition + self.t_embedder(timestep_delta, hidden_dtype)
        condition = condition.reshape(*timestep.shape, -1)
        encoder_hidden_states, caption_condition = self.y_embedder(encoder_hidden_states, caption_dropout_mask)
        return condition + caption_condition[:, None], encoder_hidden_states


class MagiRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        num_bands = head_dim // 8
        self.bands = nn.Parameter(1.0 / (10000 ** (torch.arange(num_bands).float() / num_bands)))

    def forward(self, num_frames, height, width, frame_offset=0):
        total_frames = num_frames + frame_offset
        rescale_factor = math.sqrt(height * width / 256)
        shapes = (total_frames, height, width)
        reference_shapes = (total_frames, height / rescale_factor, width / rescale_factor)
        coordinates = []
        for axis, (size, reference_size) in enumerate(zip(shapes, reference_shapes)):
            coordinate = torch.arange(size, device=self.bands.device).float()
            if axis > 0:
                coordinate = coordinate - (size - 1) / 2
            if size > 1:
                coordinate = coordinate / (size - 1) * (reference_size - 1)
            coordinates.append(coordinate)
        grid = torch.stack(torch.meshgrid(*coordinates, indexing="ij"), dim=-1)
        angles = grid.unsqueeze(-1) * self.bands.float()
        angles = angles[frame_offset:].reshape(num_frames * height * width, -1)
        return angles.cos(), angles.sin()


class MagiQueryKeyValueProjection(nn.Module):
    def __init__(self, dim, kv_dim, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=eps)
        self.q = nn.Linear(dim, dim, bias=False)
        self.qx = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, kv_dim, bias=False)
        self.v = nn.Linear(dim, kv_dim, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        return self.q(hidden_states), self.qx(hidden_states), self.k(hidden_states), self.v(hidden_states)


class MagiTextKeyValueProjection(nn.Module):
    def __init__(self, dim, kv_dim):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, 2 * kv_dim // 8, bias=False) for _ in range(8)])

    def forward(self, hidden_states):
        return torch.cat([projection(hidden_states) for projection in self.projections], dim=-1)


class MagiAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self, attn, hidden_states, encoder_hidden_states, rotary_emb, encoder_attention_mask, kv_ranges, kv_cache
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        num_chunks = encoder_hidden_states.shape[1]
        chunk_length = sequence_length // num_chunks
        query, cross_query, key, value = attn.linear_qkv(hidden_states)
        query = query.unflatten(-1, (attn.num_heads, attn.head_dim))
        key = key.unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = value.unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        query = attn.q_layernorm(query)
        key = attn.k_layernorm(key)
        if self._attention_backend in ("flash", "flash_varlen"):
            cosine, sine = rotary_emb
            query = flash_apply_rotary_emb(query.contiguous(), cosine, sine).to(hidden_states.dtype)
            key = flash_apply_rotary_emb(key.contiguous(), cosine, sine).to(hidden_states.dtype)
        else:
            cosine, sine = (embedding[None, :, None] for embedding in rotary_emb)
            rotary_dim = cosine.shape[-1] * 2
            rotated = []
            for tensor in (query, key):
                first, second = tensor[..., :rotary_dim].chunk(2, dim=-1)
                tensor = torch.cat(
                    [first * cosine - second * sine, first * sine + second * cosine, tensor[..., rotary_dim:]],
                    dim=-1,
                )
                rotated.append(tensor.to(hidden_states.dtype))
            query, key = rotated
        if kv_cache is not None:
            key = torch.cat([kv_cache[0].to(key), key], dim=1)
            value = torch.cat([kv_cache[1].to(value), value], dim=1)
        new_cache = (key, value)
        groups = attn.num_heads // attn.num_kv_heads
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)

        cross_query = cross_query.unflatten(-1, (attn.num_heads, attn.head_dim))
        cross_query = attn.q_layernorm_xattn(cross_query)
        if encoder_attention_mask is None:
            text_states = encoder_hidden_states.flatten(0, 2)
        else:
            # Match the reference GEMM shape by packing valid tokens before the KV projections.
            text_states = encoder_hidden_states[encoder_attention_mask]
            torch._check(text_states.shape[0] > 0)
        cross_kv = attn.linear_kv_xattn(text_states)
        cross_key, cross_value = cross_kv.unflatten(-1, (attn.num_kv_heads, 2 * attn.head_dim)).chunk(2, dim=-1)
        cross_key = attn.k_layernorm_xattn(cross_key)
        padded_shape = (*encoder_hidden_states.shape[:-1], attn.num_kv_heads, attn.head_dim)
        if encoder_attention_mask is None:
            cross_key = cross_key.reshape(padded_shape)
            cross_value = cross_value.reshape(padded_shape)
        else:
            padded_key = cross_key.new_zeros(padded_shape)
            padded_value = cross_value.new_zeros(padded_shape)
            padded_key[encoder_attention_mask] = cross_key
            padded_value[encoder_attention_mask] = cross_value
            cross_key, cross_value = padded_key, padded_value
        cross_key = cross_key.repeat_interleave(groups, dim=3)
        cross_value = cross_value.repeat_interleave(groups, dim=3)
        if self._attention_backend == "flash_varlen" and encoder_attention_mask is not None:
            # This backend accepts right padding; preserve the order of valid tokens when packing other masks.
            indices = (~encoder_attention_mask).to(torch.int32).argsort(dim=-1, stable=True)
            gather_indices = indices[..., None, None].expand_as(cross_key)
            cross_key = cross_key.gather(2, gather_indices)
            cross_value = cross_value.gather(2, gather_indices)
            encoder_attention_mask = encoder_attention_mask.gather(2, indices)
        self_outputs, cross_outputs = [], []
        for chunk, (start, end) in enumerate(kv_ranges):
            query_slice = slice(chunk * chunk_length, (chunk + 1) * chunk_length)
            self_outputs.append(
                dispatch_attention_fn(
                    query[:, query_slice],
                    key[:, start:end],
                    value[:, start:end],
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
            )
            mask = None if encoder_attention_mask is None else encoder_attention_mask[:, chunk, None, None, :]
            cross_outputs.append(
                dispatch_attention_fn(
                    cross_query[:, query_slice],
                    cross_key[:, chunk],
                    cross_value[:, chunk],
                    attn_mask=mask,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
            )
        self_output = torch.cat(self_outputs, dim=1).flatten(2)
        cross_output = torch.cat(cross_outputs, dim=1).flatten(2)
        hidden_states = torch.cat([self_output, cross_output], dim=-1)
        # Preserve the TP8 self/cross-attention interleave used by the checkpoint.
        hidden_states = hidden_states.unflatten(-1, (2, 8, attn.dim // 8)).transpose(-3, -2).flatten(-3)
        return attn.linear_proj(hidden_states), new_cache


class MagiAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MagiAttnProcessor
    _available_processors = [MagiAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, dim, num_heads, num_kv_heads, eps, zero_centered_gamma):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.linear_qkv = MagiQueryKeyValueProjection(dim, num_kv_heads * self.head_dim, eps)
        self.linear_kv_xattn = MagiTextKeyValueProjection(dim, num_kv_heads * self.head_dim)
        self.linear_proj = MagiFP32Linear(2 * dim, dim, bias=False)
        self.q_layernorm = MagiLayerNorm(self.head_dim, eps, zero_centered_gamma, upcast=True)
        self.k_layernorm = MagiLayerNorm(self.head_dim, eps, zero_centered_gamma, upcast=True)
        self.q_layernorm_xattn = MagiLayerNorm(self.head_dim, eps, zero_centered_gamma)
        self.k_layernorm_xattn = MagiLayerNorm(self.head_dim, eps, zero_centered_gamma)
        self.set_processor(MagiAttnProcessor())

    def forward(self, hidden_states, encoder_hidden_states, rotary_emb, encoder_attention_mask, kv_ranges, kv_cache):
        return self.processor(
            self, hidden_states, encoder_hidden_states, rotary_emb, encoder_attention_mask, kv_ranges, kv_cache
        )


class MagiAdaModulateLayer(nn.Module):
    def __init__(self, condition_dim, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Sequential(nn.Linear(condition_dim, 2 * dim))

    def forward(self, condition):
        return self.proj(self.act(condition))


class MagiMLP(nn.Module):
    def __init__(self, dim, ffn_dim, gated_linear_unit, eps):
        super().__init__()
        self.gated_linear_unit = gated_linear_unit
        self.layer_norm = nn.LayerNorm(dim, eps=eps)
        self.linear_fc1 = nn.Linear(dim, ffn_dim * (2 if gated_linear_unit else 1), bias=False)
        self.linear_fc2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.linear_fc1(self.layer_norm(hidden_states))
        if self.gated_linear_unit:
            gate, value = hidden_states.chunk(2, dim=-1)
            hidden_states = F.silu(gate) * value
        else:
            hidden_states = F.gelu(hidden_states)
        return self.linear_fc2(hidden_states)


class MagiTransformerBlock(nn.Module):
    def __init__(
        self, dim, condition_dim, ffn_dim, num_heads, num_kv_heads, gated_linear_unit, eps, zero_centered_gamma
    ):
        super().__init__()
        self.ada_modulate_layer = MagiAdaModulateLayer(condition_dim, dim)
        self.self_attention = MagiAttention(dim, num_heads, num_kv_heads, eps, zero_centered_gamma)
        self.self_attn_post_norm = MagiLayerNorm(dim, eps, zero_centered_gamma, upcast=True)
        self.mlp = MagiMLP(dim, ffn_dim, gated_linear_unit, eps)
        self.mlp_post_norm = MagiLayerNorm(dim, eps, zero_centered_gamma, upcast=True)

    def forward(
        self, hidden_states, condition, encoder_hidden_states, rotary_emb, encoder_attention_mask, kv_ranges, kv_cache
    ):
        residual = hidden_states
        hidden_states, new_cache = self.self_attention(
            hidden_states, encoder_hidden_states, rotary_emb, encoder_attention_mask, kv_ranges, kv_cache
        )
        gates = self.ada_modulate_layer(condition)
        gates = gates.float().tanh().to(gates.dtype)
        gates = gates.repeat_interleave(hidden_states.shape[1] // condition.shape[1], dim=1)
        gate_msa, gate_mlp = gates.chunk(2, dim=-1)
        hidden_states = self.self_attn_post_norm(hidden_states.float() * gate_msa.float())
        hidden_states = (hidden_states + residual.float()).to(residual.dtype)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_post_norm(hidden_states.float() * gate_mlp.float())
        return (hidden_states + residual.float()).to(residual.dtype), new_cache


class MagiTransformer3DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin):
    """
    MAGI-1 video Transformer with parallel self/cross-attention and chunk-level conditioning.

    Parameters:
        in_channels (`int`, defaults to 16): Number of input latent channels.
        out_channels (`int`, defaults to 16): Number of output latent channels.
        num_layers (`int`, defaults to 34): Number of Transformer blocks.
        num_attention_heads (`int`, defaults to 24): Number of query heads.
        num_key_value_heads (`int`, defaults to 8): Number of key/value heads.
        attention_head_dim (`int`, defaults to 128): Channels per head.
        ffn_dim (`int`, defaults to 12288): Feed-forward intermediate dimension.
        condition_dim (`int`, defaults to 768): Timestep and adaptive gating embedding dimension.
        caption_channels (`int`, defaults to 4096): Text encoder output dimension.
        caption_max_length (`int`, defaults to 800): Length of the learned null caption.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`): Temporal and spatial patch dimensions.
        frequency_embedding_size (`int`, defaults to 256): Sinusoidal timestep embedding dimension.
        gated_linear_unit (`bool`, defaults to `False`): Use SwiGLU instead of GELU, as in the 24B model.
        norm_eps (`float`, defaults to 1e-6): Layer normalization epsilon.
        zero_centered_gamma (`bool`, defaults to `True`): Add one to the custom normalization weights.
        x_rescale_factor (`float`, defaults to 1.0): Internal input scaling, inverted at the output.
        duplicate_channels (`bool`, defaults to `False`):
            Duplicate input channels and retain half the output, as in 24B.
        distilled (`bool`, defaults to `False`): Require the additional distillation timestep embedding.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MagiTransformerBlock", "MagiConditionEmbedding"]
    _repeated_blocks = ["MagiTransformerBlock"]
    _skip_layerwise_casting_patterns = ["x_embedder", "condition_embedder", "rope", "norm", "final_linear"]
    _keep_in_fp32_modules = [
        "x_embedder",
        "condition_embedder",
        "rope",
        "self_attn_post_norm",
        "q_layernorm",
        "k_layernorm",
        "mlp_post_norm",
        "final_layernorm",
        "final_linear",
    ]
    _skip_keys = ["kv_cache"]

    @register_to_config
    def __init__(
        self,
        in_channels=16,
        out_channels=16,
        num_layers=34,
        num_attention_heads=24,
        num_key_value_heads=8,
        attention_head_dim=128,
        ffn_dim=12288,
        condition_dim=768,
        caption_channels=4096,
        caption_max_length=800,
        patch_size=(1, 2, 2),
        frequency_embedding_size=256,
        gated_linear_unit=False,
        norm_eps=1e-6,
        zero_centered_gamma=True,
        x_rescale_factor=1.0,
        duplicate_channels=False,
        distilled=False,
    ):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        if dim % 8 or attention_head_dim % 8 or num_attention_heads % num_key_value_heads:
            raise ValueError("Hidden/head dimensions must be divisible by 8, and query heads by key/value heads.")
        self.x_embedder = MagiPatchEmbedding(
            in_channels * (2 if duplicate_channels else 1), dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.condition_embedder = MagiConditionEmbedding(
            caption_channels, caption_max_length, dim, condition_dim, frequency_embedding_size
        )
        self.rope = MagiRotaryEmbedding(attention_head_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                MagiTransformerBlock(
                    dim,
                    condition_dim,
                    ffn_dim,
                    num_attention_heads,
                    num_key_value_heads,
                    gated_linear_unit,
                    norm_eps,
                    zero_centered_gamma,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = MagiLayerNorm(dim, norm_eps, zero_centered_gamma, upcast=True)
        self.final_linear = MagiFP32Linear(
            dim, math.prod(patch_size) * out_channels * (2 if duplicate_channels else 1), bias=False
        )
        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        caption_dropout_mask: torch.Tensor | None = None,
        timestep_delta: torch.Tensor | None = None,
        kv_ranges: tuple[tuple[int, int], ...] | None = None,
        kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
        cache_token_count: int | None = None,
        cache_device: str | torch.device | None = None,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ):
        """
        Predict flow velocities for equally sized video chunks.

        Args:
            hidden_states (`torch.Tensor`): Latents of shape `(batch, channels, frames, height, width)`.
            encoder_hidden_states (`torch.Tensor`):
                Text features `(batch, length, channels)` or `(batch, chunks, length, channels)`.
            timestep (`torch.Tensor`): Timesteps in [0, 1], shaped `(batch,)` or `(batch, chunks)`.
            encoder_attention_mask (`torch.Tensor`, optional):
                Boolean text keep-mask `(batch, length)` or `(batch, chunks, length)`.
            caption_dropout_mask (`torch.Tensor`, optional):
                Select the unconditional adaptive embedding per batch item. Shape `(1,)` broadcasts a single projected
                condition across the batch. This does not replace cross-attention text features.
            timestep_delta (`torch.Tensor`, optional): Extra distillation timesteps, broadcastable to `timestep`.
            kv_ranges (`tuple`, optional):
                Exclusive `(start, end)` token ranges, one per current chunk, indexing cached plus current tokens.
                Defaults to chunk-causal attention over all preceding chunks.
            kv_cache (`tuple`, optional):
                Per-layer `(key, value)` tensors shaped `(batch, cached_tokens, kv_heads, head_dim)`. Contains a
                complete clean prefix and is never modified in place.
            use_cache (`bool`, defaults to `False`):
                Return cached prefix plus current keys and values. Only reuse entries belonging to clean, finalized
                chunks.
            cache_token_count (`int`, optional):
                Retain only this many leading tokens in each returned cache, copying them before the next layer.
                Requires `use_cache=True` and complete temporal patches. Does not change attention or predictions.
            cache_device (`str` or `torch.device`, optional):
                Device for returned cache tensors. Set to `"cpu"` to offload each layer's cache as it is produced.
                Requires `use_cache=True`; input caches are moved to the compute device one layer at a time.
            attention_kwargs (`dict`, optional): Keyword arguments for LoRA scaling.
            return_dict (`bool`, defaults to `True`): Return a structured output instead of a tuple.

        Returns:
            `MagiTransformer3DModelOutput` or `tuple`: Predicted velocities and, when requested, keys and values.
        """
        if (
            hidden_states.ndim != 5
            or hidden_states.shape[1] != self.config.in_channels
            or any(size <= 0 for size in hidden_states.shape)
        ):
            raise ValueError("Expected nonempty latents shaped (batch, in_channels, frames, height, width).")
        batch_size, _, frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.config.patch_size
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError("Video dimensions must be divisible by the patch dimensions.")
        timestep = timestep.reshape(batch_size, -1)
        num_chunks = timestep.shape[1]
        if frames // patch_t % num_chunks:
            raise ValueError("Temporal patches must divide evenly into timestep chunks.")
        if self.config.distilled and timestep_delta is None:
            raise ValueError("Distilled models require timestep_delta from the distillation schedule.")
        if timestep_delta is not None:
            timestep_delta = torch.broadcast_to(timestep_delta, timestep.shape)
        if encoder_hidden_states.ndim == 3:
            encoder_hidden_states = encoder_hidden_states[:, None].expand(-1, num_chunks, -1, -1)
        if encoder_hidden_states.shape[:2] != (batch_size, num_chunks):
            raise ValueError("Text features must match the batch and timestep chunk dimensions.")
        if encoder_attention_mask is not None:
            if encoder_attention_mask.ndim == 2:
                encoder_attention_mask = encoder_attention_mask[:, None].expand(-1, num_chunks, -1)
            encoder_attention_mask = encoder_attention_mask.bool()
        if caption_dropout_mask is None:
            caption_dropout_mask = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.bool)
        spatial_tokens = (height // patch_h) * (width // patch_w)
        if kv_cache is not None and len(kv_cache) != len(self.transformer_blocks):
            raise ValueError("kv_cache must contain one key/value pair per Transformer block.")
        cached_tokens = 0 if kv_cache is None else kv_cache[0][0].shape[1]
        if cached_tokens % spatial_tokens:
            raise ValueError("The cached prefix must contain complete temporal patches at the current resolution.")
        sequence_length = frames // patch_t * spatial_tokens
        if (cache_token_count is not None or cache_device is not None) and not use_cache:
            raise ValueError("Cache retention options require use_cache=True.")
        if cache_token_count is not None and (
            not isinstance(cache_token_count, int)
            or isinstance(cache_token_count, bool)
            or not 0 < cache_token_count <= cached_tokens + sequence_length
            or cache_token_count % spatial_tokens
        ):
            raise ValueError("cache_token_count must retain complete temporal patches within the available tokens.")
        chunk_length = sequence_length // num_chunks
        if kv_ranges is None:
            kv_ranges = tuple((0, cached_tokens + (chunk + 1) * chunk_length) for chunk in range(num_chunks))
        if len(kv_ranges) != num_chunks or any(
            not 0 <= start < end <= cached_tokens + sequence_length for start, end in kv_ranges
        ):
            raise ValueError("Each chunk must have a nonempty key/value range within the available tokens.")
        hidden_states = hidden_states * self.config.x_rescale_factor
        if self.config.duplicate_channels:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=1)
        hidden_states = self.x_embedder(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous().to(self.dtype)
        rotary_emb = self.rope(frames // patch_t, height // patch_h, width // patch_w, cached_tokens // spatial_tokens)
        condition, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, caption_dropout_mask, timestep_delta, hidden_states.dtype
        )
        condition = condition.to(hidden_states.dtype)
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)
        new_cache = []
        for index, block in enumerate(self.transformer_blocks):
            layer_cache = None if kv_cache is None else kv_cache[index]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, cached = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    condition,
                    encoder_hidden_states,
                    rotary_emb,
                    encoder_attention_mask,
                    kv_ranges,
                    layer_cache,
                )
            else:
                hidden_states, cached = block(
                    hidden_states,
                    condition,
                    encoder_hidden_states,
                    rotary_emb,
                    encoder_attention_mask,
                    kv_ranges,
                    layer_cache,
                )
            if use_cache:
                if cache_token_count is not None or cache_device is not None:
                    cached = tuple(
                        tensor[:, :cache_token_count].to(
                            device=cache_device if cache_device is not None else tensor.device, copy=True
                        )
                        for tensor in cached
                    )
                new_cache.append(cached)
            del cached
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.final_linear(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, frames // patch_t, height // patch_h, width // patch_w, patch_t, patch_h, patch_w, -1
        )
        hidden_states = (
            hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch_size, -1, frames, height, width).contiguous()
        )
        hidden_states = hidden_states[:, : self.config.out_channels] / self.config.x_rescale_factor
        if not return_dict:
            return (hidden_states, tuple(new_cache)) if use_cache else (hidden_states,)
        return MagiTransformer3DModelOutput(sample=hidden_states, kv_cache=tuple(new_cache) if use_cache else None)
