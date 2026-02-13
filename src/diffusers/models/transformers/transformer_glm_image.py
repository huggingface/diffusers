# Copyright 2025 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import LayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class GlmImageCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, pooled_projection_dim: int, timesteps_dim: int = 256):
        super().__init__()

        self.time_proj = Timesteps(num_channels=timesteps_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.condition_proj = Timesteps(num_channels=condition_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=timesteps_dim, time_embed_dim=embedding_dim)
        self.condition_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(
        self,
        timestep: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)

        crop_coords_proj = self.condition_proj(crop_coords.flatten()).view(crop_coords.size(0), -1)
        target_size_proj = self.condition_proj(target_size.flatten()).view(target_size.size(0), -1)

        # (B, 2 * condition_dim)
        condition_proj = torch.cat([crop_coords_proj, target_size_proj], dim=1)

        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (B, embedding_dim)
        condition_emb = self.condition_embedder(condition_proj.to(dtype=hidden_dtype))  # (B, embedding_dim)

        conditioning = timesteps_emb + condition_emb
        conditioning = F.silu(conditioning)

        return conditioning


class GlmImageImageProjector(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size, channel, post_patch_height, self.patch_size, post_patch_width, self.patch_size
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        hidden_states = self.proj(hidden_states)

        return hidden_states


class GlmImageAdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        norm_hidden_states = self.norm(hidden_states).to(dtype=dtype)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states).to(dtype=dtype)

        emb = self.linear(temb)
        (
            shift_msa,
            c_shift_msa,
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)

        hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class GlmImageLayerKVCache:
    """KV cache for GlmImage model.
    Supports per-sample caching for batch processing where each sample may have different condition images.
    """

    def __init__(self):
        self.k_caches: list[torch.Tensor | None] = []
        self.v_caches: list[torch.Tensor | None] = []
        self.mode: str | None = None  # "write", "read", "skip"
        self.current_sample_idx: int = 0  # Current sample index for writing

    def store(self, k: torch.Tensor, v: torch.Tensor):
        """Store KV cache for the current sample."""
        # k, v shape: (1, seq_len, num_heads, head_dim)
        if len(self.k_caches) <= self.current_sample_idx:
            # First time storing for this sample
            self.k_caches.append(k)
            self.v_caches.append(v)
        else:
            # Append to existing cache for this sample (multiple condition images)
            self.k_caches[self.current_sample_idx] = torch.cat([self.k_caches[self.current_sample_idx], k], dim=1)
            self.v_caches[self.current_sample_idx] = torch.cat([self.v_caches[self.current_sample_idx], v], dim=1)

    def get(self, k: torch.Tensor, v: torch.Tensor):
        """Get combined KV cache for all samples in the batch.

        Args:
            k: Current key tensor, shape (batch_size, seq_len, num_heads, head_dim)
            v: Current value tensor, shape (batch_size, seq_len, num_heads, head_dim)
        Returns:
            Combined key and value tensors with cached values prepended.
        """
        batch_size = k.shape[0]
        num_cached_samples = len(self.k_caches)
        if num_cached_samples == 0:
            return k, v
        if num_cached_samples == 1:
            # Single cache, expand for all batch samples (shared condition images)
            k_cache_expanded = self.k_caches[0].expand(batch_size, -1, -1, -1)
            v_cache_expanded = self.v_caches[0].expand(batch_size, -1, -1, -1)
        elif num_cached_samples == batch_size:
            # Per-sample cache, concatenate along batch dimension
            k_cache_expanded = torch.cat(self.k_caches, dim=0)
            v_cache_expanded = torch.cat(self.v_caches, dim=0)
        else:
            # Mismatch: try to handle by repeating the caches
            # This handles cases like num_images_per_prompt > 1
            repeat_factor = batch_size // num_cached_samples
            if batch_size % num_cached_samples == 0:
                k_cache_list = []
                v_cache_list = []
                for i in range(num_cached_samples):
                    k_cache_list.append(self.k_caches[i].expand(repeat_factor, -1, -1, -1))
                    v_cache_list.append(self.v_caches[i].expand(repeat_factor, -1, -1, -1))
                k_cache_expanded = torch.cat(k_cache_list, dim=0)
                v_cache_expanded = torch.cat(v_cache_list, dim=0)
            else:
                raise ValueError(
                    f"Cannot match {num_cached_samples} cached samples to batch size {batch_size}. "
                    f"Batch size must be a multiple of the number of cached samples."
                )

        k_combined = torch.cat([k_cache_expanded, k], dim=1)
        v_combined = torch.cat([v_cache_expanded, v], dim=1)
        return k_combined, v_combined

    def clear(self):
        self.k_caches = []
        self.v_caches = []
        self.mode = None
        self.current_sample_idx = 0

    def next_sample(self):
        """Move to the next sample for writing."""
        self.current_sample_idx += 1


class GlmImageKVCache:
    """Container for all layers' KV caches.
    Supports per-sample caching for batch processing where each sample may have different condition images.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.caches = [GlmImageLayerKVCache() for _ in range(num_layers)]

    def __getitem__(self, layer_idx: int) -> GlmImageLayerKVCache:
        return self.caches[layer_idx]

    def set_mode(self, mode: str):
        if mode is not None and mode not in ["write", "read", "skip"]:
            raise ValueError(f"Invalid mode: {mode}, must be one of 'write', 'read', 'skip'")
        for cache in self.caches:
            cache.mode = mode

    def next_sample(self):
        """Move to the next sample for writing. Call this after processing
        all condition images for one batch sample."""
        for cache in self.caches:
            cache.next_sample()

    def clear(self):
        for cache in self.caches:
            cache.clear()


class GlmImageAttnProcessor:
    """
    Processor for implementing scaled dot-product attention for the GlmImage model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.

    The processor supports passing an attention mask for text tokens. The attention mask should have shape (batch_size,
    text_seq_length) where 1 indicates a non-padded token and 0 indicates a padded token.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("GlmImageAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: GlmImageLayerKVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = encoder_hidden_states.dtype

        batch_size, text_seq_length, embed_dim = encoder_hidden_states.shape
        batch_size, image_seq_length, embed_dim = hidden_states.shape
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(dtype=dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype=dtype)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            query[:, text_seq_length:, :, :] = apply_rotary_emb(
                query[:, text_seq_length:, :, :], image_rotary_emb, sequence_dim=1, use_real_unbind_dim=-2
            )
            key[:, text_seq_length:, :, :] = apply_rotary_emb(
                key[:, text_seq_length:, :, :], image_rotary_emb, sequence_dim=1, use_real_unbind_dim=-2
            )

        if kv_cache is not None:
            if kv_cache.mode == "write":
                kv_cache.store(key, value)
            elif kv_cache.mode == "read":
                key, value = kv_cache.get(key, value)
            elif kv_cache.mode == "skip":
                pass

        # 4. Attention
        if attention_mask is not None:
            text_attn_mask = attention_mask
            assert text_attn_mask.dim() == 2, "the shape of text_attn_mask should be (batch_size, text_seq_length)"
            text_attn_mask = text_attn_mask.float().to(query.device)
            mix_attn_mask = torch.ones((batch_size, text_seq_length + image_seq_length), device=query.device)
            mix_attn_mask[:, :text_seq_length] = text_attn_mask
            mix_attn_mask = mix_attn_mask.unsqueeze(2)
            attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)
            attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class GlmImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()

        # 1. Attention
        self.norm1 = GlmImageAdaLayerNormZero(time_embed_dim, dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-5,
            processor=GlmImageAttnProcessor(),
        )

        # 2. Feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: dict[str, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: GlmImageLayerKVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Timestep conditioning
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)

        # 2. Attention
        attention_kwargs = attention_kwargs or {}

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            **attention_kwargs,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)

        # 3. Feedforward
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class GlmImageRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int, theta: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)].float() / dim_h)
        )
        w_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32)[: (dim_w // 2)].float() / dim_w)
        )
        h_seq = torch.arange(height)
        w_seq = torch.arange(width)
        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)

        # Create position matrices for height and width
        # [height, 1, dim//4] and [1, width, dim//4]
        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)
        # Broadcast freqs_h and freqs_w to [height, width, dim//4]
        freqs_h = freqs_h.expand(height, width, -1)
        freqs_w = freqs_w.expand(height, width, -1)

        # Concatenate along last dimension to get [height, width, dim//2]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)  # [height, width, dim]
        freqs = freqs.reshape(height * width, -1)
        return (freqs.cos(), freqs.sin())


class GlmImageAdaLayerNormContinuous(nn.Module):
    """
    GlmImage-only final AdaLN: LN(x) -> Linear(cond) -> chunk -> affine. Matches Megatron: **no activation** before the
    Linear on conditioning embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # *** NO SiLU here ***
        emb = self.linear(conditioning_embedding.to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class GlmImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    r"""
    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, defaults to `40`):
            The number of channels in each head.
        num_attention_heads (`int`, defaults to `64`):
            The number of heads to use for multi-head attention.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_embed_dim (`int`, defaults to `1472`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        condition_dim (`int`, defaults to `256`):
            The embedding dimension of the input SDXL-style resolution conditions (original_size, target_size,
            crop_coords).
        pos_embed_max_size (`int`, defaults to `128`):
            The maximum resolution of the positional embeddings, from which slices of shape `H x W` are taken and added
            to input patched latents, where `H` and `W` are the latent height and width respectively. A value of 128
            means that the maximum supported height and width for image generation is `128 * vae_scale_factor *
            patch_size => 128 * 8 * 2 => 2048`.
        sample_size (`int`, defaults to `128`):
            The base resolution of input latents. If height/width is not provided during generation, this value is used
            to determine the resolution as `sample_size * vae_scale_factor => 128 * 8 => 1024`
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "GlmImageTransformerBlock",
        "GlmImageImageProjector",
        "GlmImageImageProjector",
    ]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm", "proj_out"]
    _skip_keys = ["kv_caches"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        text_embed_dim: int = 1472,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        prior_vq_quantizer_codebook_size: int = 16384,
    ):
        super().__init__()

        # GlmImage uses 2 additional SDXL-like conditions - target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        pooled_projection_dim = 2 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels

        # 1. RoPE
        self.rope = GlmImageRotaryPosEmbed(attention_head_dim, patch_size, theta=10000.0)

        # 2. Patch & Text-timestep embedding
        self.image_projector = GlmImageImageProjector(in_channels, inner_dim, patch_size)
        self.glyph_projector = FeedForward(text_embed_dim, inner_dim, inner_dim=inner_dim, activation_fn="gelu")
        self.prior_token_embedding = nn.Embedding(prior_vq_quantizer_codebook_size, inner_dim)
        self.prior_projector = FeedForward(inner_dim, inner_dim, inner_dim=inner_dim, activation_fn="linear-silu")

        self.time_condition_embed = GlmImageCombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=time_embed_dim,
        )

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                GlmImageTransformerBlock(inner_dim, num_attention_heads, attention_head_dim, time_embed_dim)
                for _ in range(num_layers)
            ]
        )

        # 4. Output projection
        self.norm_out = GlmImageAdaLayerNormContinuous(inner_dim, time_embed_dim, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prior_token_id: torch.Tensor,
        prior_token_drop: torch.Tensor,
        timestep: torch.LongTensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        attention_mask: torch.Tensor | None = None,
        kv_caches: GlmImageKVCache | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        batch_size, num_channels, height, width = hidden_states.shape

        # 1. RoPE
        if image_rotary_emb is None:
            image_rotary_emb = self.rope(hidden_states)

        # 2. Patch & Timestep embeddings
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = self.image_projector(hidden_states)
        encoder_hidden_states = self.glyph_projector(encoder_hidden_states)
        prior_embedding = self.prior_token_embedding(prior_token_id)
        prior_embedding[prior_token_drop] *= 0.0
        prior_hidden_states = self.prior_projector(prior_embedding)

        hidden_states = hidden_states + prior_hidden_states

        temb = self.time_condition_embed(timestep, target_size, crop_coords, hidden_states.dtype)

        # 3. Transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    attention_kwargs,
                    kv_caches[idx] if kv_caches is not None else None,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    attention_kwargs,
                    kv_cache=kv_caches[idx] if kv_caches is not None else None,
                )

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, -1, p, p)

        # Rearrange tensor from (B, H_p, W_p, C, p, p) to (B, C, H_p * p, W_p * p)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
