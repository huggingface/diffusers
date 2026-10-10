# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
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
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...models.modeling_utils import ModelMixin
from ...models.normalization import RMSNorm
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_outputs import Transformer2DModelOutput


ADALN_EMBED_DIM = 256


# Copied from diffusers.models.transformers.transformer_z_image.TimestepEmbedder
class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


# Copied from diffusers.models.transformers.transformer_z_image.select_per_token
def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


# Copied from diffusers.models.transformers.transformer_z_image.FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x) -> torch.Tensor:
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


# Copied from diffusers.models.transformers.transformer_z_image.FinalLayer
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None) -> torch.Tensor:
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            # Original global modulation
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x


# Copied from diffusers.models.transformers.transformer_z_image.RopeEmbedder
class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: list[int], end: list[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


def compute_gist_count(
    text_lengths: list[int], min_gist: int, max_gist: int, gist_step: int, min_text_len: int, max_text_len: int
) -> list[int]:
    """Number of gist tokens for each prompt, from its text length in tokens.

    Lengths between `min_text_len` and `max_text_len` map linearly onto `[min_gist, max_gist]`, rounded up to a
    multiple of `gist_step`.
    """
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        quantized = int(math.ceil(raw / gist_step)) * gist_step
        counts.append(max(min_gist, min(max_gist, quantized)))
    return counts


class BriaFibo2AttnProcessor:
    """Self-attention (with RoPE when `freqs_cis` is given) or cross-attention to `encoder_hidden_states`."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "BriaFibo2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
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

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)

        return attn.to_out[0](hidden_states)


class BriaFibo2Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = BriaFibo2AttnProcessor
    _available_processors = [BriaFibo2AttnProcessor]
    # The processor reads the separate to_q, to_k and to_v projections
    _supports_qkv_fusion = False

    def __init__(self, query_dim: int, heads: int, dim_head: int, kv_dim: int | None = None, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        kv_dim = kv_dim or query_dim
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.set_processor(self._default_processor_cls())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, freqs_cis)


@maybe_allow_in_graph
class BriaFibo2TransformerBlock(nn.Module):
    """Z-Image's block. With `cross_attention_dim` it becomes an injection block that also reads the text encoder."""

    def __init__(self, dim: int, n_heads: int, norm_eps: float, cross_attention_dim: int | None = None):
        super().__init__()
        self.attention = BriaFibo2Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=(dim * 8 // 3) // 64 * 64)

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

        if cross_attention_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps=norm_eps)
            self.cross_attention = BriaFibo2Attention(
                dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps
            )
            self.cross_attn_norm2 = RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attention = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation: different modulation for noisy/clean tokens
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)

            scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
            scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

            gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
            gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

            scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
            scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

            scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
            scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
            gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
            gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
        else:
            # Global modulation: same modulation for all tokens (avoid double select)
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # Attention block
        attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)

        # Gated cross-attention to the text encoder (injection blocks only)
        if self.cross_attention is not None:
            cross_out = self.cross_attention(
                self.cross_attn_norm1(x),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(cross_out)

        # FFN block
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))

        return x


class BriaFibo2PerceiverLayer(nn.Module):
    """Gist tokens read the text encoder (cross-attention), then each other (self-attention), then an FFN."""

    def __init__(self, dim: int, n_heads: int, cross_attention_dim: int, norm_eps: float):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.cross_attention = BriaFibo2Attention(
            dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps
        )
        self.self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.self_attention = BriaFibo2Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim=dim, hidden_dim=(dim * 8 // 3) // 64 * 64)

    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = latents + self.cross_attention(
            self.cross_attn_norm(latents),
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        latents = latents + self.self_attention(self.self_attn_norm(latents), attention_mask=attention_mask)
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class BriaFibo2PerceiverResampler(nn.Module):
    """Squeezes the text encoder's hidden states into `num_gist_tokens` gist tokens."""

    def __init__(
        self, dim: int, n_heads: int, cross_attention_dim: int, num_queries: int, num_layers: int, norm_eps: float
    ):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList(
            [BriaFibo2PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps) for _ in range(num_layers)]
        )
        self.norm_out = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        num_gist_tokens: int,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `attention_mask` marks each prompt's real gist tokens when a batch pads them to `num_gist_tokens`
        batch_size = encoder_hidden_states.shape[0]
        latents = self.latent_queries[:num_gist_tokens].unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states, encoder_attention_mask, attention_mask)
        return self.norm_out(latents)


class BriaFibo2Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BriaFibo2TransformerBlock", "BriaFibo2PerceiverLayer"]
    _repeated_blocks = ["BriaFibo2TransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        patch_size: int = 2,
        dim: int = 3584,
        n_layers: int = 44,
        n_refiner_layers: int = 2,
        n_heads: int = 28,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 7680,
        injection_layer_ids: tuple[int, ...] = (6, 14, 22, 30, 38),
        perceiver_num_layers: int = 2,
        min_num_gist_tokens: int = 256,
        max_num_gist_tokens: int = 512,
        gist_step: int = 64,
        gist_min_text_len: int = 256,
        gist_max_text_len: int = 2048,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (2048, 512, 512),
    ):
        super().__init__()
        self.gradient_checkpointing = False

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self.noise_refiner = nn.ModuleList(
            [BriaFibo2TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_refiner_layers)]
        )
        self.perceiver = BriaFibo2PerceiverResampler(
            dim, n_heads, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps
        )
        self.layers = nn.ModuleList(
            [
                BriaFibo2TransformerBlock(
                    dim, n_heads, norm_eps, cross_attention_dim=cap_feat_dim if i in injection_layer_ids else None
                )
                for i in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size * patch_size * in_channels)

    @staticmethod
    # Copied from diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel.create_coordinate_grid
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # 2x2 latent patches become tokens, features ordered (channel, dy, dx) as in training
        batch_size, channels, height, width = latents.shape
        patch_size = self.config.patch_size
        latents = latents.view(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
        return latents.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size, (height // patch_size) * (width // patch_size), -1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        context_latents: list[torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, height, width)`):
                Normalized VAE latents of the image being generated.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Noise level in `[0, 1]`: 1 is pure noise, 0 is a clean image.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, 1 + len(injection_layer_ids), text_len, cap_feat_dim)`):
                Text encoder features: the Perceiver's bundle first, then one bundle per injection block.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_len)`, *optional*):
                Bool mask of the real text tokens, for a batch of prompts padded to the same length. Leave it as `None`
                when nothing is padded.
            context_latents (`list[torch.Tensor]`, *optional*):
                Normalized VAE latents of the images to edit, one tensor of shape `(batch_size, in_channels, height,
                width)` per image, in order. Each image can have its own size. Their tokens condition the generation as
                clean tokens and are not predicted.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`: The predicted velocity, with the shape of
            `hidden_states`. If `return_dict` is `False`, a tuple whose first element is that tensor.
        """
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.config.patch_size
        height_tokens, width_tokens = height // patch_size, width // patch_size
        num_image_tokens = height_tokens * width_tokens
        device = hidden_states.device

        # 1. Timestep embeddings: image tokens get the real timestep, clean tokens (the gist and the context) get t = 0
        adaln_noisy = self.t_embedder(timestep * self.config.t_scale)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep))

        # 2. Patchify the image and the context images, which share the patch embedding. The image sits on RoPE plane
        # `max_num_gist_tokens + 1` whatever the gist count, and context image k on the k-th plane after it with its
        # own rows and columns
        image_plane = self.config.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [self.create_coordinate_grid((1, height_tokens, width_tokens), (image_plane, 0, 0), device)]
        for k, latents in enumerate(context_latents or []):
            tokens.append(self._patchify(latents))
            context_size = (1, latents.shape[2] // patch_size, latents.shape[3] // patch_size)
            ids.append(self.create_coordinate_grid(context_size, (image_plane + k + 1, 0, 0), device))
        hidden_states = self.x_embedder(torch.cat(tokens, dim=1))
        image_freqs = self.rope_embedder(torch.cat([grid.flatten(0, 2) for grid in ids])).unsqueeze(0)

        # 3. Noise refiner. With context images, the noise mask marks the image tokens as the only noisy ones
        refiner_noise_mask = None
        if context_latents:
            refiner_noise_mask = torch.zeros(batch_size, hidden_states.shape[1], dtype=torch.long, device=device)
            refiner_noise_mask[:, :num_image_tokens] = 1
        for block in self.noise_refiner:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, None, image_freqs, adaln_noisy, refiner_noise_mask, adaln_noisy, adaln_clean
                )
            else:
                hidden_states = block(
                    hidden_states,
                    None,
                    image_freqs,
                    adaln_input=adaln_noisy,
                    noise_mask=refiner_noise_mask,
                    adaln_noisy=adaln_noisy,
                    adaln_clean=adaln_clean,
                )

        # 4. Gist count per prompt from its text length, then the Perceiver turns the first bundle into gist tokens.
        # In a batch, prompts with fewer gist tokens are padded to the largest count and masked out
        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[2]] * batch_size
        gist_counts = compute_gist_count(
            text_lengths,
            min_gist=self.config.min_num_gist_tokens,
            max_gist=self.config.max_num_gist_tokens,
            gist_step=self.config.gist_step,
            min_text_len=self.config.gist_min_text_len,
            max_text_len=self.config.gist_max_text_len,
        )
        num_gist_tokens = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < num_gist_tokens:
            gist_mask = (
                torch.arange(num_gist_tokens, device=device) < torch.tensor(gist_counts, device=device)[:, None]
            )
        gist = self.perceiver(encoder_hidden_states[:, 0], num_gist_tokens, encoder_attention_mask, gist_mask)
        gist_ids = self.create_coordinate_grid((num_gist_tokens, 1, 1), (1, 0, 0), device).flatten(0, 2)

        # 5. One stream: gist tokens first, then the image tokens, then the context tokens, the order fibo-2 was trained
        # with. When the batch pads gist tokens, the gist goes last instead, so the padding ends every row as varlen
        # attention backends expect. Positions come from RoPE, so the order changes nothing but float rounding. The
        # noise mask marks which tokens are noisy
        num_context_tokens = hidden_states.shape[1] - num_image_tokens
        gist_freqs = self.rope_embedder(gist_ids).unsqueeze(0)
        attention_mask = None
        if gist_mask is None:
            image_start = num_gist_tokens
            hidden_states = torch.cat([gist, hidden_states], dim=1)
            freqs_cis = torch.cat([gist_freqs, image_freqs], dim=1)
        else:
            image_start = 0
            hidden_states = torch.cat([hidden_states, gist], dim=1)
            freqs_cis = torch.cat([image_freqs, gist_freqs], dim=1)
            image_mask = torch.ones(batch_size, num_image_tokens + num_context_tokens, dtype=torch.bool, device=device)
            attention_mask = torch.cat([image_mask, gist_mask], dim=1)
        noise_mask = torch.zeros(batch_size, hidden_states.shape[1], dtype=torch.long, device=device)
        noise_mask[:, image_start : image_start + num_image_tokens] = 1

        # 6. The main blocks. Injection blocks also read their own bundle, in `injection_layer_ids` order
        bundle_index = {layer_id: index + 1 for index, layer_id in enumerate(self.config.injection_layer_ids)}
        for layer_id, block in enumerate(self.layers):
            text = encoder_hidden_states[:, bundle_index[layer_id]] if layer_id in bundle_index else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    None,
                    noise_mask,
                    adaln_noisy,
                    adaln_clean,
                    text,
                    encoder_attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    noise_mask=noise_mask,
                    adaln_noisy=adaln_noisy,
                    adaln_clean=adaln_clean,
                    encoder_hidden_states=text,
                    encoder_attention_mask=encoder_attention_mask,
                )

        # 7. Final layer on the image tokens only, then unpatchify back to latents
        hidden_states = self.final_layer(hidden_states[:, image_start : image_start + num_image_tokens], c=adaln_noisy)
        hidden_states = hidden_states.view(batch_size, height_tokens, width_tokens, channels, patch_size, patch_size)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
