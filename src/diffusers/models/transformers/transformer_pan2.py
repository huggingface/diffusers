# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import apply_lora_scale, logging
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.transformers.transformer_flux._get_projections with FluxAttention->PAN2Attention
def _get_projections(attn: "PAN2Attention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


# Copied from diffusers.models.transformers.transformer_flux._get_fused_projections with FluxAttention->PAN2Attention
def _get_fused_projections(attn: "PAN2Attention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


# Copied from diffusers.models.transformers.transformer_flux._get_qkv_projections with FluxAttention->PAN2Attention
def _get_qkv_projections(attn: "PAN2Attention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class PAN2AttnProcessor:
    r"""
    Attention processor for PAN2. Without `encoder_hidden_states` it runs plain self-attention (token refiner). With
    `encoder_hidden_states` it runs joint attention over `[video; text]`, where each stream has its own projections and
    only the video stream is rotated.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "PAN2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query).to(encoder_value.dtype)
            encoder_key = attn.norm_added_k(encoder_key).to(encoder_value.dtype)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

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
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if encoder_hidden_states is None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]], dim=1
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class PAN2Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = PAN2AttnProcessor
    _available_processors = [PAN2AttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        processor=None,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        self.use_bias = bias
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = bias

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=bias), nn.Dropout(0.0)])

        if added_kv_proj_dim is not None:
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


class PAN2RotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: list[int],
        theta: float = 256.0,
        temporal_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta
        self.temporal_scale = temporal_scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [T, H, W]
        grid = torch.stack(grid, dim=0)  # [3, T, H, W]
        # Only the frame positions are rescaled, to the frame rate the rotary embedding was trained at.
        grid[0] = grid[0] * self.temporal_scale

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (T * H * W, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (T * H * W, D / 2)
        return freqs_cos, freqs_sin


class PAN2TokenRefinerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = PAN2Attention(dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps, bias=False)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff = FeedForward(dim, inner_dim=int(dim * mlp_ratio), activation_fn="swiglu", bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class PAN2TransformerBlock(nn.Module):
    r"""
    Joint-attention block. Video and text tokens attend to each other in a single attention call, but each stream has
    its own QKV / output projections and feed-forward network. The video stream is modulated by the timestep, the text
    stream by a learned, timestep-independent table.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        context_mlp_ratio: float,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, norm_type="layer_norm")
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.scale_shift_table_context = nn.Parameter(torch.zeros(6 * dim))

        self.attn = PAN2Attention(
            dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps, added_kv_proj_dim=dim
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim, inner_dim=int(dim * mlp_ratio), activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = FeedForward(dim, inner_dim=int(dim * context_mlp_ratio), activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        (
            context_shift_msa,
            context_scale_msa,
            context_gate_msa,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.scale_shift_table_context.to(temb.dtype).chunk(6, dim=-1)
        norm_encoder_hidden_states = (
            self.norm1_context(encoder_hidden_states) * (1 + context_scale_msa) + context_shift_msa
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            rotary_emb=rotary_emb,
        )

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * context_gate_msa

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (
            self.norm2_context(encoder_hidden_states) * (1 + context_scale_mlp) + context_shift_mlp
        )

        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.ff_context(norm_encoder_hidden_states) * context_gate_mlp

        return hidden_states, encoder_hidden_states


class PAN2Transformer3DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin, CacheMixin):
    r"""
    The video generation transformer of PAN2.

    A stack of joint-attention blocks over the concatenation of the patchified video latents and the refined text
    tokens. The video input carries the noisy latents, the conditioning latents and a conditioning mask stacked along
    the channel axis, so the same model serves text-to-video and image-to-video.

    Args:
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The `(t, h, w)` patch size of the patch embedding.
        in_channels (`int`, defaults to `97`):
            The number of input channels: noisy latents, conditioning latents and one conditioning mask channel.
        out_channels (`int`, defaults to `48`):
            The number of output channels.
        num_attention_heads (`int`, defaults to `32`):
            The number of attention heads.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head.
        num_layers (`int`, defaults to `44`):
            The number of joint-attention blocks.
        num_refiner_layers (`int`, defaults to `4`):
            The number of token refiner blocks applied to the text embeddings.
        mlp_ratio (`float`, defaults to `8.0`):
            The ratio of the video feed-forward inner dimension to the hidden dimension.
        context_mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the text feed-forward inner dimension to the hidden dimension.
        refiner_mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the token refiner feed-forward inner dimension to the hidden dimension.
        text_embed_dim (`int`, defaults to `4096`):
            The dimension of the text encoder hidden states.
        rope_axes_dim (`tuple[int, int, int]`, defaults to `(16, 56, 56)`):
            The number of channels of each `(t, h, w)` axis in the rotary embedding.
        rope_theta (`float`, defaults to `256.0`):
            The base of the rotary embedding.
        rope_temporal_scale (`float`, defaults to `16 / 24`):
            The factor the frame positions of the rotary embedding are multiplied with: the reference frame rate of the
            rotary embedding over the frame rate of the videos (16 over 24 fps).
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = ["PAN2TransformerBlock", "PAN2TokenRefinerBlock"]
    _repeated_blocks = ["PAN2TransformerBlock", "PAN2TokenRefinerBlock"]
    # The text tokens are split before the token refiner, whose attention runs under context parallelism too, and the
    # video tokens once they are patchified. Attention runs over the whole joint sequence, so the order in which the
    # ranks hold the video and text shards does not matter. The number of text tokens varies with the prompt, so use
    # `ulysses_anything` or `ring_anything` when it is not divisible by the context parallel size.
    _cp_plan = {
        "": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "rope": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "transformer_blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 97,
        out_channels: int = 48,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 44,
        num_refiner_layers: int = 4,
        mlp_ratio: float = 8.0,
        context_mlp_ratio: float = 4.0,
        refiner_mlp_ratio: float = 4.0,
        text_embed_dim: int = 4096,
        rope_axes_dim: tuple[int, int, int] = (16, 56, 56),
        rope_theta: float = 256.0,
        rope_temporal_scale: float = 16 / 24,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.x_embedder = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.rope = PAN2RotaryPosEmbed(patch_size[1], patch_size[0], rope_axes_dim, rope_theta, rope_temporal_scale)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        self.context_embedder = nn.Sequential(RMSNorm(text_embed_dim, eps=1e-5), nn.Linear(text_embed_dim, inner_dim))
        self.context_refiner = nn.ModuleList(
            [
                PAN2TokenRefinerBlock(inner_dim, num_attention_heads, attention_head_dim, refiner_mlp_ratio, eps=1e-5)
                for _ in range(num_refiner_layers)
            ]
        )
        self.context_type_embedding = nn.Parameter(torch.zeros(inner_dim))

        self.transformer_blocks = nn.ModuleList(
            [
                PAN2TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, context_mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)

        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        r"""
        The [`PAN2Transformer3DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames, height, width)`):
                The noisy latents, the conditioning latents and the conditioning mask, concatenated along channels.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                The denoising timestep in `[0, 1000]`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_seq_len, text_embed_dim)`):
                The text encoder hidden states.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed along to the `AttentionProcessor`; only used for the LoRA scale.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The predicted velocity of shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. Rotary embedding, patch and timestep embeddings
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.x_embedder(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        timesteps_proj = self.time_proj(timestep).to(hidden_states.dtype)
        temb = self.time_embedder(timesteps_proj)

        # 2. Text embedding and refinement
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        for block in self.context_refiner:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states = self._gradient_checkpointing_func(block, encoder_hidden_states)
            else:
                encoder_hidden_states = block(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + self.context_type_embedding

        # 3. Joint-attention blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, rotary_emb
                )
            else:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary_emb)

        # 4. Output norm, projection and unpatchify
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
