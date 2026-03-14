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

import numpy as np
import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import is_torchvision_available
from ..attention import FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..embeddings import Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


if is_torchvision_available():
    from torchvision import transforms


class CosmosPatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: tuple[int, int, int], bias: bool = True
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], out_channels, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size, num_channels, num_frames // p_t, p_t, height // p_h, p_h, width // p_w, p_w
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class CosmosEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()

        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.embedding_dim = in_features

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb[..., : 2 * self.embedding_dim]

        shift, scale = embedded_timestep.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class CosmosAdaLayerNormZero(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()

        if hidden_features is None:
            self.linear_1 = nn.Identity()
        else:
            self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)

        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb

        shift, scale, gate = embedded_timestep.chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale, gate = (x.unsqueeze(1) for x in (shift, scale, gate))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states, gate


class CosmosAttnProcessor2_0:
    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("CosmosAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Apply RoPE
        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # 4. Prepare for GQA
        if torch.onnx.is_in_onnx_export():
            query_idx = torch.tensor(query.size(3), device=query.device)
            key_idx = torch.tensor(key.size(3), device=key.device)
            value_idx = torch.tensor(value.size(3), device=value.device)
        else:
            query_idx = query.size(3)
            key_idx = key.size(3)
            value_idx = value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        # 5. Attention
        hidden_states = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CosmosAttnProcessor2_5:
    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("CosmosAttnProcessor2_5 requires PyTorch 2.0. Please upgrade PyTorch to 2.0 or newer.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: tuple[torch.Tensor, torch.Tensor],
        attention_mask: tuple[torch.Tensor, torch.Tensor],
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, tuple):
            raise ValueError("Expected encoder_hidden_states as (text_context, img_context) tuple.")

        text_context, img_context = encoder_hidden_states if encoder_hidden_states else (None, None)
        text_mask, img_mask = attention_mask if attention_mask else (None, None)

        if text_context is None:
            text_context = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(text_context)
        value = attn.to_v(text_context)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        if torch.onnx.is_in_onnx_export():
            query_idx = torch.tensor(query.size(3), device=query.device)
            key_idx = torch.tensor(key.size(3), device=key.device)
            value_idx = torch.tensor(value.size(3), device=value.device)
        else:
            query_idx = query.size(3)
            key_idx = key.size(3)
            value_idx = value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        attn_out = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=text_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.flatten(2, 3).type_as(query)

        if img_context is not None:
            q_img = attn.q_img(hidden_states)
            k_img = attn.k_img(img_context)
            v_img = attn.v_img(img_context)

            batch_size = hidden_states.shape[0]
            dim_head = attn.out_dim // attn.heads

            q_img = q_img.view(batch_size, -1, attn.heads, dim_head).transpose(1, 2)
            k_img = k_img.view(batch_size, -1, attn.heads, dim_head).transpose(1, 2)
            v_img = v_img.view(batch_size, -1, attn.heads, dim_head).transpose(1, 2)

            q_img = attn.q_img_norm(q_img)
            k_img = attn.k_img_norm(k_img)

            q_img_idx = q_img.size(3)
            k_img_idx = k_img.size(3)
            v_img_idx = v_img.size(3)
            k_img = k_img.repeat_interleave(q_img_idx // k_img_idx, dim=3)
            v_img = v_img.repeat_interleave(q_img_idx // v_img_idx, dim=3)

            img_out = dispatch_attention_fn(
                q_img.transpose(1, 2),
                k_img.transpose(1, 2),
                v_img.transpose(1, 2),
                attn_mask=img_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            img_out = img_out.flatten(2, 3).type_as(q_img)
            hidden_states = attn_out + img_out
        else:
            hidden_states = attn_out

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CosmosAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add parameters for image q/k/v
        inner_dim = self.heads * self.to_q.out_features // self.heads
        self.q_img = nn.Linear(self.query_dim, inner_dim, bias=False)
        self.k_img = nn.Linear(self.query_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(self.query_dim, inner_dim, bias=False)
        self.q_img_norm = RMSNorm(self.to_q.out_features // self.heads, eps=1e-6, elementwise_affine=True)
        self.k_img_norm = RMSNorm(self.to_k.out_features // self.heads, eps=1e-6, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=hidden_states,
            # NOTE: type-hint in base class can be ignored
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


class CosmosTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        qk_norm: str = "rms_norm",
        out_bias: bool = False,
        img_context: bool = False,
        before_proj: bool = False,
        after_proj: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.img_context = img_context
        self.attn1 = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            elementwise_affine=True,
            out_bias=out_bias,
            processor=CosmosAttnProcessor2_0(),
        )

        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        if img_context:
            self.attn2 = CosmosAttention(
                query_dim=hidden_size,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                qk_norm=qk_norm,
                elementwise_affine=True,
                out_bias=out_bias,
                processor=CosmosAttnProcessor2_5(),
            )
        else:
            self.attn2 = Attention(
                query_dim=hidden_size,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                qk_norm=qk_norm,
                elementwise_affine=True,
                out_bias=out_bias,
                processor=CosmosAttnProcessor2_0(),
            )

        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu", bias=out_bias)

        # NOTE: zero conv for CosmosControlNet
        self.before_proj = None
        self.after_proj = None
        if before_proj:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
        if after_proj:
            self.after_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None | tuple[torch.Tensor | None, torch.Tensor | None],
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        controlnet_residual: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        block_idx: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.before_proj is not None:
            hidden_states = self.before_proj(hidden_states) + latents

        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. Self Attention
        norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)
        attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output

        # 2. Cross Attention
        norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep, temb)
        attn_output = self.attn2(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        hidden_states = hidden_states + gate * attn_output

        # 3. Feed Forward
        norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep, temb)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate * ff_output

        if controlnet_residual is not None:
            assert self.after_proj is None
            # NOTE: this is assumed to be scaled by the controlnet
            hidden_states += controlnet_residual

        if self.after_proj is not None:
            assert controlnet_residual is None
            hs_proj = self.after_proj(hidden_states)
            return hidden_states, hs_proj

        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.base_fps = base_fps

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(self, hidden_states: torch.Tensor, fps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]
        device = hidden_states.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)
        dim_h_range = (
            torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        )
        dim_w_range = (
            torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        )
        dim_t_range = (
            torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t
        )
        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

        # Apply sequence scaling in temporal dimension
        if fps is None:
            # Images
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:
            # Videos
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.eps = eps

        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2], hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]

        emb_t = self.pos_emb_t[: pe_size[0]][None, :, None, None, :].repeat(batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[: pe_size[1]][None, None, :, None, :].repeat(batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[: pe_size[2]][None, None, None, :, :].repeat(batch_size, pe_size[0], pe_size[1], 1, 1)
        emb = emb_t + emb_h + emb_w
        emb = emb.flatten(1, 3)

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


class CosmosTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A Transformer model for video-like data used in [Cosmos](https://github.com/NVIDIA/Cosmos).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head.
        num_layers (`int`, defaults to `28`):
            The number of layers of transformer blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        adaln_lora_dim (`int`, defaults to `256`):
            The hidden dimension of the Adaptive LayerNorm LoRA layer.
        max_size (`tuple[int, int, int]`, defaults to `(128, 240, 240)`):
            The maximum size of the input latent tensors in the temporal, height, and width dimensions.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The patch size to use for patchifying the input latent tensors in the temporal, height, and width
            dimensions.
        rope_scale (`tuple[float, float, float]`, defaults to `(2.0, 1.0, 1.0)`):
            The scaling factor to use for RoPE in the temporal, height, and width dimensions.
        concat_padding_mask (`bool`, defaults to `True`):
            Whether to concatenate the padding mask to the input latent tensors.
        extra_pos_embed_type (`str`, *optional*, defaults to `learnable`):
            The type of extra positional embeddings to use. Can be one of `None` or `learnable`.
        controlnet_block_every_n (`int`, *optional*):
            Interval between transformer blocks that should receive control residuals (for example, `7` to inject after
            every seventh block). Required for Cosmos Transfer2.5.
        img_context_dim_in (`int`, *optional*):
            The dimension of the input image context feature vector, i.e. it is the D in [B, N, D].
        img_context_num_tokens (`int`):
            The number of tokens in the image context feature vector, i.e. it is the N in [B, N, D]. If
            `img_context_dim_in` is not provided, then this parameter is ignored.
        img_context_dim_out (`int`):
            The output dimension of the image context projection layer. If `img_context_dim_in` is not provided, then
            this parameter is ignored.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embed", "final_layer", "norm"]
    _no_split_modules = ["CosmosTransformerBlock"]
    _keep_in_fp32_modules = ["learnable_pos_embed"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: str | None = "learnable",
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        encoder_hidden_states_channels: int = 1024,
        controlnet_block_every_n: int | None = None,
        img_context_dim_in: int | None = None,
        img_context_num_tokens: int = 256,
        img_context_dim_out: int = 2048,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)

        # 2. Positional Embedding
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim, max_size=max_size, patch_size=patch_size, rope_scale=rope_scale
        )

        self.learnable_pos_embed = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size,
                max_size=max_size,
                patch_size=patch_size,
            )

        # 3. Time Embedding
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)

        # 4. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                    img_context=self.config.img_context_dim_in is not None and self.config.img_context_dim_in > 0,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False
        )

        if self.config.use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, encoder_hidden_states_channels, bias=True),
                nn.GELU(),
            )

        self.gradient_checkpointing = False

        if self.config.img_context_dim_in:
            self.img_context_proj = nn.Sequential(
                nn.Linear(self.config.img_context_dim_in, self.config.img_context_dim_out, bias=True),
                nn.GELU(),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_controlnet_hidden_states: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            padding_mask_resized = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        # 3. Patchify input
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # 4. Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            # We can do this because num_frames == post_patch_num_frames, as p_t is 1
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )  # [BT, C] -> [B, T, 1, 1, C] -> [B, T, H, W, C] -> [B, THW, C]
        else:
            raise ValueError(f"Expected timestep to have shape [B, 1, T, 1, 1] or [T], but got {timestep.shape}")

        # 5. Process encoder hidden states
        text_context, img_context = (
            encoder_hidden_states if isinstance(encoder_hidden_states, tuple) else (encoder_hidden_states, None)
        )
        if self.config.use_crossattn_projection:
            text_context = self.crossattn_proj(text_context)

        if img_context is not None and self.config.img_context_dim_in:
            img_context = self.img_context_proj(img_context)

        processed_encoder_hidden_states = (
            (text_context, img_context) if isinstance(encoder_hidden_states, tuple) else text_context
        )

        # 6. Build controlnet block index map
        controlnet_block_index_map = {}
        if block_controlnet_hidden_states is not None:
            n_blocks = len(self.transformer_blocks)
            controlnet_block_index_map = {
                block_idx: block_controlnet_hidden_states[idx]
                for idx, block_idx in list(enumerate(range(0, n_blocks, self.config.controlnet_block_every_n)))
            }

        # 7. Transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            controlnet_residual = controlnet_block_index_map.get(block_idx)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    processed_encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                    controlnet_residual,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    processed_encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                    controlnet_residual,
                )

        # 8. Output norm & projection & unpatchify
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        # NOTE: The permutation order here is not the inverse operation of what happens when patching as usually expected.
        # It might be a source of confusion to the reader, but this is correct
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
