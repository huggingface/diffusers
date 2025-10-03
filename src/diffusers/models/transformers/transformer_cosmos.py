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

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import is_torchvision_available
from ..attention import FeedForward
from ..attention_processor import Attention
from ..embeddings import Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


if is_torchvision_available():
    from torchvision import transforms


class CosmosPatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: Tuple[int, int, int], bias: bool = True
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
        self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor, temb: Optional[torch.Tensor] = None
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
    def __init__(self, in_features: int, hidden_features: Optional[int] = None) -> None:
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
        temb: Optional[torch.Tensor] = None,
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
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CosmosAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
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
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        # 6. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


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
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
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

    def forward(self, hidden_states: torch.Tensor, fps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        max_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
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
        max_size (`Tuple[int, int, int]`, defaults to `(128, 240, 240)`):
            The maximum size of the input latent tensors in the temporal, height, and width dimensions.
        patch_size (`Tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The patch size to use for patchifying the input latent tensors in the temporal, height, and width
            dimensions.
        rope_scale (`Tuple[float, float, float]`, defaults to `(2.0, 1.0, 1.0)`):
            The scaling factor to use for RoPE in the temporal, height, and width dimensions.
        concat_padding_mask (`bool`, defaults to `True`):
            Whether to concatenate the padding mask to the input latent tensors.
        extra_pos_embed_type (`str`, *optional*, defaults to `learnable`):
            The type of extra positional embeddings to use. Can be one of `None` or `learnable`.
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
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: Optional[str] = "learnable",
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
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
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
            assert False

        # 5. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                )

        # 6. Output norm & projection & unpatchify
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
