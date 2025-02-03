# Copyright 2024 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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
from einops import rearrange, repeat
from torchvision import transforms

from ..attention import FeedForward
from ..attention_processor import Attention
from ..embeddings import Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..normalization import RMSNorm


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(hidden_states)
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
        embedded_timestep = self.t_embedder(timesteps_proj)
        norm_timesteps_proj = self.norm(timesteps_proj)
        return norm_timesteps_proj, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):
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
        self, hidden_states: torch.Tensor, temb: torch.Tensor, embedded_timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        temb = self.activation(temb)
        temb = self.linear_1(temb)
        temb = self.linear_2(temb)

        if embedded_timestep is not None:
            temb = temb + embedded_timestep

        shift, scale, gate = temb.chunk(3, dim=1)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return hidden_states, gate


class FinalLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        patch_size: Tuple[int, int, int],
        out_channels: int,
        modulation_dim: int = 256,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            embedding_dim, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False
        )
        self.hidden_size = embedding_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, modulation_dim, bias=False),
            nn.Linear(modulation_dim, 2 * embedding_dim, bias=False),
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, embedded_timestep: torch.Tensor
    ) -> torch.Tensor:
        temb = self.adaLN_modulation(temb) + embedded_timestep[:, : 2 * self.hidden_size]
        shift, scale = temb.chunk(2, dim=1)

        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


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

            query = apply_rotary_emb(query, image_rotary_emb, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real_unbind_dim=-2)

        # 4. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, enable_gqa=True
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        len_h: int,
        len_w: int,
        len_t: int,
        patch_size: Tuple[int, int, int],
        base_fps: int = 24,
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
    ) -> None:
        super().__init__()

        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.patch_size = patch_size

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(self, hidden_states: torch.Tensor, fps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_h, self.max_w, self.max_t), dtype=torch.float32)
        dim_h_range = torch.arange(0, self.dim_h, 2, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t
        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[: rope_sizes[1]], h_spatial_freqs)
        emb_w = torch.outer(seq[: rope_sizes[2]], w_spatial_freqs)

        # Apply sequence scaling in temporal dimension
        if fps is None:
            # Images
            emb_t = torch.outer(seq[: rope_sizes[0]], temporal_freqs)
        else:
            # Videos
            emb_t = torch.outer(seq[: rope_sizes[0]] / fps * self.base_fps, temporal_freqs)

        freqs = torch.cat(
            [
                repeat(emb_t, "t d -> t h w d", h=rope_sizes[1], w=rope_sizes[2]),
                repeat(emb_h, "h d -> t h w d", t=rope_sizes[0], w=rope_sizes[2]),
                repeat(emb_w, "w d -> t h w d", t=rope_sizes[0], h=rope_sizes[1]),
            ]
            * 2,
            dim=-1,
        )

        freqs = rearrange(freqs, "t h w d -> (t h w) d").float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        len_h: int,
        len_w: int,
        len_t: int,
        patch_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, hidden_size))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_sizes = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]

        emb_t_T = self.pos_emb_t[: pe_sizes[0]]
        emb_h_H = self.pos_emb_h[: pe_sizes[1]]
        emb_w_W = self.pos_emb_w[: pe_sizes[2]]
        emb = (
            repeat(emb_t_T, "t d -> b t h w d", b=batch_size, h=pe_sizes[1], w=pe_sizes[2])
            + repeat(emb_h_H, "h d -> b t h w d", b=batch_size, t=pe_sizes[0], w=pe_sizes[2])
            + repeat(emb_w_W, "w d -> b t h w d", b=batch_size, t=pe_sizes[0], h=pe_sizes[1])
        )
        emb = emb.flatten(1, 3)

        eps = 1e-6
        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


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

        self.norm1 = CosmosAdaLayerNorm(in_features=hidden_size, hidden_features=adaln_lora_dim)
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

        self.norm2 = CosmosAdaLayerNorm(in_features=hidden_size, hidden_features=adaln_lora_dim)
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

        self.norm3 = CosmosAdaLayerNorm(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu", bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        embedded_timestep: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. Self Attention
        norm_hidden_states, gate = self.norm1(hidden_states, temb, embedded_timestep)
        attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate.unsqueeze(1) * attn_output

        # 2. Cross Attention
        norm_hidden_states, gate = self.norm2(hidden_states, temb, embedded_timestep)
        attn_output = self.attn2(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        hidden_states = hidden_states + gate.unsqueeze(1) * attn_output

        # 3. Feed Forward
        norm_hidden_states, gate = self.norm3(hidden_states, temb, embedded_timestep)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate.unsqueeze(1) * ff_output

        return hidden_states


class GeneralDIT(nn.Module):
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_size: Tuple[int, int, int],
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 4096,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        # positional embedding settings
        pos_emb_learnable: bool = False,
        adaln_lora_dim: int = 256,
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        extra_per_block_abs_pos_emb_type: Optional[str] = "learnable",
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.patch_size = patch_size
        self.concat_padding_mask = concat_padding_mask
        # positional embedding settings
        self.pos_emb_learnable = pos_emb_learnable
        self.extra_per_block_abs_pos_emb_type = extra_per_block_abs_pos_emb_type.lower()
        self.adaln_lora_dim = adaln_lora_dim

        # 1. Patch Embedding
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, model_channels, patch_size, bias=False)

        # 2. Positional Embedding
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=model_channels // num_heads,
            len_h=max_img_h // patch_size[1],
            len_w=max_img_w // patch_size[2],
            len_t=max_frames // patch_size[0],
            patch_size=patch_size,
            rope_scale=rope_scale,
        )

        self.learnable_pos_embedder = None
        if extra_per_block_abs_pos_emb_type == "learnable":
            self.learnable_pos_embedder = CosmosLearnablePositionalEmbed(
                hidden_size=model_channels,
                len_h=max_img_h // patch_size[1],
                len_w=max_img_w // patch_size[2],
                len_t=max_frames // patch_size[0],
                patch_size=patch_size,
            )

        # 3. Time Embedding
        self.time_embed = CosmosEmbedding(model_channels, model_channels)

        # 4. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_heads,
                    attention_head_dim=model_channels // num_heads,
                    cross_attention_dim=crossattn_emb_channels,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                )
                for _ in range(num_blocks)
            ]
        )

        # 5. Output norm & projection
        self.final_layer = FinalLayer(
            embedding_dim=model_channels,
            patch_size=patch_size,
            out_channels=out_channels,
            modulation_dim=adaln_lora_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        # 1. Concatenate padding mask if needed
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(1).repeat(1, 1, hidden_states.shape[2], 1, 1)], dim=1
            )

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embedder(hidden_states) if self.extra_per_block_abs_pos_emb_type else None

        # 3. Patchify input
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // self.patch_size[0]
        post_patch_height = height // self.patch_size[1]
        post_patch_width = width // self.patch_size[2]
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] => [B, THW, C]

        # 4. Timestep embeddings
        temb, embedded_timestep = self.time_embed(hidden_states, timestep)

        # 5. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                embedded_timestep=embedded_timestep,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=extra_pos_emb,
            )

        # 6. Output norm & projection
        hidden_states = self.final_layer(hidden_states, temb, embedded_timestep)
        hidden_states = hidden_states.unflatten(2, (-1, *self.patch_size))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
