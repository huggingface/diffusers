# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.loaders import FromOriginalModelMixin

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention, AttentionProcessor
from ..cache_utils import CacheMixin
from ..embeddings import (
    CombinedTimestepTextProjEmbeddings,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HunyuanImageAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanImageAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))  # batch_size, seq_len, heads, head_dim
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(
                            query[:, : -encoder_hidden_states.shape[1]], image_rotary_emb, sequence_dim=1
                        ),
                        query[:, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=1,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, : -encoder_hidden_states.shape[1]], image_rotary_emb, sequence_dim=1),
                        key[:, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=1,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        # 5. Attention
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

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanImagePatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]] = (16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size

        if len(patch_size) == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif len(patch_size) == 3:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise ValueError(f"patch_size must be a tuple of length 2 or 3, got {len(patch_size)}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class HunyuanImageByT5TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_features)
        self.act_fn = nn.GELU()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


class HunyuanImageAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanImageCombinedTimeGuidanceEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        guidance_embeds: bool = False,
        use_meanflow: bool = False,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_meanflow = use_meanflow

        self.time_proj_r = None
        self.timestep_embedder_r = None
        if use_meanflow:
            self.time_proj_r = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder_r = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.guidance_embedder = None
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_r: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))

        if timestep_r is not None:
            timesteps_proj_r = self.time_proj_r(timestep_r)
            timesteps_emb_r = self.timestep_embedder_r(timesteps_proj_r.to(dtype=timestep.dtype))
            timesteps_emb = (timesteps_emb + timesteps_emb_r) / 2

        if self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=timestep.dtype))
            conditioning = timesteps_emb + guidance_emb
        else:
            conditioning = timesteps_emb

        return conditioning


# IndividualTokenRefinerBlock
@maybe_allow_in_graph
class HunyuanImageIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,  # 28
        attention_head_dim: int,  # 128
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate)

        self.norm_out = HunyuanImageAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanImageIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanImageIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device)
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


# txt_in
class HunyuanImageTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanImageIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_hidden_states = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_hidden_states = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_hidden_states = pooled_hidden_states.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanImageRotaryPosEmbed(nn.Module):
    def __init__(
        self, patch_size: Union[Tuple, List[int]], rope_dim: Union[Tuple, List[int]], theta: float = 256.0
    ) -> None:
        super().__init__()

        if not isinstance(patch_size, (tuple, list)) or len(patch_size) not in [2, 3]:
            raise ValueError(f"patch_size must be a tuple or list of length 2 or 3, got {patch_size}")

        if not isinstance(rope_dim, (tuple, list)) or len(rope_dim) not in [2, 3]:
            raise ValueError(f"rope_dim must be a tuple or list of length 2 or 3, got {rope_dim}")

        if not len(patch_size) == len(rope_dim):
            raise ValueError(f"patch_size and rope_dim must have the same length, got {patch_size} and {rope_dim}")

        self.patch_size = patch_size
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 5:
            _, _, frame, height, width = hidden_states.shape
            patch_size_frame, patch_size_height, patch_size_width = self.patch_size
            rope_sizes = [frame // patch_size_frame, height // patch_size_height, width // patch_size_width]
        elif hidden_states.ndim == 4:
            _, _, height, width = hidden_states.shape
            patch_size_height, patch_size_width = self.patch_size
            rope_sizes = [height // patch_size_height, width // patch_size_width]
        else:
            raise ValueError(f"hidden_states must be a 4D or 5D tensor, got {hidden_states.shape}")

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # dim x [H, W]
        grid = torch.stack(grid, dim=0)  # [2, H, W]

        freqs = []
        for i in range(len(rope_sizes)):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class HunyuanImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanImageAttnProcessor(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class HunyuanImageTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanImageAttnProcessor(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    The Transformer model used in [HunyuanImage-2.1](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of dual-stream blocks to use.
        num_single_layers (`int`, defaults to `40`):
            The number of layers of single-stream blocks to use.
        num_refiner_layers (`int`, defaults to `2`):
            The number of layers of refiner blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings in the model.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        pooled_projection_dim (`int`, defaults to `768`):
            The dimension of the pooled projection of the text embeddings.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
        image_condition_type (`str`, *optional*, defaults to `None`):
            The type of image conditioning to use. If `None`, no image conditioning is used. If `latent_concat`, the
            image is concatenated to the latent stream. If `token_replace`, the image is used to replace first-frame
            tokens in the latent stream and apply conditioning.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanImageTransformerBlock",
        "HunyuanImageSingleTransformerBlock",
        "HunyuanImagePatchEmbed",
        "HunyuanImageTokenRefiner",
    ]
    _repeated_blocks = [
        "HunyuanImageTransformerBlock",
        "HunyuanImageSingleTransformerBlock",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        num_attention_heads: int = 28,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: Tuple[int, int] = (1, 1),
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = False,
        text_embed_dim: int = 3584,
        text_embed_2_dim: Optional[int] = None,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (64, 64),
        use_meanflow: bool = False,
    ) -> None:
        super().__init__()

        if not (isinstance(patch_size, (tuple, list)) and len(patch_size) in [2, 3]):
            raise ValueError(f"patch_size must be a tuple of length 2 or 3, got {patch_size}")

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanImagePatchEmbed(patch_size, in_channels, inner_dim)
        self.context_embedder = HunyuanImageTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )

        if text_embed_2_dim is not None:
            self.context_embedder_2 = HunyuanImageByT5TextProjection(text_embed_2_dim, 2048, inner_dim)
        else:
            self.context_embedder_2 = None

        self.time_guidance_embed = HunyuanImageCombinedTimeGuidanceEmbedding(inner_dim, guidance_embeds, use_meanflow)

        # 2. RoPE
        self.rope = HunyuanImageRotaryPosEmbed(patch_size, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks

        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanImageTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanImageSingleTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, math.prod(patch_size) * out_channels)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: Optional[torch.LongTensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if hidden_states.ndim == 4:
            batch_size, channels, height, width = hidden_states.shape
            sizes = (height, width)
        elif hidden_states.ndim == 5:
            batch_size, channels, frame, height, width = hidden_states.shape
            sizes = (frame, height, width)
        else:
            raise ValueError(f"hidden_states must be a 4D or 5D tensor, got {hidden_states.shape}")

        post_patch_sizes = tuple(d // p for d, p in zip(sizes, self.config.patch_size))

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        encoder_attention_mask = encoder_attention_mask.bool()
        temb = self.time_guidance_embed(timestep, guidance=guidance, timestep_r=timestep_r)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        if self.context_embedder_2 is not None and encoder_hidden_states_2 is not None:
            encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

            encoder_attention_mask_2 = encoder_attention_mask_2.bool()

            # reorder and combine text tokens: combine valid tokens first, then padding
            new_encoder_hidden_states = []
            new_encoder_attention_mask = []

            for text, text_mask, text_2, text_mask_2 in zip(
                encoder_hidden_states, encoder_attention_mask, encoder_hidden_states_2, encoder_attention_mask_2
            ):
                # Concatenate: [valid_mllm, valid_byt5, invalid_mllm, invalid_byt5]
                new_encoder_hidden_states.append(
                    torch.cat(
                        [
                            text_2[text_mask_2],  # valid byt5
                            text[text_mask],  # valid mllm
                            text_2[~text_mask_2],  # invalid byt5
                            text[~text_mask],  # invalid mllm
                        ],
                        dim=0,
                    )
                )

                # Apply same reordering to attention masks
                new_encoder_attention_mask.append(
                    torch.cat(
                        [
                            text_mask_2[text_mask_2],
                            text_mask[text_mask],
                            text_mask_2[~text_mask_2],
                            text_mask[~text_mask],
                        ],
                        dim=0,
                    )
                )

            encoder_hidden_states = torch.stack(new_encoder_hidden_states)
            encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        attention_mask = torch.nn.functional.pad(encoder_attention_mask, (hidden_states.shape[1], 0), value=True)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # 3. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

        # 4. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 5. unpatchify
        # reshape: [batch_size, *post_patch_dims, channels, *patch_size]
        out_channels = self.config.out_channels
        reshape_dims = [batch_size] + list(post_patch_sizes) + [out_channels] + list(self.config.patch_size)
        hidden_states = hidden_states.reshape(*reshape_dims)

        # create permutation pattern: batch, channels, then interleave post_patch and patch dims
        # For 4D: [0, 3, 1, 4, 2, 5] -> batch, channels, post_patch_height, patch_size_height, post_patch_width, patch_size_width
        # For 5D: [0, 4, 1, 5, 2, 6, 3, 7] -> batch, channels, post_patch_frame, patch_size_frame, post_patch_height, patch_size_height, post_patch_width, patch_size_width
        ndim = len(post_patch_sizes)
        permute_pattern = [0, ndim + 1]  # batch, channels
        for i in range(ndim):
            permute_pattern.extend([i + 1, ndim + 2 + i])  # post_patch_sizes[i], patch_sizes[i]
        hidden_states = hidden_states.permute(*permute_pattern)

        # flatten patch dimensions: flatten each (post_patch_size, patch_size) pair
        # batch_size, channels, post_patch_sizes[0] * patch_sizes[0], post_patch_sizes[1] * patch_sizes[1], ...
        final_dims = [batch_size, out_channels] + [
            post_patch * patch for post_patch, patch in zip(post_patch_sizes, self.config.patch_size)
        ]
        hidden_states = hidden_states.reshape(*final_dims)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
