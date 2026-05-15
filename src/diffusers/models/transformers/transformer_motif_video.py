# Copyright 2026 Motif Technologies and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MotifVideoCrossAttnProcessor2_0:
    """Attention processor for Motif-Video text cross-attention."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "MotifVideoCrossAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: "MotifVideoCrossAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_embed_seq_len: int = 0,
    ) -> torch.Tensor:
        txt_kv = encoder_hidden_states[:, image_embed_seq_len:, :]

        text_mask = None
        if attention_mask is not None:
            text_mask = attention_mask[:, :, :, image_embed_seq_len - encoder_hidden_states.shape[1] :]

        query = attn.to_q(hidden_states)
        key = attn.to_k(txt_kv)
        value = attn.to_v(txt_kv)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=text_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MotifVideoAttnProcessor2_0:
    """Attention processor for Motif-Video self-attention."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "MotifVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: "MotifVideoAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Concatenate hidden states with encoder hidden states for joint attention if needed
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # Project QKV
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Normalize QK
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                split_idx = -encoder_hidden_states.shape[1]
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :split_idx, :, :], image_rotary_emb, sequence_dim=1),
                        query[:, split_idx:, :, :],
                    ],
                    dim=1,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :split_idx, :, :], image_rotary_emb, sequence_dim=1),
                        key[:, split_idx:, :, :],
                    ],
                    dim=1,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Add encoder conditioning QKV projections and normalization
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

        # Compute attention with backend dispatch
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Apply output projections and split encoder states
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if attn.to_out is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if attn.to_add_out is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states

        if attn.to_out is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MotifVideoCrossAttention(nn.Module, AttentionModuleMixin):
    """Dedicated cross-attention module for Motif-Video text cross-attention."""

    _default_processor_cls = MotifVideoCrossAttnProcessor2_0
    _available_processors = [MotifVideoCrossAttnProcessor2_0]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        qk_norm: str = "rms_norm",
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.heads = heads

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, query_dim, bias=out_bias),
                nn.Dropout(dropout),
            ]
        )

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_embed_seq_len: int = 0,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            image_embed_seq_len,
        )


class MotifVideoAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = MotifVideoAttnProcessor2_0
    _available_processors = [MotifVideoAttnProcessor2_0]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        pre_only: bool = False,
        context_pre_only: bool = False,
        qk_norm: str = "rms_norm",
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.pre_only = pre_only

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.context_pre_only = context_pre_only

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        # QK Norm
        if qk_norm == "rms_norm":
            self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "layer_norm":
            self.norm_q = torch.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = torch.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        else:
            self.norm_q = None
            self.norm_k = None

        if not pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(torch.nn.Dropout(dropout))
        else:
            self.to_out = None

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            if not context_pre_only:
                self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)
            else:
                self.to_add_out = None
        else:
            self.norm_added_q = None
            self.norm_added_k = None
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class MotifVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class MotifVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class MotifVideoConditionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        param_dtype = get_parameter_dtype(self.timestep_embedder)
        # Timesteps always returns FP32 output, so cast to the weight dtype of timestep_embedder if we're operating in
        # FP16 or BF16 (and no quantization)
        if param_dtype in (torch.float16, torch.bfloat16):
            timesteps_proj = timesteps_proj.to(param_dtype)
        conditioning = self.timestep_embedder(timesteps_proj)  # (N, D)

        return conditioning


class MotifVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: List[int],
        theta: float = 256.0,
    ):
        """
        Rotary Positional Embedding (RoPE) for video latents.

        Args:
            patch_size (`int`): Spatial patch size.
            patch_size_t (`int`): Temporal patch size.
            rope_dim (`List[int]`): Dimensions for RoPE across [Time, Height, Width] axes.
            theta (`float`, *optional*, defaults to 256.0): Base frequency for rotary embeddings.
        """
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = []
        for i in range(3):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)

        freqs = []
        is_mps = hidden_states.device.type == "mps"
        is_npu = hidden_states.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(3):
            freq = get_1d_rotary_pos_embed(
                dim=self.rope_dim[i],
                pos=grid[i].reshape(-1),
                theta=self.theta,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
        return freqs_cos, freqs_sin


class MotifVideoImageProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, in_features)
        self.act_fn = nn.GELU()
        self.linear_2 = nn.Linear(in_features, hidden_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class MotifVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
        norm_type: str = "layer_norm",
        enable_text_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = MotifVideoAttention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=hidden_size,
            bias=True,
            pre_only=True,
            qk_norm=qk_norm,
            eps=1e-6,
            processor=MotifVideoAttnProcessor2_0(),
        )

        self.cross_attn = (
            MotifVideoCrossAttention(
                query_dim=hidden_size,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                bias=True,
                qk_norm=qk_norm,
                eps=1e-6,
            )
            if enable_text_cross_attention
            else None
        )

        self.enable_text_cross_attention = enable_text_cross_attention

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type=norm_type)
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
        image_embed_seq_len: int = 0,
    ) -> torch.Tensor:
        encoder_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-encoder_seq_length, :],
            norm_hidden_states[:, -encoder_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # 3. Text cross-attention
        if self.cross_attn is not None:
            cross_output = self.cross_attn(
                hidden_states=attn_output,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                image_embed_seq_len=image_embed_seq_len,
            )
            attn_output = attn_output + cross_output

        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 4. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-encoder_seq_length, :],
            hidden_states[:, -encoder_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class MotifVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
        norm_type: str = "layer_norm",
        enable_text_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type=norm_type)
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type=norm_type)

        self.attn = MotifVideoAttention(
            query_dim=hidden_size,
            added_kv_proj_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=hidden_size,
            bias=True,
            context_pre_only=False,
            qk_norm=qk_norm,
            eps=1e-6,
            processor=MotifVideoAttnProcessor2_0(),
        )

        self.cross_attn = (
            MotifVideoCrossAttention(
                query_dim=hidden_size,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                bias=True,
                qk_norm=qk_norm,
                eps=1e-6,
            )
            if enable_text_cross_attention
            else None
        )

        self.enable_text_cross_attention = enable_text_cross_attention

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_embed_seq_len: int = 0,
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

        # 4. Text cross-attention
        if self.cross_attn is not None:
            cross_output = self.cross_attn(
                hidden_states=attn_output,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                image_embed_seq_len=image_embed_seq_len,
            )
            hidden_states = hidden_states + cross_output

        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 5. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class MotifVideoTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Motif-Video model.

    Args:
        in_channels (`int`, defaults to `33`):
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
        num_decoder_layers (`int`, defaults to `0`):
            The number of decoder layers in single-stream blocks.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the temporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        image_embed_dim (`int`, *optional*):
            Input dimension of image embeddings from a vision encoder. If provided, enables image conditioning.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _repeated_blocks = ["MotifVideoSingleTransformerBlock", "MotifVideoTransformerBlock"]
    _no_split_modules = [
        "MotifVideoTransformerBlock",
        "MotifVideoSingleTransformerBlock",
        "MotifVideoPatchEmbed",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 33,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_decoder_layers: int = 0,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        norm_type: str = "layer_norm",
        text_embed_dim: int = 4096,
        image_embed_dim: int | None = None,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
        enable_text_cross_attention_dual: bool = False,
        enable_text_cross_attention_single: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = MotifVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = PixArtAlphaTextProjection(in_features=text_embed_dim, hidden_size=inner_dim)

        # First frame conditioning: Image conditioning embedders
        self.image_embed_dim = image_embed_dim
        if image_embed_dim is not None:
            self.image_embedder = MotifVideoImageProjection(in_features=image_embed_dim, hidden_size=inner_dim)

        self.time_text_embed = MotifVideoConditionEmbedding(inner_dim)

        # 2. RoPE
        self.rope = MotifVideoRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # Cross-attention config
        self.enable_text_cross_attention_dual = enable_text_cross_attention_dual
        self.enable_text_cross_attention_single = enable_text_cross_attention_single

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MotifVideoTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                    enable_text_cross_attention=enable_text_cross_attention_dual,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        # Encoder blocks get cross-attention; decoder blocks do not (no text stream in decoder)
        num_encoder_single = num_single_layers - num_decoder_layers
        self.single_transformer_blocks = nn.ModuleList(
            [
                MotifVideoSingleTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                    enable_text_cross_attention=enable_text_cross_attention_single
                    if i < num_encoder_single
                    else False,
                )
                for i in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(
            inner_dim,
            inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            norm_type=norm_type,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        # Verify cross-attention config matches actual block state.
        # Catches silent misconfiguration (e.g. checkpoint config with renamed keys).
        for i, block in enumerate(self.transformer_blocks):
            if block.enable_text_cross_attention != enable_text_cross_attention_dual:
                raise ValueError(
                    f"transformer_blocks[{i}].enable_text_cross_attention="
                    f"{block.enable_text_cross_attention}, expected {enable_text_cross_attention_dual}. "
                    f"Check checkpoint config.json key names match __init__ parameters."
                )
        for i, block in enumerate(self.single_transformer_blocks):
            expected = enable_text_cross_attention_single if i < num_encoder_single else False
            if block.enable_text_cross_attention != expected:
                raise ValueError(
                    f"single_transformer_blocks[{i}].enable_text_cross_attention="
                    f"{block.enable_text_cross_attention}, expected {expected}. "
                    f"Check checkpoint config.json key names match __init__ parameters."
                )

        self.gradient_checkpointing = False
        self.num_decoder_layers = num_decoder_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the MotifVideoTransformer3DModel.

        Args:
            hidden_states (`torch.Tensor`):
                Input latent tensor of shape `(batch_size, channels, num_frames, height, width)`.
            timestep (`torch.LongTensor`):
                Diffusion timesteps of shape `(batch_size,)`.
            encoder_hidden_states (`torch.Tensor`):
                Text conditioning of shape `(batch_size, sequence_length, embed_dim)`.
            encoder_attention_mask (`torch.Tensor`):
                Mask for text conditioning of shape `(batch_size, sequence_length)`.
            image_embeds (`torch.Tensor`, *optional*):
                Image embeddings from vision encoder of shape `(batch_size, num_tokens, embed_dim)`.
            attention_kwargs (`dict`, *optional*):
                Additional arguments for attention processors.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`].

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The predicted samples.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, _, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # First frame conditioning: Image embeddings from vision encoder
        if image_embeds is not None:
            image_embeds = self.image_embedder(image_embeds)
            encoder_hidden_states = torch.cat([image_embeds, encoder_hidden_states], dim=1)
            if encoder_attention_mask is not None:
                image_mask = torch.ones(
                    image_embeds.shape[0],
                    image_embeds.shape[1],
                    device=encoder_attention_mask.device,
                    dtype=encoder_attention_mask.dtype,
                )
                encoder_attention_mask = torch.cat([image_mask, encoder_attention_mask], dim=1)

        # image_embed_seq_len: used by cross-attention blocks to slice text from encoder_hidden_states
        image_embed_seq_len = image_embeds.shape[1] if image_embeds is not None else 0

        if self.num_decoder_layers > 0:
            decoder_hidden_states = hidden_states.clone()

        if encoder_attention_mask is not None:
            attention_mask = F.pad(
                encoder_attention_mask.to(torch.bool),
                (hidden_states.shape[1], 0),
                value=True,
            )
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        else:
            attention_mask = None

        # 3. Dual stream transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = (
                self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    image_embed_seq_len,
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb, image_embed_seq_len
                )
            )

        # 4. Single stream transformer blocks (Encoder)
        single_transformer_blocks = self.single_transformer_blocks

        for block in single_transformer_blocks[: len(single_transformer_blocks) - self.num_decoder_layers]:
            hidden_states, encoder_hidden_states = (
                self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    image_embed_seq_len,
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb, image_embed_seq_len
                )
            )

        # 5. Single stream transformer blocks (Decoder)
        if self.num_decoder_layers > 0:
            encoder_hidden_states = hidden_states
            attention_mask = None

            for block in single_transformer_blocks[-self.num_decoder_layers :]:
                decoder_hidden_states, encoder_hidden_states = (
                    self._gradient_checkpointing_func(
                        block, decoder_hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                    )
                    if torch.is_grad_enabled() and self.gradient_checkpointing
                    else block(decoder_hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb)
                )

            hidden_states = decoder_hidden_states

        # 6. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(
            sample=hidden_states,
        )
