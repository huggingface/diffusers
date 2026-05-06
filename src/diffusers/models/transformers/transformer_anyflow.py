# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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
#
# This file derives from the FAR architecture (Gu et al., 2025, arXiv:2503.19325) and adds AnyFlow's
# dual-timestep flow-map embedding (AnyFlowDualTimestepTextImageEmbedding). The base 3D DiT structure
# is adapted from the v0.35.1 Wan2.1 transformer (transformer_wan.py); upstream Wan has since been
# refactored, so this file is intentionally self-contained rather than annotated with `# Copied from`.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import BaseOutput, logging
from ..attention import FeedForward
from ..attention_processor import Attention
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnyFlowFARTransformerOutput(BaseOutput):
    """
    Output dataclass for ``AnyFlowTransformer3DModel``'s causal forward paths.

    Args:
        sample (`torch.Tensor` or `None`):
            Predicted denoising target for the autoregressive chunk. ``None`` for the cache-prefill path,
            which only writes the KV cache and produces no usable sample.
        kv_cache (`list[dict[str, torch.Tensor]]`, *optional*):
            Per-block KV cache state used by subsequent autoregressive steps.
    """

    sample: Optional[torch.Tensor] = None
    kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None


# `flex_attention` is JIT-compiled lazily on first call so that importing this module does not require
# Triton or a CUDA-capable device (CPU CI / older PyTorch builds otherwise fail at import time).
try:
    flex_attention = torch.compile(flex_attention, dynamic=True)
except Exception as e:  # pragma: no cover - environment-dependent
    logger.warning(
        "Failed to torch.compile flex_attention; falling back to the eager kernel. Error: %s",
        e,
    )


def build_block_mask(mask_2d, device):
    assert mask_2d.dim() == 2 and mask_2d.dtype == torch.bool
    mask_2d = mask_2d.contiguous()

    Q_LEN, KV_LEN = mask_2d.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return mask_2d[q_idx, kv_idx]

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device, _compile=False)


def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
    x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(hidden_states)


class AnyFlowSelfAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AnyFlowSelfAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        kv_cache=None,
        kv_cache_flag=None,
    ) -> torch.Tensor:

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if kv_cache is not None:
            if kv_cache_flag["is_cache_step"]:
                kv_cache["compressed_cache"][0, :, :, : kv_cache_flag["num_compressed_tokens"], :] = key[
                    :, :, : kv_cache_flag["num_compressed_tokens"]
                ]
                kv_cache["compressed_cache"][1, :, :, : kv_cache_flag["num_compressed_tokens"], :] = value[
                    :, :, : kv_cache_flag["num_compressed_tokens"]
                ]
                kv_cache["full_cache"][0, :, :, : kv_cache_flag["num_full_tokens"], :] = key[
                    :, :, kv_cache_flag["num_compressed_tokens"] :
                ]
                kv_cache["full_cache"][1, :, :, : kv_cache_flag["num_full_tokens"], :] = value[
                    :, :, kv_cache_flag["num_compressed_tokens"] :
                ]
            else:
                key = torch.cat(
                    [
                        kv_cache["compressed_cache"][0, :, :, : kv_cache_flag["num_cached_compressed_tokens"], :],
                        kv_cache["full_cache"][0, :, :, : kv_cache_flag["num_cached_full_tokens"], :],
                        key,
                    ],
                    dim=2,
                )
                value = torch.cat(
                    [
                        kv_cache["compressed_cache"][1, :, :, : kv_cache_flag["num_cached_compressed_tokens"], :],
                        kv_cache["full_cache"][1, :, :, : kv_cache_flag["num_cached_full_tokens"], :],
                        value,
                    ],
                    dim=2,
                )

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb["query"])
            key = apply_rotary_emb(key, rotary_emb["key"])

        if attention_mask is None:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            seq_len = query.shape[2]
            padded_length = int(math.ceil(seq_len / 128.0) * 128.0 - seq_len)
            query = torch.cat(
                [
                    query,
                    torch.zeros(
                        [query.shape[0], query.shape[1], padded_length, query.shape[3]],
                        device=query.device,
                        dtype=query.dtype,
                    ),
                ],
                dim=2,
            )  # noqa: E501
            key = torch.cat(
                [
                    key,
                    torch.zeros(
                        [key.shape[0], key.shape[1], padded_length, key.shape[3]], device=key.device, dtype=key.dtype
                    ),
                ],
                dim=2,
            )  # noqa: E501
            value = torch.cat(
                [
                    value,
                    torch.zeros(
                        [value.shape[0], value.shape[1], padded_length, value.shape[3]],
                        device=value.device,
                        dtype=value.dtype,
                    ),
                ],
                dim=2,
            )  # noqa: E501

            hidden_states = flex_attention(query, key, value, block_mask=attention_mask)[:, :, :seq_len]

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AnyFlowCrossAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AnyFlowCrossAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb["query"])
            key = apply_rotary_emb(key, rotary_emb["key"])

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AnyFlowImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class AnyFlowDualTimestepTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        gate_value: float,
        deltatime_type: str,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.delta_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = AnyFlowImageEmbedding(image_embed_dim, dim)

        self.register_buffer("delta_emb_gate", torch.tensor([gate_value], dtype=torch.float32), persistent=False)
        self.deltatime_type = deltatime_type

    def forward_timestep(
        self, timestep: torch.Tensor, delta_timestep: torch.Tensor, encoder_hidden_states, token_per_frame
    ):
        batch_size, num_frames = timestep.shape
        timestep = rearrange(timestep, "b t -> (b t)")
        delta_timestep = rearrange(delta_timestep, "b t -> (b t)")

        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        delta_timestep = self.timesteps_proj(delta_timestep)

        delta_embedder_dtype = next(iter(self.delta_embedder.parameters())).dtype
        if delta_timestep.dtype != delta_embedder_dtype and delta_embedder_dtype != torch.int8:
            delta_timestep = delta_timestep.to(delta_embedder_dtype)
        delta_emb = self.delta_embedder(delta_timestep).type_as(encoder_hidden_states)

        gate = self.delta_emb_gate.to(delta_embedder_dtype)
        self.gate_track = float(gate)

        rt_emb = (1 - gate) * temb + gate * delta_emb
        timestep_proj = self.time_proj(self.act_fn(rt_emb))

        rt_emb = rearrange(rt_emb, "(b t) c -> b t c", b=batch_size).repeat_interleave(token_per_frame, dim=1)
        timestep_proj = rearrange(timestep_proj, "(b t) c -> b t c", b=batch_size).repeat_interleave(
            token_per_frame, dim=1
        )

        return rt_emb, timestep_proj

    def forward(
        self,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        far_cfg=None,
        clean_timestep=None,
        is_causal=True,
    ):
        if self.deltatime_type == "r":
            delta_timestep = r_timestep
        elif self.deltatime_type == "t-r":
            delta_timestep = timestep - r_timestep
        else:
            raise NotImplementedError

        if is_causal:
            full_frame_timestep, full_frame_timestep_proj = self.forward_timestep(
                timestep[:, -far_cfg["num_full_frames"] :],
                delta_timestep[:, -far_cfg["num_full_frames"] :],
                encoder_hidden_states,
                far_cfg["full_token_per_frame"],
            )  # noqa: E501
            compressed_frame_timestep, compressed_frame_timestep_proj = self.forward_timestep(
                timestep[:, : -far_cfg["num_full_frames"]],
                delta_timestep[:, : -far_cfg["num_full_frames"]],
                encoder_hidden_states,
                far_cfg["compressed_token_per_frame"],
            )  # noqa: E501

            if clean_timestep is not None:
                clean_timestep, clean_timestep_proj = self.forward_timestep(
                    clean_timestep, clean_timestep, encoder_hidden_states, far_cfg["full_token_per_frame"]
                )  # noqa: E501
                timestep = torch.cat([compressed_frame_timestep, full_frame_timestep, clean_timestep], dim=1)
                timestep_proj = torch.cat(
                    [compressed_frame_timestep_proj, full_frame_timestep_proj, clean_timestep_proj], dim=1
                )
            else:
                timestep = torch.cat([compressed_frame_timestep, full_frame_timestep], dim=1)
                timestep_proj = torch.cat([compressed_frame_timestep_proj, full_frame_timestep_proj], dim=1)

            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
            if encoder_hidden_states_image is not None:
                encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
        else:
            timestep, timestep_proj = self.forward_timestep(
                timestep, delta_timestep, encoder_hidden_states, far_cfg["full_token_per_frame"]
            )  # noqa: E501

            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
            if encoder_hidden_states_image is not None:
                encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return timestep, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class AnyFlowRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        compressed_patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.compressed_patch_size = compressed_patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def avg_pool_complex(self, freq: torch.Tensor, kernel_size: int, stride: int):

        real = freq.real  # [B, C, L], float
        real = real.transpose(0, 1).unsqueeze(0)
        imag = freq.imag  # [B, C, L], float
        imag = imag.transpose(0, 1).unsqueeze(0)

        pr = F.avg_pool1d(real, kernel_size, stride)
        pi = F.avg_pool1d(imag, kernel_size, stride)

        pr = pr.squeeze(0).transpose(0, 1)
        pi = pi.squeeze(0).transpose(0, 1)

        norm = torch.sqrt(pr**2 + pi**2)
        pr_unit = pr / norm
        pi_unit = pi / norm

        return torch.complex(pr_unit, pi_unit)

    def _forward_compressed_frame(self, num_frames, height, width, device):
        ppf, pph, ppw = num_frames, height, width
        downscale = [self.compressed_patch_size[i] // self.patch_size[i] for i in range(len(self.patch_size))]

        self.freqs = self.freqs.to(device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = self.avg_pool_complex(freqs[0], kernel_size=downscale[0], stride=downscale[0])
        freqs_h = self.avg_pool_complex(freqs[1], kernel_size=downscale[1], stride=downscale[1])
        freqs_w = self.avg_pool_complex(freqs[2], kernel_size=downscale[2], stride=downscale[2])

        freqs_f = freqs_f[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs_h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs_w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)
        return freqs

    def _forward_full_frame(self, num_frames, height, width, device) -> torch.Tensor:
        ppf, pph, ppw = num_frames, height, width

        self.freqs = self.freqs.to(device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)
        return freqs

    def forward(self, far_cfg, device, clean_hidden_states=None, is_causal=True):
        if is_causal:
            full_frame_freqs = self._forward_full_frame(
                num_frames=far_cfg["total_frames"],
                height=far_cfg["full_frame_shape"][0],
                width=far_cfg["full_frame_shape"][1],
                device=device,
            )
            compressed_frame_freqs = self._forward_compressed_frame(
                num_frames=far_cfg["total_frames"],
                height=far_cfg["compressed_frame_shape"][0],
                width=far_cfg["compressed_frame_shape"][1],
                device=device,
            )

            compressed_frame_freqs, full_frame_freqs = (
                compressed_frame_freqs[: far_cfg["num_compressed_frames"]],
                full_frame_freqs[far_cfg["num_compressed_frames"] :],
            )  # noqa: E501

            compressed_frame_freqs = compressed_frame_freqs.flatten(start_dim=0, end_dim=2)
            full_frame_freqs = full_frame_freqs.flatten(start_dim=0, end_dim=2)

            if clean_hidden_states is not None:
                freqs = torch.cat([compressed_frame_freqs, full_frame_freqs, full_frame_freqs], dim=0)
            else:
                freqs = torch.cat([compressed_frame_freqs, full_frame_freqs], dim=0)

            freqs = freqs[None, None, ...]

            return {"query": freqs, "key": freqs}
        else:
            freqs = self._forward_full_frame(
                num_frames=far_cfg["total_frames"],
                height=far_cfg["full_frame_shape"][0],
                width=far_cfg["full_frame_shape"][1],
                device=device,
            )
            freqs = freqs.flatten(start_dim=0, end_dim=2)
            freqs = freqs[None, None, ...]
            return {"query": freqs, "key": freqs}


class AnyFlowTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=AnyFlowSelfAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=AnyFlowCrossAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache=None,
        kv_cache_flag=None,
    ) -> torch.Tensor:

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            shift_msa.squeeze(2),
            scale_msa.squeeze(2),
            gate_msa.squeeze(2),
            c_shift_msa.squeeze(2),
            c_scale_msa.squeeze(2),
            c_gate_msa.squeeze(2),
        )  # noqa: E501

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            kv_cache_flag=kv_cache_flag,
        )  # noqa: E501
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class AnyFlowTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    Bidirectional 3D Transformer for AnyFlow flow-map sampling.

    The architecture is the v0.35.1 Wan2.1 3D DiT backbone with one structural change: the timestep
    embedder is replaced by ``AnyFlowDualTimestepTextImageEmbedding`` so that every forward call conditions
    on both the source timestep ``t`` and the target timestep ``r``. This is the embedding required to
    learn the flow map :math:`\Phi_{r\leftarrow t}` introduced in
    [AnyFlow](https://huggingface.co/papers/<arxiv-id>).

    For frame-level autoregressive (FAR causal) generation, use ``AnyFlowFARTransformer3DModel`` instead;
    that variant adds the FAR causal block-mask and a compressed-frame patch embedding on top of the same
    backbone.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Number of attention heads.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input latent.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output latent.
        text_dim (`int`, defaults to `4096`):
            Input dimension for text embeddings (UMT5).
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            Number of transformer blocks.
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`Optional[str]`, defaults to `'rms_norm_across_heads'`):
            Query/key normalization scheme.
        eps (`float`, defaults to `1e-6`):
            Epsilon for normalization layers.
        image_dim (`Optional[int]`, *optional*, defaults to `None`):
            Image embedding dimension for I2V conditioning (`1280` for the original Wan2.1-I2V model).
        added_kv_proj_dim (`Optional[int]`, *optional*, defaults to `None`):
            The number of channels to use for the added key/value projections. If `None`, no projection is added.
        rope_max_seq_len (`int`, defaults to `1024`):
            Maximum sequence length used to precompute rotary position frequencies.
        gate_value (`float`, defaults to `0.25`):
            Mixing gate between source-timestep and delta-timestep embeddings (the AnyFlow paper's :math:`g`
            parameter, fixed at 0.25 in stage-1 distillation).
        deltatime_type (`str`, defaults to `'r'`):
            Either ``"r"`` (delta is the target timestep) or ``"t-r"`` (delta is the absolute interval).
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["AnyFlowTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        gate_value: float = 0.25,
        deltatime_type: str = "r",
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding. ``compressed_patch_size`` is unused in the bidirectional path;
        # we forward ``patch_size`` itself so the rotary helper has a valid (no-op) value.
        self.rope = AnyFlowRotaryPosEmbed(attention_head_dim, patch_size, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embedding (always dual-timestep for AnyFlow distilled checkpoints).
        self.condition_embedder = AnyFlowDualTimestepTextImageEmbedding(
            dim=inner_dim,
            gate_value=gate_value,
            deltatime_type=deltatime_type,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                AnyFlowTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def _unpack_latent_sequence(self, latents, num_frames, height, width, patch_size):
        batch_size, num_patches, channels = latents.shape
        height, width = height // patch_size, width // patch_size

        latents = latents.view(
            batch_size * num_frames, height, width, patch_size, patch_size, channels // (patch_size * patch_size)
        )
        latents = latents.permute(0, 5, 1, 3, 2, 4)
        latents = latents.reshape(
            batch_size, num_frames, channels // (patch_size * patch_size), height * patch_size, width * patch_size
        )
        return latents

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple]:
        """
        Bidirectional flow-map forward pass.

        ``hidden_states`` is laid out as ``(B, F, C, H, W)`` (per-frame latents). The input is patchified
        with the standard ``patch_embedding`` (kernel = stride = ``patch_size``) and denoised with global
        bidirectional self-attention over the resulting flat token sequence.
        """
        hidden_states = rearrange(hidden_states, "b f c h w -> b c f h w")
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        full_token_per_frame = (height * width) // (self.config.patch_size[1] * self.config.patch_size[2])

        far_cfg = {
            "total_frames": num_frames,
            "full_frame_shape": (height // self.config.patch_size[1], width // self.config.patch_size[2]),
            "full_token_per_frame": full_token_per_frame,
        }

        rotary_emb = self.rope(far_cfg=far_cfg, device=hidden_states.device, is_causal=False)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            r_timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            is_causal=False,
            far_cfg=far_cfg,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        attention_mask = None

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_mask)

        # Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move shift/scale to hidden_states' device for multi-GPU accelerate inference.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        output = self._unpack_latent_sequence(
            hidden_states,
            num_frames=far_cfg["total_frames"],
            height=height,
            width=width,
            patch_size=self.config.patch_size[1],
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class AnyFlowFARTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    Causal (FAR) 3D Transformer for AnyFlow flow-map sampling with frame-level autoregressive generation.

    Extends the v0.35.1 Wan2.1 backbone with:

    * **FAR causal block-mask** via :func:`torch.nn.attention.flex_attention`, supporting frame-level
      autoregressive generation (FAR; [Gu et al., 2025](https://arxiv.org/abs/2503.19325)).
    * **Compressed-frame patch embedding** ``far_patch_embedding`` for context (already-generated) frames,
      initialized from ``patch_embedding`` via trilinear interpolation so a freshly constructed model is
      already at a reasonable starting point even before LoRA fine-tuning.
    * **Dual-timestep flow-map embedding** for any-step sampling (same as ``AnyFlowTransformer3DModel``).

    Use ``AnyFlowTransformer3DModel`` instead for plain bidirectional T2V — that variant skips the FAR
    causal masking and ``far_patch_embedding`` and is ~5–10% smaller.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for full-resolution chunks.
        compressed_patch_size (`Tuple[int]`, defaults to `(1, 4, 4)`):
            Larger patch dimensions for the FAR-compressed (context) chunks.
        full_chunk_limit (`int`, defaults to `3`):
            Maximum number of full-resolution chunks before earlier chunks are demoted to compressed FAR
            context. The released checkpoints use ``3``.
        num_attention_heads (`int`, defaults to `40`):
            Number of attention heads.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input latent.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output latent.
        text_dim (`int`, defaults to `4096`):
            Input dimension for text embeddings (UMT5).
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            Number of transformer blocks.
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`Optional[str]`, defaults to `'rms_norm_across_heads'`):
            Query/key normalization scheme.
        eps (`float`, defaults to `1e-6`):
            Epsilon for normalization layers.
        image_dim (`Optional[int]`, *optional*, defaults to `None`):
            Image embedding dimension for I2V conditioning.
        added_kv_proj_dim (`Optional[int]`, *optional*, defaults to `None`):
            The number of channels to use for the added key/value projections.
        rope_max_seq_len (`int`, defaults to `1024`):
            Maximum sequence length used to precompute rotary position frequencies.
        gate_value (`float`, defaults to `0.25`):
            Mixing gate between source-timestep and delta-timestep embeddings.
        deltatime_type (`str`, defaults to `'r'`):
            Either ``"r"`` (delta is the target timestep) or ``"t-r"`` (delta is the absolute interval).

    .. note::
        ``chunk_partition`` is **not** a model config field — it is a per-call argument passed to
        :meth:`forward`. Different inference setups (varying ``num_frames`` or full-vs-compressed schedules)
        therefore do not require separate checkpoints.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "far_patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["AnyFlowTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        compressed_patch_size: Tuple[int] = (1, 4, 4),
        full_chunk_limit: int = 3,
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        gate_value: float = 0.25,
        deltatime_type: str = "r",
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding (full + FAR-compressed branches).
        self.rope = AnyFlowRotaryPosEmbed(attention_head_dim, patch_size, compressed_patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        self.far_patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=compressed_patch_size, stride=compressed_patch_size
        )
        # Warm-start the compressed branch from the full-resolution branch by trilinear interpolation. This
        # matches FAR-Dev's `setup_far_model()` initialization. State-dict loading will overwrite these
        # weights for trained checkpoints; the warm-start only matters when constructing a fresh model.
        original_weight = self.patch_embedding.weight.data.view(-1, 1, *patch_size)
        new_weight = F.interpolate(original_weight, size=compressed_patch_size, mode="trilinear", align_corners=False)
        new_weight = new_weight.view(inner_dim, in_channels, *compressed_patch_size)
        with torch.no_grad():
            self.far_patch_embedding.weight.copy_(new_weight)
            self.far_patch_embedding.bias.copy_(self.patch_embedding.bias)

        # 2. Condition embedding (always dual-timestep for AnyFlow distilled checkpoints).
        self.condition_embedder = AnyFlowDualTimestepTextImageEmbedding(
            dim=inner_dim,
            gate_value=gate_value,
            deltatime_type=deltatime_type,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                AnyFlowTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        chunk_partition: List[int],
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        clean_hidden_states: Optional[torch.Tensor] = None,
        clean_timestep: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        kv_cache_flag: Optional[Dict[str, Any]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, AnyFlowFARTransformerOutput, Tuple]:
        """
        FAR causal forward pass. Dispatches to one of three internal paths:

        * ``kv_cache is None`` → causal training rollout (returns
          :class:`Transformer2DModelOutput`).
        * ``kv_cache is not None`` and ``kv_cache_flag["is_cache_step"]`` → cache-prefill (returns
          :class:`AnyFlowFARTransformerOutput` with ``sample=None``).
        * Otherwise → autoregressive inference step (returns :class:`AnyFlowFARTransformerOutput`).

        Args:
            hidden_states (`torch.Tensor`): Latent input of shape ``(B, F, C, H, W)``.
            timestep, r_timestep (`torch.Tensor`): Source / target diffusion timesteps.
            encoder_hidden_states (`torch.Tensor`): UMT5 text embeddings.
            chunk_partition (`List[int]`): Per-chunk frame counts; total must match the number of latent
                frames in ``hidden_states``.
            encoder_hidden_states_image (`torch.Tensor`, *optional*): I2V image embedding.
            clean_hidden_states, clean_timestep (`torch.Tensor`, *optional*): Clean conditioning frames
                used by the training rollout.
            kv_cache, kv_cache_flag (*optional*): Per-block KV cache and metadata for autoregressive
                inference.
            attention_kwargs (*optional*): forwarded to the attention processors.
            return_dict (`bool`, defaults to `True`): If `False`, returns positional tuples.
        """
        common = {
            "hidden_states": hidden_states,
            "chunk_partition": chunk_partition,
            "timestep": timestep,
            "r_timestep": r_timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_image": encoder_hidden_states_image,
            "return_dict": return_dict,
            "attention_kwargs": attention_kwargs,
        }
        if kv_cache is not None:
            common["kv_cache"] = kv_cache
            common["kv_cache_flag"] = kv_cache_flag
            if kv_cache_flag is not None and kv_cache_flag.get("is_cache_step"):
                return self._forward_cache(
                    clean_hidden_states=clean_hidden_states,
                    clean_timestep=clean_timestep,
                    **common,
                )
            return self._forward_inference(**common)
        return self._forward_train(
            clean_hidden_states=clean_hidden_states,
            clean_timestep=clean_timestep,
            **common,
        )

    def _unpack_latent_sequence(self, latents, num_frames, height, width, patch_size):
        batch_size, num_patches, channels = latents.shape
        height, width = height // patch_size, width // patch_size

        latents = latents.view(
            batch_size * num_frames, height, width, patch_size, patch_size, channels // (patch_size * patch_size)
        )

        latents = latents.permute(0, 5, 1, 3, 2, 4)
        latents = latents.reshape(
            batch_size, num_frames, channels // (patch_size * patch_size), height * patch_size, width * patch_size
        )
        return latents

    def forward_far_patchify(self, hidden_states, far_cfg, clean_hidden_states=None):

        full_hidden_states, compressed_hidden_states = (
            hidden_states[:, :, far_cfg["num_compressed_frames"] :],
            hidden_states[:, :, : far_cfg["num_compressed_frames"]],
        )  # noqa: E501

        patchified_full_hidden_states = (
            self.patch_embedding(full_hidden_states).flatten(start_dim=2, end_dim=4).transpose(1, 2)
        )
        if clean_hidden_states is not None:
            clean_hidden_states = (
                self.patch_embedding(clean_hidden_states).flatten(start_dim=2, end_dim=4).transpose(1, 2)
            )
            patchified_full_hidden_states = torch.cat([patchified_full_hidden_states, clean_hidden_states], dim=1)

        if far_cfg["num_compressed_frames"] > 0:
            patchified_compressed_hidden_states = (
                self.far_patch_embedding(compressed_hidden_states).flatten(start_dim=2, end_dim=4).transpose(1, 2)
            )
            hidden_states = torch.cat([patchified_compressed_hidden_states, patchified_full_hidden_states], dim=1)
        else:
            hidden_states = patchified_full_hidden_states
        return hidden_states

    def forward_far_patchify_inference(self, hidden_states):
        hidden_states = self.patch_embedding(hidden_states).flatten(start_dim=2, end_dim=4).transpose(1, 2)
        return hidden_states

    def _build_causal_mask(self, far_cfg, clean_hidden_states, device, dtype):
        chunk_partition = far_cfg["chunk_partition"]

        noise_seq_len = clean_seq_len = far_cfg["num_full_frames"] * far_cfg["full_token_per_frame"]
        context_seq_len = far_cfg["num_compressed_frames"] * far_cfg["compressed_token_per_frame"]

        noise_start = context_seq_len
        noise_end = noise_start + noise_seq_len

        clean_start = context_seq_len + noise_seq_len
        clean_end = clean_start + clean_seq_len

        if clean_hidden_states is not None:
            real_seq_len = context_seq_len + noise_seq_len + clean_seq_len
        else:
            real_seq_len = context_seq_len + noise_seq_len

        padded_seq_len = int(math.ceil(real_seq_len / 128.0) * 128.0)

        if clean_hidden_states is not None:
            context_chunk_partition, noise_chunk_partition = (
                chunk_partition[: far_cfg["num_compressed_chunk"]],
                chunk_partition[far_cfg["num_compressed_chunk"] :],
            )  # noqa: E501

            if len(context_chunk_partition) != 0:
                context_frame_idx = torch.cat(
                    [
                        torch.ones(chunk_len * far_cfg["compressed_token_per_frame"], device=device) * chunk_idx
                        for chunk_idx, chunk_len in enumerate(context_chunk_partition)
                    ]
                )  # noqa: E501
            else:
                context_frame_idx = None
            noise_frame_idx = clean_frame_idx = torch.cat(
                [
                    torch.ones(chunk_len * far_cfg["full_token_per_frame"], device=device)
                    * (chunk_idx + len(context_chunk_partition))
                    for chunk_idx, chunk_len in enumerate(noise_chunk_partition)
                ]
            )  # noqa: E501
            pad_frame_idx = torch.zeros(padded_seq_len - real_seq_len, device=device)

            if len(context_chunk_partition) != 0:
                frame_idx = torch.cat([context_frame_idx, noise_frame_idx, clean_frame_idx, pad_frame_idx], dim=0)
            else:
                frame_idx = torch.cat([noise_frame_idx, clean_frame_idx, pad_frame_idx], dim=0)

            def mask_mod(b, h, q_idx, kv_idx):
                # q_idx, kv_idx: LongTensor, range: [0, padded_seq_len)

                # 1) whether is padding
                is_padding = (q_idx >= real_seq_len) | (kv_idx >= real_seq_len)

                # 3) chunk casual
                base = frame_idx[q_idx] >= frame_idx[kv_idx]

                # 4) interval mask
                q_is_noise = (q_idx >= noise_start) & (q_idx < noise_end)
                q_is_clean = (q_idx >= clean_start) & (q_idx < clean_end)

                k_is_noise = (kv_idx >= noise_start) & (kv_idx < noise_end)
                k_is_clean = (kv_idx >= clean_start) & (kv_idx < clean_end)

                # 5) clean -> noise: disallowed
                is_clean_to_noise = q_is_clean & k_is_noise

                # 6) noise -> noise: only same frame
                same_frame_idx = frame_idx[q_idx] == frame_idx[kv_idx]

                noise_to_noise = q_is_noise & k_is_noise
                noise_to_clean = q_is_noise & k_is_clean

                noise_to_noise_allow = noise_to_noise & same_frame_idx
                noise_to_noise_mask = (~noise_to_noise) | noise_to_noise_allow

                noise_to_clean_same = noise_to_clean & same_frame_idx
                noise_to_clean_disallow = noise_to_clean_same

                # attention mask is chunk casual
                allowed = base & ~is_padding & ~is_clean_to_noise & noise_to_noise_mask & ~noise_to_clean_disallow
                return allowed

            return create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=padded_seq_len,
                KV_LEN=padded_seq_len,
                device=device,
                _compile=False,
            )
        else:
            context_chunk_partition, noise_chunk_partition = (
                chunk_partition[: far_cfg["num_compressed_chunk"]],
                chunk_partition[far_cfg["num_compressed_chunk"] :],
            )  # noqa: E501

            if len(context_chunk_partition) != 0:
                context_frame_idx = torch.cat(
                    [
                        torch.ones(chunk_len * far_cfg["compressed_token_per_frame"], device=device) * chunk_idx
                        for chunk_idx, chunk_len in enumerate(context_chunk_partition)
                    ]
                )  # noqa: E501
            else:
                context_frame_idx = None

            noise_frame_idx = torch.cat(
                [
                    torch.ones(chunk_len * far_cfg["full_token_per_frame"], device=device)
                    * (chunk_idx + len(context_chunk_partition))
                    for chunk_idx, chunk_len in enumerate(noise_chunk_partition)
                ]
            )  # noqa: E501
            pad_frame_idx = torch.zeros(padded_seq_len - real_seq_len, device=device)

            if len(context_chunk_partition) != 0:
                frame_idx = torch.cat([context_frame_idx, noise_frame_idx, pad_frame_idx], dim=0)
            else:
                frame_idx = torch.cat([noise_frame_idx, pad_frame_idx], dim=0)

            def mask_mod(b, h, q_idx, kv_idx):
                is_padding = (q_idx >= real_seq_len) | (kv_idx >= real_seq_len)
                base = frame_idx[q_idx] >= frame_idx[kv_idx]
                return base & ~is_padding

            return create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=padded_seq_len,
                KV_LEN=padded_seq_len,
                device=device,
                _compile=False,
            )

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        chunk_partition,
        timestep: torch.LongTensor,
        r_timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        kv_cache=None,
        kv_cache_flag=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        hidden_states = rearrange(hidden_states, "b f c h w -> b c f h w")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        full_token_per_frame = (height // self.config.patch_size[1]) * (width // self.config.patch_size[2])
        compressed_token_per_frame = (height // self.config.compressed_patch_size[1]) * (
            width // self.config.compressed_patch_size[2]
        )

        total_chunks = 1 + kv_cache_flag["num_cached_chunks"]

        if total_chunks >= self.config.full_chunk_limit:
            num_full_chunk, num_compressed_chunk = (
                self.config.full_chunk_limit,
                total_chunks - self.config.full_chunk_limit,
            )
        else:
            num_full_chunk, num_compressed_chunk = total_chunks, 0

        kv_cache_flag["num_cached_full_tokens"] = (
            sum(chunk_partition[num_compressed_chunk : num_compressed_chunk + (num_full_chunk - 1)])
            * full_token_per_frame
        )  # noqa: E501
        kv_cache_flag["num_cached_compressed_tokens"] = (
            sum(chunk_partition[:num_compressed_chunk]) * compressed_token_per_frame
        )

        far_cfg = {
            "total_frames": sum(chunk_partition),
            "num_full_frames": sum(chunk_partition[num_compressed_chunk:]),
            "num_compressed_frames": sum(chunk_partition[:num_compressed_chunk]),
            "full_frame_shape": (height // self.config.patch_size[1], width // self.config.patch_size[2]),
            "compressed_frame_shape": (
                height // self.config.compressed_patch_size[1],
                width // self.config.compressed_patch_size[2],
            ),
            "full_token_per_frame": full_token_per_frame,
            "compressed_token_per_frame": compressed_token_per_frame,
        }

        # step 3: generate attention mask
        attention_mask = None
        hidden_states = self.forward_far_patchify_inference(hidden_states)

        rotary_emb = self.rope(far_cfg=far_cfg, device=hidden_states.device)
        rotary_emb["query"] = rotary_emb["query"][:, :, -hidden_states.shape[1] :]

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            r_timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            far_cfg=far_cfg,  # noqa: E501
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        for index_block, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_mask,
                    kv_cache[index_block],
                    kv_cache_flag,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_mask,
                    kv_cache[index_block],
                    kv_cache_flag,
                )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

        output = self.proj_out(hidden_states)
        output = self._unpack_latent_sequence(
            output, num_frames=chunk_partition[-1], height=height, width=width, patch_size=self.config.patch_size[1]
        )

        if not return_dict:
            return output, kv_cache

        return AnyFlowFARTransformerOutput(sample=output, kv_cache=kv_cache)

    def _forward_cache(
        self,
        hidden_states: torch.Tensor,
        chunk_partition,
        timestep: torch.LongTensor,
        r_timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        clean_hidden_states=None,
        clean_timestep=None,
        kv_cache=None,
        kv_cache_flag=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        hidden_states = rearrange(hidden_states, "b f c h w -> b c f h w")
        if clean_hidden_states is not None:
            clean_hidden_states = rearrange(clean_hidden_states, "b f c h w -> b c f h w")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        full_token_per_frame = (height // self.config.patch_size[1]) * (width // self.config.patch_size[2])
        compressed_token_per_frame = (height // self.config.compressed_patch_size[1]) * (
            width // self.config.compressed_patch_size[2]
        )
        total_chunks = len(chunk_partition)

        full_chunk_limit = self.config.full_chunk_limit - 1

        if total_chunks > full_chunk_limit:
            num_full_chunk, num_compressed_chunk = full_chunk_limit, total_chunks - full_chunk_limit
        else:
            num_full_chunk, num_compressed_chunk = total_chunks, 0

        far_cfg = {
            "total_frames": sum(chunk_partition),
            "num_full_chunk": num_full_chunk,
            "num_full_frames": sum(chunk_partition[num_compressed_chunk:]),
            "num_compressed_chunk": num_compressed_chunk,
            "num_compressed_frames": sum(chunk_partition[:num_compressed_chunk]),
            "full_frame_shape": (height // self.config.patch_size[1], width // self.config.patch_size[2]),
            "compressed_frame_shape": (
                height // self.config.compressed_patch_size[1],
                width // self.config.compressed_patch_size[2],
            ),
            "full_token_per_frame": full_token_per_frame,
            "compressed_token_per_frame": compressed_token_per_frame,
            "chunk_partition": chunk_partition,
        }

        kv_cache_flag["num_full_tokens"] = far_cfg["num_full_frames"] * far_cfg["full_token_per_frame"]
        kv_cache_flag["num_compressed_tokens"] = (
            far_cfg["num_compressed_frames"] * far_cfg["compressed_token_per_frame"]
        )

        # step 3: generate attention mask
        attention_mask = self._build_causal_mask(
            far_cfg, clean_hidden_states=clean_hidden_states, device=hidden_states.device, dtype=hidden_states.dtype
        )

        rotary_emb = self.rope(far_cfg=far_cfg, clean_hidden_states=clean_hidden_states, device=hidden_states.device)
        hidden_states = self.forward_far_patchify(
            hidden_states, far_cfg=far_cfg, clean_hidden_states=clean_hidden_states
        )

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            r_timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            far_cfg=far_cfg,
            clean_timestep=clean_timestep,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        for index_block, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_mask,
                    kv_cache[index_block],
                    kv_cache_flag,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_mask,
                    kv_cache[index_block],
                    kv_cache_flag,
                )

        if not return_dict:
            return None, kv_cache

        return AnyFlowFARTransformerOutput(sample=None, kv_cache=kv_cache)

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        chunk_partition,
        timestep: torch.LongTensor,
        r_timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        clean_hidden_states=None,
        clean_timestep=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        hidden_states = rearrange(hidden_states, "b f c h w -> b c f h w")
        if clean_hidden_states is not None:
            clean_hidden_states = rearrange(clean_hidden_states, "b f c h w -> b c f h w")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        full_token_per_frame = (height // self.config.patch_size[1]) * (width // self.config.patch_size[2])
        compressed_token_per_frame = (height // self.config.compressed_patch_size[1]) * (
            width // self.config.compressed_patch_size[2]
        )
        total_chunks = len(chunk_partition)

        if total_chunks > self.config.full_chunk_limit:
            num_full_chunk, num_compressed_chunk = (
                self.config.full_chunk_limit,
                total_chunks - self.config.full_chunk_limit,
            )
        else:
            num_full_chunk, num_compressed_chunk = total_chunks, 0

        far_cfg = {
            "total_frames": sum(chunk_partition),
            "num_full_chunk": num_full_chunk,
            "num_full_frames": sum(chunk_partition[num_compressed_chunk:]),
            "num_compressed_chunk": num_compressed_chunk,
            "num_compressed_frames": sum(chunk_partition[:num_compressed_chunk]),
            "full_frame_shape": (height // self.config.patch_size[1], width // self.config.patch_size[2]),
            "compressed_frame_shape": (
                height // self.config.compressed_patch_size[1],
                width // self.config.compressed_patch_size[2],
            ),
            "full_token_per_frame": full_token_per_frame,
            "compressed_token_per_frame": compressed_token_per_frame,
            "chunk_partition": chunk_partition,
        }

        # step 3: generate attention mask
        attention_mask = self._build_causal_mask(
            far_cfg, clean_hidden_states=clean_hidden_states, device=hidden_states.device, dtype=hidden_states.dtype
        )

        rotary_emb = self.rope(far_cfg=far_cfg, clean_hidden_states=clean_hidden_states, device=hidden_states.device)

        hidden_states = self.forward_far_patchify(
            hidden_states, far_cfg=far_cfg, clean_hidden_states=clean_hidden_states
        )

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            r_timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            far_cfg=far_cfg,
            clean_timestep=clean_timestep,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        for index_block, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_mask)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

        if clean_hidden_states is not None:
            hidden_states = hidden_states[
                :, : -(far_cfg["num_full_frames"] * far_cfg["full_token_per_frame"])
            ]  # remove clean copy
        output = self.proj_out(
            hidden_states[:, far_cfg["num_compressed_frames"] * far_cfg["compressed_token_per_frame"] :]
        )  # remove far context
        output = self._unpack_latent_sequence(
            output,
            num_frames=far_cfg["num_full_frames"],
            height=height,
            width=width,
            patch_size=self.config.patch_size[1],
        )  # noqa: E501

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
