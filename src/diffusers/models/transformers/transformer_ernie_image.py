# Copyright (c) 2025, Baidu Inc. All rights reserved.
# Author: fengzhida (fengzhida@baidu.com)
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

"""
Ernie-Image Transformer2DModel for HuggingFace Diffusers.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...configuration_utils import ConfigMixin, register_to_config
from ..embeddings import Timesteps
from ..modeling_utils import ModelMixin
from ...utils import BaseOutput


@dataclass
class ErnieImageTransformer2DModelOutput(BaseOutput):
    sample: torch.Tensor


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    return out.float()


class EmbedND3(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)], dim=-1)
        emb = emb.unsqueeze(1).permute(2, 0, 1, 3)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


class PatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        return x.reshape(B, D, Hp * Wp).transpose(1, 2).contiguous()


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample.to(self.linear_1.weight.dtype))
        return self.linear_2(self.act(sample).to(self.linear_2.weight.dtype))


class RMSNorm(nn.Module):
    """RMSNorm implementation matching Megatron's TENorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 内部计算转换为FP32，对齐transform engine的TENorm计算精度
        x_norm = self._norm(x.float())
        output = x_norm * self.weight.float()
        return output.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6, qk_layernorm: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # Separate Q, K, V projections (matches converted weights)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.qk_layernorm = qk_layernorm
        if qk_layernorm:
            # self.q_layernorm = RMSNorm(self.head_dim, eps=eps)
            # self.k_layernorm = RMSNorm(self.head_dim, eps=eps)
            self.q_layernorm = torch.nn.RMSNorm(self.head_dim, eps=eps)
            self.k_layernorm = torch.nn.RMSNorm(self.head_dim, eps=eps)

    def forward(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        S, B, H = x.shape
        # Separate Q, K, V projections
        q = self.q_proj(x).view(S, B, self.num_heads, self.head_dim).contiguous()
        k = self.k_proj(x).view(S, B, self.num_heads, self.head_dim).contiguous()
        v = self.v_proj(x).view(S, B, self.num_heads, self.head_dim).contiguous()
        if self.qk_layernorm:
            q, k = self.q_layernorm(q), self.k_layernorm(k)
        q, k = self._apply_rotary(q, rotary_pos_emb), self._apply_rotary(k, rotary_pos_emb)
        q, k, v = q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3)
        attn_mask = ~attention_mask if attention_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        return self.linear_proj(out.permute(2, 0, 1, 3).reshape(S, B, H))

    def _apply_rotary(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding.

        Matches Megatron's _apply_rotary_pos_emb_bshd with rotary_interleaved=False.
        freqs: [S, B, 1, dim] containing angles [θ0, θ0, θ1, θ1, ...]
        """
        rot_dim = freqs.shape[-1]
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]

        cos_ = torch.cos(freqs).to(x.dtype)
        sin_ = torch.sin(freqs).to(x.dtype)

        # Non-interleaved rotate_half: [-x2, x1]
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)

        x = x * cos_ + x_rotated * sin_
        return torch.cat((x, x_pass), dim=-1)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        # Separate gate and up projections (matches converted weights)
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))


class SharedAdaLNBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float = 1e-6, qk_layernorm: bool = True):
        super().__init__()
        # self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps)
        self.adaLN_sa_ln = torch.nn.RMSNorm(hidden_size, eps=eps)
        self.self_attention = Attention(hidden_size, num_heads, eps=eps, qk_layernorm=qk_layernorm)
        # self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps)
        self.adaLN_mlp_ln = torch.nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = FeedForward(hidden_size, ffn_hidden_size)

    def forward(self, x, rotary_pos_emb, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, attention_mask=None):
        residual = x
        x = self.adaLN_sa_ln(x)
        x = self._modulate(x, shift_msa, scale_msa)
        attn_out = self.self_attention(x, rotary_pos_emb, attention_mask)
        x = residual + self._apply_gate(gate_msa, attn_out)
        residual = x
        x = self._modulate(self.adaLN_mlp_ln(x), shift_mlp, scale_mlp)
        return residual + self._apply_gate(gate_mlp, self.mlp(x))

    def _modulate(self, x, shift, scale):
        """AdaLN modulation: x * (1 + scale) + shift，在FP32下计算确保数值稳定"""
        x_fp32 = x.float()
        shift_fp32 = shift.float()
        scale_fp32 = scale.float()
        out = x_fp32 * (1 + scale_fp32) + shift_fp32
        return out.to(x.dtype)

    def _apply_gate(self, gate, x):
        """Gate乘法在FP32下计算，对齐TE精度"""
        return (gate.float() * x.float()).to(x.dtype)

class AdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)
        # 对齐 Megatron 实现：zero init
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        # Broadcast conditioning to sequence dimension
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x


class ErnieImageTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        num_layers: int = 24,
        ffn_hidden_size: int = 8192,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 2560,
        rope_theta: int = 256,
        rope_axes_dim: Tuple[int, int, int] = (32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_in_dim = text_in_dim

        self.x_embedder = PatchEmbedDynamic(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(hidden_size, hidden_size)
        self.pos_embed = EmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.layers = nn.ModuleList([SharedAdaLNBlock(hidden_size, num_attention_heads, ffn_hidden_size, eps, qk_layernorm=qk_layernorm) for _ in range(num_layers)])
        self.final_norm = AdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: List[torch.Tensor], return_dict: bool = True):
        device, dtype = hidden_states.device, hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
        N_img = Hp * Wp

        img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
        text_bth, text_lens = self._pad_text(encoder_hidden_states, device, dtype)
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]
        text_sbh = text_bth.transpose(0, 1).contiguous()

        x = torch.cat([img_sbh, text_sbh], dim=0)
        S = x.shape[0]

        # Position IDs
        text_ids = torch.cat([torch.arange(Tmax, device=device, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1), torch.zeros((B, Tmax, 2), device=device)], dim=-1) if Tmax > 0 else torch.zeros((B, 0, 3), device=device)
        grid_yx = torch.stack(torch.meshgrid(torch.arange(Hp, device=device, dtype=torch.float32), torch.arange(Wp, device=device, dtype=torch.float32), indexing="ij"), dim=-1).reshape(-1, 2)
        image_ids = torch.cat([text_lens.float().view(B, 1, 1).expand(-1, N_img, -1), grid_yx.view(1, N_img, 2).expand(B, -1, -1)], dim=-1)
        rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

        # Attention mask
        valid_text = torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1) if Tmax > 0 else torch.zeros((B, 0), device=device, dtype=torch.bool)
        attention_mask = (~torch.cat([torch.ones((B, N_img), device=device, dtype=torch.bool), valid_text], dim=1))[:, None, None, :]

        # AdaLN
        c = self.time_embedding(self.time_proj(timestep.to(dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [t.unsqueeze(0).expand(S, -1, -1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)]
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x, rotary_pos_emb, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, attention_mask
                )
            else:
                x = layer(x, rotary_pos_emb, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, attention_mask)
        x = self.final_norm(x, c).type_as(x)
        patches = self.final_linear(x)[:N_img].transpose(0, 1).contiguous()
        output = patches.view(B, Hp, Wp, p, p, self.out_channels).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.out_channels, H, W)

        return ErnieImageTransformer2DModelOutput(sample=output) if return_dict else (output,)

    def _pad_text(self, text_hiddens: List[torch.Tensor], device: torch.device, dtype: torch.dtype):
        B = len(text_hiddens)
        if B == 0:
            return torch.zeros((0, 0, self.text_in_dim), device=device, dtype=dtype), torch.zeros((0,), device=device, dtype=torch.long)
        normalized = [th.squeeze(1).to(device).to(dtype) if th.dim() == 3 else th.to(device).to(dtype) for th in text_hiddens]
        lens = torch.tensor([t.shape[0] for t in normalized], device=device, dtype=torch.long)
        Tmax = int(lens.max().item())
        text_bth = torch.zeros((B, Tmax, self.text_in_dim), device=device, dtype=dtype)
        for i, t in enumerate(normalized):
            text_bth[i, :t.shape[0], :] = t
        return text_bth, lens
