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

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


def adaln_norm_state(norm_state, x, scale, shift):
    normalized = norm_state(x)
    return normalized * (1 + scale) + shift


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    cur_seq_len = t.shape[0]

    freqs = freqs[:cur_seq_len]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        out = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        if self.weight is None:
            return out
        else:
            return out * self.weight.to(dtype=x.dtype, device=x.device)


def get_normalization(name: str, channels: int):
    if name == "I":
        return nn.Identity()
    elif name == "R":
        return RMSNorm(channels, eps=1e-6)
    else:
        raise ValueError(f"Normalization {name} not found")


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "SSI",
        qkv_norm_mode: str = "per_head",
        backend: str = "transformer_engine",
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()

        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format

        if self.qkv_norm_mode == "per_head":
            norm_dim = dim_head
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        self.backend = backend

        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[0], norm_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[1], norm_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[2], norm_dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

    def cal_qkv(
        self, x, context=None, mask=None, rope_emb=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.to_q[0](x)
        context = x if context is None else context
        k = self.to_k[0](context)
        v = self.to_v[0](context)
        q, k, v = (rearrange(t, "s b (n c) -> b n s c", n=self.heads, c=self.dim_head) for t in (q, k, v))

        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            apply_rotary_pos_emb(q, rope_emb)
            apply_rotary_pos_emb(k, rope_emb)
            q_shape = q.shape
            q = q.reshape(*q.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
            q = torch.cat([rope_emb[..., 0] * q[..., 0], rope_emb[..., 1] * q[..., 1]], dim=-1)
            # q = rope_emb[..., 0] * q[..., 0] + rope_emb[..., 1] * q[..., 1]
            q = q.movedim(-1, -2).reshape(*q_shape).to(x.dtype)

            # apply_rotary_pos_emb inlined
            k_shape = k.shape
            k = k.reshape(*k.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
            k = torch.cat([rope_emb[..., 0] * k[..., 0], rope_emb[..., 1] * k[..., 1]], dim=-1)
            # k = rope_emb[..., 0] * k[..., 0] + rope_emb[..., 1] * k[..., 1]
            k = k.movedim(-1, -2).reshape(*k_shape).to(x.dtype)
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = rearrange(out, "b n s c -> s b (n c)")
        out = self.to_out(out)
        return out

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]):
                The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        return self.cal_attn(q, k, v, mask)


class VideoAttn(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        bias: bool = False,
        qkv_norm_mode: str = "per_head",
        x_format: str = "BTHWD",
    ) -> None:
        super().__init__()
        self.x_format = x_format

        self.attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_bias=bias,
            qkv_norm="RRI",
            out_bias=bias,
            qkv_norm_mode=qkv_norm_mode,
            qkv_format="sbhd",
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_T_H_W_B_D = x
        context_M_B_D = context
        T, H, W, B, D = x_T_H_W_B_D.shape
        x_THW_B_D = rearrange(x_T_H_W_B_D, "t h w b d -> (t h w) b d")
        x_THW_B_D = self.attn(
            x_THW_B_D,
            context_M_B_D,
            crossattn_mask,
            rope_emb=rope_emb_L_1_1_D,
        )
        x_T_H_W_B_D = rearrange(x_THW_B_D, "(t h w) b d -> t h w b d", h=H, w=W)
        return x_T_H_W_B_D


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        assert self.dropout.p == 0.0, "we skip dropout"
        return self.layer2(x)


class GPT2FeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.GELU(),
            is_gated=False,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class DITBuildingBlock(nn.Module):
    def __init__(
        self,
        block_type: str,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        mlp_dropout: float = 0.0,
        qkv_norm_mode: str = "per_head",
        x_format: str = "BTHWD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ) -> None:
        block_type = block_type.lower()

        super().__init__()
        self.x_format = x_format
        if block_type in ["cross_attn", "ca"]:
            self.block = VideoAttn(
                x_dim,
                context_dim,
                num_heads,
                bias=bias,
                qkv_norm_mode=qkv_norm_mode,
                x_format=self.x_format,
            )
        elif block_type in ["full_attn", "fa"]:
            self.block = VideoAttn(
                x_dim, None, num_heads, bias=bias, qkv_norm_mode=qkv_norm_mode, x_format=self.x_format
            )
        elif block_type in ["mlp", "ff"]:
            self.block = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), dropout=mlp_dropout, bias=bias)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.block_type = block_type
        self.use_adaln_lora = use_adaln_lora

        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = 3
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, self.n_adaln_chunks * x_dim, bias=False))

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for dynamically configured blocks with adaptive normalization.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, D) or (T, H, W, B, D).
            emb_B_D (Tensor): Embedding tensor for adaptive layer normalization modulation.
            crossattn_emb (Tensor): Tensor for cross-attention blocks.
            crossattn_mask (Optional[Tensor]): Optional mask for cross-attention.
            rope_emb_L_1_1_D (Optional[Tensor]):
            Rotary positional embedding tensor of shape (L, 1, 1, D). L == THW for current video training.

        Returns:
            Tensor: The output tensor after processing through the configured block and adaptive normalization.
        """
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift_1_1_1_B_D, scale_1_1_1_B_D, gate_1_1_1_B_D = (
            shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        if self.block_type in ["mlp", "ff"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
            )
        elif self.block_type in ["full_attn", "fa"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                context=None,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
        elif self.block_type in ["cross_attn", "ca"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                context=crossattn_emb,
                crossattn_mask=crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

        return x
