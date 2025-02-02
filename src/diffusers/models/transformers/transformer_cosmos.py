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

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms

from ..attention import FeedForward
from ..embeddings import Timesteps
from ..normalization import RMSNorm


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
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


class PatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
        bias=True,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=bias
            ),
        )
        self.out = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size, C is the number of channels, T is the temporal dimension, H is the height, and W is
            the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

    def forward(
        self,
        x_BT_HW_D,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = self.norm_final(x_BT_HW_D) * (1 + scale_BT_D.unsqueeze(1)) + shift_BT_D.unsqueeze(1)

        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D


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
            RMSNorm(norm_dim, eps=1e-6, elementwise_affine=True),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            RMSNorm(norm_dim, eps=1e-6, elementwise_affine=True),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            nn.Identity(),
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
        # Note: Does not seem to use atttention masks
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, enable_gqa=True)
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


class DITBuildingBlock(nn.Module):
    def __init__(
        self,
        block_type: str,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
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
            self.block = FeedForward(x_dim, mult=mlp_ratio, activation_fn="gelu", bias=bias)
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
                self.norm_state(x) * (1 + scale_1_1_1_B_D) + shift_1_1_1_B_D,
            )
        elif self.block_type in ["full_attn", "fa"]:
            x = x + gate_1_1_1_B_D * self.block(
                self.norm_state(x) * (1 + scale_1_1_1_B_D) + shift_1_1_1_B_D,
                context=None,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
        elif self.block_type in ["cross_attn", "ca"]:
            x = x + gate_1_1_1_B_D * self.block(
                self.norm_state(x) * (1 + scale_1_1_1_B_D) + shift_1_1_1_B_D,
                context=crossattn_emb,
                crossattn_mask=crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

        return x


class DataType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class VideoPositionEmb(nn.Module):
    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor]) -> torch.Tensor:
        """
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]):
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        # self.register_buffer(
        #     "dim_spatial_range",
        #     torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
        #     persistent=False,
        # )

        # self.register_buffer(
        #     "dim_temporal_range",
        #     torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
        #     persistent=False,
        # )
        self.dim_h = dim_h
        self.dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        dim_spatial_range = (
            torch.arange(0, self.dim_h, 2, dtype=torch.float32, device=self.seq.device)[: (self.dim_h // 2)]
            / self.dim_h
        )
        dim_temporal_range = (
            torch.arange(0, self.dim_t, 2, dtype=torch.float32, device=self.seq.device)[: (self.dim_t // 2)]
            / self.dim_t
        )
        h_spatial_freqs = 1.0 / (h_theta**dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**dim_temporal_range)

        # h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        # w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        # temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        uniform_fps = (fps is None) or (fps.min() == fps.max())
        assert (
            uniform_fps or B == 1 or T == 1
        ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        # apply sequence scaling in temporal dimension
        if fps is None:  # image case
            assert T == 1, "T should be 1 for image batch."
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        """
        Args:
            interpolation (str):
                we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust
                frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        return normalize(emb, dim=-1, eps=1e-6)


class GeneralDITTransformerBlock(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        block_config: str,
        mlp_ratio: float = 4.0,
        x_format: str = "BTHWD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.x_format = x_format
        for block_type in block_config.split("-"):
            self.blocks.append(
                DITBuildingBlock(
                    block_type,
                    x_dim,
                    context_dim,
                    num_heads,
                    mlp_ratio,
                    x_format=self.x_format,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x = x + extra_per_block_pos_emb
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
            )
        return x


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

    def forward(self, timestep: torch.LongTensor, hidden_states_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type(hidden_states_dtype)
        timestep = self.t_embedder(timesteps_proj)
        norm_timesteps_proj = self.norm(timesteps_proj)
        return norm_timesteps_proj, timestep


class GeneralDIT(nn.Module):
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        block_config: str = "FA-CA-MLP",
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        block_x_format: str = "BTHWD",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_cross_attn_mask: bool = False,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 2.0,
        extra_per_block_abs_pos_emb: bool = True,
        extra_per_block_abs_pos_emb_type: str = "learnable",
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.use_cross_attn_mask = use_cross_attn_mask
        self.concat_padding_mask = concat_padding_mask
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_per_block_abs_pos_emb_type = extra_per_block_abs_pos_emb_type.lower()
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio

        self.build_patch_embed()
        self.build_pos_embed()
        self.block_x_format = block_x_format
        self.adaln_lora_dim = adaln_lora_dim

        self.condition_embedder = CosmosEmbedding(model_channels, model_channels)

        self.blocks = nn.ModuleDict()

        for idx in range(num_blocks):
            self.blocks[f"block{idx}"] = GeneralDITTransformerBlock(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                block_config=block_config,
                mlp_ratio=mlp_ratio,
                x_format=self.block_x_format,
                use_adaln_lora=True,
                adaln_lora_dim=adaln_lora_dim,
            )

        self.build_decode_head()

    def build_decode_head(self):
        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=True,
            adaln_lora_dim=self.adaln_lora_dim,
        )

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
        )

    def build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        kwargs = {
            "model_channels": self.model_channels,
            "len_h": self.max_img_h // self.patch_spatial,
            "len_w": self.max_img_w // self.patch_spatial,
            "len_t": self.max_frames // self.patch_temporal,
            "is_learnable": self.pos_emb_learnable,
            "interpolation": self.pos_emb_interpolation,
            "head_dim": self.model_channels // self.num_heads,
            "h_extrapolation_ratio": self.rope_h_extrapolation_ratio,
            "w_extrapolation_ratio": self.rope_w_extrapolation_ratio,
            "t_extrapolation_ratio": self.rope_t_extrapolation_ratio,
        }
        self.pos_embedder = cls_type(
            **kwargs,
        )

        if self.extra_per_block_abs_pos_emb:
            assert self.extra_per_block_abs_pos_emb_type in [
                "learnable",
            ], f"Unknown extra_per_block_abs_pos_emb_type {self.extra_per_block_abs_pos_emb_type}"
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(
                **kwargs,
            )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def decoder_head(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        origin_shape: Tuple[int, int, int, int, int],  # [B, C, T, H, W]
        crossattn_mask: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del crossattn_emb, crossattn_mask
        B, C, T_before_patchify, H_before_patchify, W_before_patchify = origin_shape
        x_BT_HW_D = rearrange(x_B_T_H_W_D, "B T H W D -> (B T) (H W) D")
        x_BT_HW_D = self.final_layer(x_BT_HW_D, emb_B_D, adaln_lora_B_3D=adaln_lora_B_3D)
        # This is to ensure x_BT_HW_D has the correct shape because
        # when we merge T, H, W into one dimension, x_BT_HW_D has shape (B * T * H * W, 1*1, D).
        x_BT_HW_D = x_BT_HW_D.view(
            B * T_before_patchify // self.patch_temporal,
            H_before_patchify // self.patch_spatial * W_before_patchify // self.patch_spatial,
            -1,
        )
        x_B_D_T_H_W = rearrange(
            x_BT_HW_D,
            "(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
            t=self.patch_temporal,
            B=B,
        )
        return x_B_D_T_H_W

    def forward_before_blocks(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        del kwargs
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        original_shape = x.shape
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
        )

        affline_emb_B_D, adaln_lora_B_3D = self.condition_embedder(timesteps, x.dtype)

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.blocks["block0"].x_format == "THWBD":
            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )
            crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")

            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

        elif self.blocks["block0"].x_format == "BTHWD":
            x = x_B_T_H_W_D
        else:
            raise ValueError(f"Unknown x_format {self.blocks[0].x_format}")
        output = {
            "x": x,
            "affline_emb_B_D": affline_emb_B_D,
            "crossattn_emb": crossattn_emb,
            "crossattn_mask": crossattn_mask,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_3D": adaln_lora_B_3D,
            "original_shape": original_shape,
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        return output

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        inputs = self.forward_before_blocks(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            data_type=data_type,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )
        x, affline_emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D, original_shape = (
            inputs["x"],
            inputs["affline_emb_B_D"],
            inputs["crossattn_emb"],
            inputs["crossattn_mask"],
            inputs["rope_emb_L_1_1_D"],
            inputs["adaln_lora_B_3D"],
            inputs["original_shape"],
        )
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"]
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"

        for _, block in self.blocks.items():
            assert (
                self.blocks["block0"].x_format == block.x_format
            ), f"First block has x_format {self.blocks[0].x_format}, got {block.x_format}"

            x = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )

        x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")

        x_B_D_T_H_W = self.decoder_head(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_D=affline_emb_B_D,
            crossattn_emb=None,
            origin_shape=original_shape,
            crossattn_mask=None,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        return x_B_D_T_H_W
