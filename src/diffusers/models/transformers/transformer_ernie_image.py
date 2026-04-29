# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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

import inspect
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ..attention import AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ErnieImageTransformer2DModelOutput(BaseOutput):
    sample: torch.Tensor


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    return out.float()


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)], dim=-1)
        emb = emb.unsqueeze(2)  # [B, S, 1, head_dim//2]
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)  # [B, S, 1, head_dim]


class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, dim, height, width = x.shape
        return x.reshape(batch_size, dim, height * width).transpose(1, 2).contiguous()


class ErnieImageSingleStreamAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ErnieImageSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE: same rotate_half logic as Megatron _apply_rotary_pos_emb_bshd (rotary_interleaved=False)
        # x_in: [B, S, heads, head_dim], freqs_cis: [B, S, 1, head_dim] with angles [θ0,θ0,θ1,θ1,...]
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            rot_dim = freqs_cis.shape[-1]
            x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
            cos_ = torch.cos(freqs_cis).to(x.dtype)
            sin_ = torch.sin(freqs_cis).to(x.dtype)
            # Non-interleaved rotate_half: [-x2, x1]
            x1, x2 = x.chunk(2, dim=-1)
            x_rotated = torch.cat((-x2, x1), dim=-1)
            return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
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

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        output = attn.to_out[0](hidden_states)

        return output


class ErnieImageAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = ErnieImageSingleStreamAttnProcessor

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: str = "rms_norm",
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_proj_bias = added_proj_bias

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        # QK Norm
        if qk_norm == "layer_norm":
            self.norm_q = torch.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = torch.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm":
            self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."
            )

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))

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
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


class ErnieImageFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        # Separate gate and up projections (matches converted weights)
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))


class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float = 1e-6, qk_layernorm: bool = True
    ):
        super().__init__()
        self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieImageAttention(
            query_dim=hidden_size,
            dim_head=hidden_size // num_heads,
            heads=num_heads,
            qk_norm="rms_norm" if qk_layernorm else None,
            eps=eps,
            bias=False,
            out_bias=False,
            processor=ErnieImageSingleStreamAttnProcessor(),
        )
        self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size)

    def forward(
        self,
        x,
        rotary_pos_emb,
        temb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb
        residual = x
        x = self.adaLN_sa_ln(x)
        x = (x.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)
        x_bsh = x.permute(1, 0, 2)  # [S, B, H] → [B, S, H] for diffusers Attention (batch-first)
        attn_out = self.self_attention(x_bsh, attention_mask=attention_mask, image_rotary_emb=rotary_pos_emb)
        attn_out = attn_out.permute(1, 0, 2)  # [B, S, H] → [S, B, H]
        x = residual + (gate_msa.float() * attn_out.float()).to(x.dtype)
        residual = x
        x = self.adaLN_mlp_ln(x)
        x = (x.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(x.dtype)
        return residual + (gate_mlp.float() * self.mlp(x).float()).to(x.dtype)


class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        # Broadcast conditioning to sequence dimension
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x


class ErnieImageTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _repeated_blocks = ["ErnieImageSharedAdaLNBlock"]

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
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_in_dim = text_in_dim

        self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(hidden_size, hidden_size)
        self.pos_embed = ErnieImageEmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.layers = nn.ModuleList(
            [
                ErnieImageSharedAdaLNBlock(
                    hidden_size, num_attention_heads, ffn_hidden_size, eps, qk_layernorm=qk_layernorm
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = ErnieImageAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        # encoder_hidden_states: List[torch.Tensor],
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
        return_dict: bool = True,
    ):
        device, dtype = hidden_states.device, hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
        N_img = Hp * Wp

        img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
        # text_bth, text_lens = self._pad_text(encoder_hidden_states, device, dtype)
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]
        text_sbh = text_bth.transpose(0, 1).contiguous()

        x = torch.cat([img_sbh, text_sbh], dim=0)
        S = x.shape[0]

        # Position IDs
        text_ids = (
            torch.cat(
                [
                    torch.arange(Tmax, device=device, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1),
                    torch.zeros((B, Tmax, 2), device=device),
                ],
                dim=-1,
            )
            if Tmax > 0
            else torch.zeros((B, 0, 3), device=device)
        )
        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(Hp, device=device, dtype=torch.float32),
                torch.arange(Wp, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        image_ids = torch.cat(
            [text_lens.float().view(B, 1, 1).expand(-1, N_img, -1), grid_yx.view(1, N_img, 2).expand(B, -1, -1)],
            dim=-1,
        )
        rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

        # Attention mask: True = valid (attend), False = padding (mask out), matches sdpa bool convention
        valid_text = (
            torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1)
            if Tmax > 0
            else torch.zeros((B, 0), device=device, dtype=torch.bool)
        )
        attention_mask = torch.cat([torch.ones((B, N_img), device=device, dtype=torch.bool), valid_text], dim=1)[
            :, None, None, :
        ]

        # AdaLN
        sample = self.time_proj(timestep)
        sample = sample.to(dtype=dtype)
        c = self.time_embedding(sample)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.unsqueeze(0).expand(S, -1, -1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)
        ]
        for layer in self.layers:
            temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(
                    layer,
                    x,
                    rotary_pos_emb,
                    temb,
                    attention_mask,
                )
            else:
                x = layer(x, rotary_pos_emb, temb, attention_mask)
        x = self.final_norm(x, c).type_as(x)
        patches = self.final_linear(x)[:N_img].transpose(0, 1).contiguous()
        output = (
            patches.view(B, Hp, Wp, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, self.out_channels, H, W)
        )

        return ErnieImageTransformer2DModelOutput(sample=output) if return_dict else (output,)
