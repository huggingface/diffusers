# Copyright 2024 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ..attention import FeedForward
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle


def attention(q, k, v, attn_mask=None):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(q.dtype)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0, is_causal=False)

    x = x.transpose(1, 2)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def get_activation_layer(act_type):
    """get activation layer

    Args:
        act_type (str): the activation type

    Returns:
        torch.nn.functional: the activation layer
    """
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        # Approximate `tanh` requires torch >= 1.13
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x' for the purpose of
    broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False. When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape. AssertionError: If the target tensor
        'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided frequency
    tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor is reshaped for
    broadcasting compatibility. The resulting tensors contain rotary embeddings and are returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")


def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLPEmbedder(nn.Module):
    """copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py"""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()

        patch_size = tuple(patch_size)
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=t.device
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True),
        )

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        act_layer = get_activation_layer(act_type)

        self.mlp = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="silu", dropout=mlp_drop_rate)

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        attn = attention(q, k, v, attn_mask=attn_mask)

        x = x + self.self_attn_proj(attn) * gate_msa.unsqueeze(1)

        # FFN Layer
        x = x + self.mlp(self.norm2(x)) * gate_mlp.unsqueeze(1)

        return x


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ):
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device).bool()
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of heads_num
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens
            self_attn_mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    """
    A single token refiner block for llm text embedding refine.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        num_attention_heads,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True)

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer)
        # Build context embedding layer
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer)

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=num_attention_heads,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
    ):
        original_dtype = x.dtype
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            context_aware_representations = context_aware_representations.to(original_dtype)

        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, mask)

        return x


class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6)
        self.k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        txt_len: int,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        
        qkv, mlp = torch.split(self.linear1(norm_hidden_states), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if image_rotary_emb is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, image_rotary_emb, head_first=False)
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        attn = attention(q, k, v)

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        output = hidden_states + output * gate.unsqueeze(1)
        return output


class HunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
    ):
        super().__init__()

        self.heads_num = heads_num
        head_dim = hidden_size // heads_num

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6)
        self.img_attn_k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.txt_attn_q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6)
        self.txt_attn_k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (
        #     shift_msa,
        #     scale_msa,
        #     gate_msa,
        #     shift_mlp,
        #     scale_mlp,
        #     gate_mlp,
        # ) = self.img_mod(temb).chunk(6, dim=-1)
        # (
        #     c_shift_msa,
        #     c_scale_msa,
        #     c_gate_msa,
        #     c_shift_mlp,
        #     c_scale_mlp,
        #     c_gate_mlp,
        # ) = self.txt_mod(temb).chunk(6, dim=-1)

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # # Prepare image for attention.
        # img_modulated = self.norm1(hidden_states)
        # img_modulated = modulate(img_modulated, shift=shift_msa, scale=scale_msa)
        
        img_qkv = self.img_attn_qkv(norm_hidden_states)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        # txt_modulated = self.norm1_context(encoder_hidden_states)
        # txt_modulated = modulate(txt_modulated, shift=c_shift_msa, scale=c_scale_msa)
        
        txt_qkv = self.txt_attn_qkv(norm_encoder_hidden_states)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        attn = attention(q, k, v)

        img_attn, txt_attn = attn[:, : hidden_states.shape[1]], attn[:, hidden_states.shape[1] :]

        hidden_states = hidden_states + self.img_attn_proj(img_attn) * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.txt_attn_proj(txt_attn) * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformer3DModel(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference: [1] Flux.1: https://github.com/black-forest-labs/flux [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters ---------- args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        if sum(rope_dim_list) != attention_head_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {attention_head_dim}")

        # image projection
        self.img_in = PatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)

        # text projection
        self.txt_in = SingleTokenRefiner(text_states_dim, inner_dim, num_attention_heads, depth=2)

        # time modulation
        self.time_in = TimestepEmbedder(inner_dim, get_activation_layer("silu"))

        # text modulation
        self.vector_in = MLPEmbedder(text_states_dim_2, inner_dim)

        # guidance modulation
        self.guidance_in = TimestepEmbedder(inner_dim, get_activation_layer("silu")) if guidance_embed else None

        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    mlp_width_ratio=mlp_width_ratio,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # Prepare modulation vectors.
        temb = self.time_in(timestep)

        # text modulation
        temb = temb + self.vector_in(encoder_hidden_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")

            temb = temb + self.guidance_in(guidance)

        # Embed image and text.
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states, timestep, encoder_attention_mask)

        txt_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        for _, block in enumerate(self.transformer_blocks):
            double_block_args = [
                hidden_states,
                encoder_hidden_states,
                temb,
                freqs_cis,
            ]

            hidden_states, encoder_hidden_states = block(*double_block_args)

        hidden_states = torch.cat((hidden_states, encoder_hidden_states), dim=1)
        
        for block in self.single_transformer_blocks:
            single_block_args = [
                hidden_states,
                temb,
                txt_seq_len,
                (freqs_cos, freqs_sin),
            ]

            hidden_states = block(*single_block_args)

        hidden_states = hidden_states[:, :img_seq_len, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
