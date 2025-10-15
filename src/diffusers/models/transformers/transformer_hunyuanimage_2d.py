# Copyright 2025 Tencent Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention_processor import Attention
from ..embeddings import PatchEmbed, TimestepEmbedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)


def get_activation_layer(act_type: str):
    """Get activation layer by name."""
    if act_type == "gelu":
        return nn.GELU
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_norm_layer(norm_type: str):
    """Get normalization layer by name."""
    if norm_type == "layer":
        return nn.LayerNorm
    elif norm_type == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_type} is not implemented")


def modulate(x: torch.Tensor, shift: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None):
    """Apply modulation with shift and scale."""
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x: torch.Tensor, gate: Optional[torch.Tensor] = None):
    """Apply gating."""
    if gate is None:
        return x
    return x * gate.unsqueeze(1)


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: nn.Module,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, dtype=dtype, device=device)
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


class MLP(nn.Module):
    """MLP layer with GELU activation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        act_layer: nn.Module,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias, dtype=dtype, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LinearWarpforSingle(nn.Module):
    """Linear layer wrapper for single stream blocks."""

    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = False, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_cat = torch.cat([x.contiguous(), y.contiguous()], dim=2).contiguous()
        return self.fc(input_cat)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        xq: Query tensor of shape [B, L, H, D]
        xk: Key tensor of shape [B, L, H, D]
        freqs_cis: Frequency tensor (cos, sin) or complex
    
    Returns:
        Tuple of rotated query and key tensors
    """
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
        # Reshape for broadcasting
        cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
        sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
        
        # Rotate half
        def rotate_half(x):
            x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
            return torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # Complex rotation
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    
    return xq_out, xk_out


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal DiT block with separate modulation for text and image/video.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # Image stream components
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), dtype=dtype, device=device)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.img_attn_k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, dtype=dtype, device=device)

        # Text stream components
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), dtype=dtype, device=device)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)
        self.txt_attn_q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.txt_attn_k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, dtype=dtype, device=device)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, dtype=dtype, device=device)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract modulation parameters
        (img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(vec).chunk(6, dim=-1)
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(vec).chunk(6, dim=-1)

        # Process image stream
        img_modulated = modulate(self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale)
        img_q = rearrange(self.img_attn_q(img_modulated), "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(self.img_attn_k(img_modulated), "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(self.img_attn_v(img_modulated), "B L (H D) -> B L H D", H=self.heads_num)

        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if provided
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)

        # Process text stream
        txt_modulated = modulate(self.txt_norm1(txt), shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = rearrange(self.txt_attn_q(txt_modulated), "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(self.txt_attn_k(txt_modulated), "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(self.txt_attn_v(txt_modulated), "B L (H D) -> B L H D", H=self.heads_num)

        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Cross-modal attention
        q = torch.cat([img_q, txt_q], dim=1)
        k = torch.cat([img_k, txt_k], dim=1)
        v = torch.cat([img_v, txt_v], dim=1)

        # Use scaled dot product attention
        attn = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

        # Split attention outputs
        img_attn, txt_attn = attn[:, :img_q.shape[1]], attn[:, img_q.shape[1]:]
        img_attn = rearrange(img_attn, "B L H D -> B L (H D)")
        txt_attn = rearrange(txt_attn, "B L H D -> B L (H D)")

        # Apply projections and residuals
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate)

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate)

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers for multimodal processing.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # Parallel linear layers
        self.linear1_q = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, dtype=dtype, device=device)

        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device) if qk_norm else nn.Identity()

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Extract modulation parameters
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        # Compute Q, K, V, and MLP input
        q = rearrange(self.linear1_q(x_mod), "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(self.linear1_k(x_mod), "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(self.linear1_v(x_mod), "B L (H D) -> B L H D", H=self.heads_num)
        mlp = self.linear1_mlp(x_mod)

        # Apply QK-Norm
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Split into image and text sequences
        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
        img_v, txt_v = v[:, :-txt_len], v[:, -txt_len:]

        # Apply RoPE to image sequence
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)

        # Concatenate back
        q = torch.cat([img_q, txt_q], dim=1)
        k = torch.cat([img_k, txt_k], dim=1)
        v = torch.cat([img_v, txt_v], dim=1)

        # Attention
        attn = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        attn = rearrange(attn, "B L H D -> B L (H D)")

        # Combine with MLP
        output = self.linear2(attn, self.mlp_act(mlp))
        return x + apply_gate(output, gate=mod_gate)


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self, 
        hidden_size: int, 
        patch_size: Union[int, List[int]], 
        out_channels: int, 
        dtype: Optional[torch.dtype] = None, 
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        
        if isinstance(patch_size, int):
            out_size = patch_size * patch_size * out_channels
        else:
            out_size = (patch_size[0] * patch_size[1] if len(patch_size) == 2 else patch_size[0] * patch_size[1] * patch_size[2]) * out_channels
        
        self.linear = nn.Linear(hidden_size, out_size, bias=True, dtype=dtype, device=device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            get_activation_layer("silu")(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class HunyuanImage2DModel(ModelMixin, ConfigMixin):
    """
    HunyuanImage 2.1 Transformer model for text-to-image generation.
    
    This model uses a dual-stream architecture with both double-stream and single-stream blocks,
    supporting 2K image generation with ByT5 glyph-aware text encoding.
    
    Parameters:
        patch_size (`List[int]`, *optional*, defaults to `[1, 1]`):
            The size of the patches to use in the patch embedding layer.
        in_channels (`int`, *optional*, defaults to 64):
            The number of input channels (latent channels from VAE).
        out_channels (`int`, *optional*, defaults to 64):
            The number of output channels.
        hidden_size (`int`, *optional*, defaults to 3584):
            The hidden size of the transformer blocks.
        heads_num (`int`, *optional*, defaults to 28):
            The number of attention heads.
        mlp_width_ratio (`float`, *optional*, defaults to 4.0):
            The expansion ratio for MLP layers.
        mlp_act_type (`str`, *optional*, defaults to `"gelu_tanh"`):
            The activation function to use in MLP layers.
        mm_double_blocks_depth (`int`, *optional*, defaults to 20):
            The number of double-stream transformer blocks.
        mm_single_blocks_depth (`int`, *optional*, defaults to 40):
            The number of single-stream transformer blocks.
        rope_dim_list (`List[int]`, *optional*, defaults to `[64, 64]`):
            The dimensions for rotary position embeddings per axis.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in QKV projections.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use QK normalization.
        qk_norm_type (`str`, *optional*, defaults to `"rms"`):
            The type of QK normalization.
        guidance_embed (`bool`, *optional*, defaults to `False`):
            Whether to use guidance embedding (for distilled models).
        text_states_dim (`int`, *optional*, defaults to 3584):
            The dimension of text encoder outputs.
        rope_theta (`int`, *optional*, defaults to 256):
            The theta value for RoPE.
        use_meanflow (`bool`, *optional*, defaults to `False`):
            Whether to use MeanFlow (for distilled models).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: List[int] = [1, 1],
        in_channels: int = 64,
        out_channels: int = 64,
        hidden_size: int = 3584,
        heads_num: int = 28,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [64, 64],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        text_states_dim: int = 3584,
        rope_theta: int = 256,
        use_meanflow: bool = False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_meanflow = use_meanflow

        # Patch embedding
        if len(patch_size) == 2:
            self.pos_embed = PatchEmbed(
                height=None,
                width=None,
                patch_size=patch_size[0],
                in_channels=in_channels,
                embed_dim=hidden_size,
                interpolation_scale=None,
                pos_embed_type=None,
            )
        else:
            raise ValueError(f"Unsupported patch_size: {patch_size}")

        # Text embedding
        self.text_embedder = nn.Sequential(
            nn.Linear(text_states_dim, hidden_size),
            get_activation_layer("silu")(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Time embedding
        self.time_embedder = TimestepEmbedding(hidden_size, hidden_size, act_fn="silu")

        # MeanFlow support
        if use_meanflow:
            self.time_r_embedder = TimestepEmbedding(hidden_size, hidden_size, act_fn="silu")

        # Guidance embedding
        if guidance_embed:
            self.guidance_embedder = TimestepEmbedding(hidden_size, hidden_size, act_fn="silu")

        # Double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
            )
            for _ in range(mm_double_blocks_depth)
        ])

        # Single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
            )
            for _ in range(mm_single_blocks_depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

    def get_rotary_pos_embed(self, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary position embeddings.
        
        Args:
            height: Height in patches
            width: Width in patches
            
        Returns:
            Tuple of (cos, sin) frequency tensors
        """
        from ..embeddings import get_2d_rotary_pos_embed
        
        head_dim = self.hidden_size // self.heads_num
        return get_2d_rotary_pos_embed(
            self.rope_dim_list,
            (height, width),
            theta=self.rope_theta,
        )

    def unpatchify(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Unpatchify the output tensor.
        
        Args:
            x: Tensor of shape (B, H*W, patch_size**2 * C)
            height: Height in patches
            width: Width in patches
            
        Returns:
            Tensor of shape (B, C, H, W)
        """
        c = self.out_channels
        ph, pw = self.patch_size
        
        x = x.reshape(x.shape[0], height, width, c, ph, pw)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(x.shape[0], c, height * ph, width * pw)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        return_dict: bool = True,
        guidance: Optional[torch.Tensor] = None,
        timestep_r: Optional[torch.LongTensor] = None,
    ) -> Union[Transformer2DModelOutput, Tuple]:
        """
        Forward pass of the model.
        
        Args:
            hidden_states: Latent image tensor of shape (B, C, H, W)
            timestep: Timestep tensor
            encoder_hidden_states: Text embeddings
            encoder_attention_mask: Attention mask for text
            return_dict: Whether to return a dict
            guidance: Guidance scale for distilled models
            timestep_r: Second timestep for MeanFlow
            
        Returns:
            Transformer2DModelOutput or tuple
        """
        batch_size = hidden_states.shape[0]
        height, width = hidden_states.shape[2], hidden_states.shape[3]
        
        # Patch embed
        hidden_states = self.pos_embed(hidden_states)
        
        # Get sequence lengths
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]
        
        # Time embedding
        vec = self.time_embedder(timestep)
        
        # MeanFlow support
        if self.use_meanflow and timestep_r is not None:
            vec_r = self.time_r_embedder(timestep_r)
            vec = (vec + vec_r) / 2
        
        # Guidance embedding
        if self.guidance_embed:
            if guidance is None:
                guidance = torch.full((batch_size,), 1000.0, device=hidden_states.device, dtype=hidden_states.dtype)
            vec = vec + self.guidance_embedder(guidance)
        
        # Text embedding
        txt = self.text_embedder(encoder_hidden_states)
        
        # Get RoPE embeddings
        freqs_cis = self.get_rotary_pos_embed(height // self.patch_size[0], width // self.patch_size[1])
        
        # Double stream blocks
        for block in self.double_blocks:
            hidden_states, txt = block(hidden_states, txt, vec, freqs_cis=freqs_cis, text_mask=encoder_attention_mask)
        
        # Concatenate for single stream
        x = torch.cat([hidden_states, txt], dim=1)
        
        # Single stream blocks
        for block in self.single_blocks:
            x = block(x, vec, txt_seq_len, freqs_cis=freqs_cis, text_mask=encoder_attention_mask)
        
        # Extract image tokens
        hidden_states = x[:, :img_seq_len]
        
        # Final layer
        hidden_states = self.final_layer(hidden_states, vec)
        
        # Unpatchify
        output = self.unpatchify(hidden_states, height // self.patch_size[0], width // self.patch_size[1])
        
        if not return_dict:
            return (output,)
        
        return Transformer2DModelOutput(sample=output)
