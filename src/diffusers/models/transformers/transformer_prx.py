# Copyright 2025 The Photoroom and The HuggingFace Teams. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.functional import fold, unfold

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import get_timestep_embedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)


def get_image_ids(batch_size: int, height: int, width: int, patch_size: int, device: torch.device) -> torch.Tensor:
    r"""
    Generates 2D patch coordinate indices for a batch of images.

    Args:
        batch_size (`int`):
            Number of images in the batch.
        height (`int`):
            Height of the input images (in pixels).
        width (`int`):
            Width of the input images (in pixels).
        patch_size (`int`):
            Size of the square patches that the image is divided into.
        device (`torch.device`):
            The device on which to create the tensor.

    Returns:
        `torch.Tensor`:
            Tensor of shape `(batch_size, num_patches, 2)` containing the (row, col) coordinates of each patch in the
            image grid.
    """

    img_ids = torch.zeros(height // patch_size, width // patch_size, 2, device=device)
    img_ids[..., 0] = torch.arange(height // patch_size, device=device)[:, None]
    img_ids[..., 1] = torch.arange(width // patch_size, device=device)[None, :]
    return img_ids.reshape((height // patch_size) * (width // patch_size), 2).unsqueeze(0).repeat(batch_size, 1, 1)


def apply_rope(xq: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    r"""
    Applies rotary positional embeddings (RoPE) to a query tensor.

    Args:
        xq (`torch.Tensor`):
            Input tensor of shape `(..., dim)` representing the queries.
        freqs_cis (`torch.Tensor`):
            Precomputed rotary frequency components of shape `(..., dim/2, 2)` containing cosine and sine pairs.

    Returns:
        `torch.Tensor`:
            Tensor of the same shape as `xq` with rotary embeddings applied.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    # Ensure freqs_cis is on the same device as queries to avoid device mismatches with offloading
    freqs_cis = freqs_cis.to(device=xq.device, dtype=xq_.dtype)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


class PRXAttnProcessor2_0:
    r"""
    Processor for implementing PRX-style attention with multi-source tokens and RoPE. Supports multiple attention
    backends (Flash Attention, Sage Attention, etc.) via dispatch_attention_fn.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("PRXAttnProcessor2_0 requires PyTorch 2.0, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: "PRXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply PRX attention using PRXAttention module.

        Args:
            attn: PRXAttention module containing projection layers
            hidden_states: Image tokens [B, L_img, D]
            encoder_hidden_states: Text tokens [B, L_txt, D]
            attention_mask: Boolean mask for text tokens [B, L_txt]
            image_rotary_emb: Rotary positional embeddings [B, 1, L_img, head_dim//2, 2, 2]
        """

        if encoder_hidden_states is None:
            raise ValueError("PRXAttnProcessor2_0 requires 'encoder_hidden_states' containing text tokens.")

        # Project image tokens to Q, K, V
        img_qkv = attn.img_qkv_proj(hidden_states)
        B, L_img, _ = img_qkv.shape
        img_qkv = img_qkv.reshape(B, L_img, 3, attn.heads, attn.head_dim)
        img_qkv = img_qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L_img, D]
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]

        # Apply QK normalization to image tokens
        img_q = attn.norm_q(img_q)
        img_k = attn.norm_k(img_k)

        # Project text tokens to K, V
        txt_kv = attn.txt_kv_proj(encoder_hidden_states)
        B, L_txt, _ = txt_kv.shape
        txt_kv = txt_kv.reshape(B, L_txt, 2, attn.heads, attn.head_dim)
        txt_kv = txt_kv.permute(2, 0, 3, 1, 4)  # [2, B, H, L_txt, D]
        txt_k, txt_v = txt_kv[0], txt_kv[1]

        # Apply K normalization to text tokens
        txt_k = attn.norm_added_k(txt_k)

        # Apply RoPE to image queries and keys
        if image_rotary_emb is not None:
            img_q = apply_rope(img_q, image_rotary_emb)
            img_k = apply_rope(img_k, image_rotary_emb)

        # Concatenate text and image keys/values
        k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt + L_img, D]
        v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt + L_img, D]

        # Build attention mask if provided
        attn_mask_tensor = None
        if attention_mask is not None:
            bs, _, l_img, _ = img_q.shape
            l_txt = txt_k.shape[2]

            if attention_mask.dim() != 2:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")
            if attention_mask.shape[-1] != l_txt:
                raise ValueError(f"attention_mask last dim {attention_mask.shape[-1]} must equal text length {l_txt}")

            device = img_q.device
            ones_img = torch.ones((bs, l_img), dtype=torch.bool, device=device)
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            joint_mask = torch.cat([attention_mask, ones_img], dim=-1)
            attn_mask_tensor = joint_mask[:, None, None, :].expand(-1, attn.heads, l_img, -1)

        # Apply attention using dispatch_attention_fn for backend support
        # Reshape to match dispatch_attention_fn expectations: [B, L, H, D]
        query = img_q.transpose(1, 2)  # [B, L_img, H, D]
        key = k.transpose(1, 2)  # [B, L_txt + L_img, H, D]
        value = v.transpose(1, 2)  # [B, L_txt + L_img, H, D]

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask_tensor,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape from [B, L_img, H, D] to [B, L_img, H*D]
        batch_size, seq_len, num_heads, head_dim = attn_output.shape
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)

        # Apply output projection
        attn_output = attn.to_out[0](attn_output)
        if len(attn.to_out) > 1:
            attn_output = attn.to_out[1](attn_output)  # dropout if present

        return attn_output


class PRXAttention(nn.Module, AttentionModuleMixin):
    r"""
    PRX-style attention module that handles multi-source tokens and RoPE. Similar to FluxAttention but adapted for
    PRX's architecture.
    """

    _default_processor_cls = PRXAttnProcessor2_0
    _available_processors = [PRXAttnProcessor2_0]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        out_bias: bool = False,
        eps: float = 1e-6,
        processor=None,
    ):
        super().__init__()

        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim

        self.img_qkv_proj = nn.Linear(query_dim, query_dim * 3, bias=bias)

        self.norm_q = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)

        self.txt_kv_proj = nn.Linear(query_dim, query_dim * 2, bias=bias)
        self.norm_added_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, query_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(0.0))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


# inspired from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class PRXEmbedND(nn.Module):
    r"""
    N-dimensional rotary positional embedding.

    This module creates rotary embeddings (RoPE) across multiple axes, where each axis can have its own embedding
    dimension. The embeddings are combined and returned as a single tensor

    Args:
        dim (int):
        Base embedding dimension (must be even).
        theta (int):
        Scaling factor that controls the frequency spectrum of the rotary embeddings.
        axes_dim (list[int]):
        List of embedding dimensions for each axis (each must be even).
    """

    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        out = pos.unsqueeze(-1) * omega.unsqueeze(0)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        # Native PyTorch equivalent of: Rearrange("b n d (i j) -> b n d i j", i=2, j=2)
        # out shape: (b, n, d, 4) -> reshape to (b, n, d, 2, 2)
        out = out.reshape(*out.shape[:-1], 2, 2)
        return out.float()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [self.rope(ids[:, :, i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    r"""
    A simple 2-layer MLP used for embedding inputs.

    Args:
        in_dim (`int`):
            Dimensionality of the input features.
        hidden_dim (`int`):
            Dimensionality of the hidden and output embedding space.

    Returns:
        `torch.Tensor`:
            Tensor of shape `(..., hidden_dim)` containing the embedded representations.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Modulation(nn.Module):
    r"""
    Modulation network that generates scale, shift, and gating parameters.

    Given an input vector, the module projects it through a linear layer to produce six chunks, which are grouped into
    two tuples `(shift, scale, gate)`.

    Args:
        dim (`int`):
            Dimensionality of the input vector. The output will have `6 * dim` features internally.

    Returns:
        ((`torch.Tensor`, `torch.Tensor`, `torch.Tensor`), (`torch.Tensor`, `torch.Tensor`, `torch.Tensor`)):
            Two tuples `(shift, scale, gate)`.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(
        self, vec: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)
        return tuple(out[:3]), tuple(out[3:])


class PRXBlock(nn.Module):
    r"""
    Multimodal transformer block with text–image cross-attention, modulation, and MLP.

    Args:
        hidden_size (`int`):
            Dimension of the hidden representations.
        num_heads (`int`):
            Number of attention heads.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Expansion ratio for the hidden dimension inside the MLP.
        qk_scale (`float`, *optional*):
            Scale factor for queries and keys. If not provided, defaults to ``head_dim**-0.5``.

    Attributes:
        img_pre_norm (`nn.LayerNorm`):
            Pre-normalization applied to image tokens before attention.
        attention (`PRXAttention`):
            Multi-head attention module with built-in QKV projections and normalizations for cross-attention between
            image and text tokens.
        post_attention_layernorm (`nn.LayerNorm`):
            Normalization applied after attention.
        gate_proj / up_proj / down_proj (`nn.Linear`):
            Feedforward layers forming the gated MLP.
        mlp_act (`nn.GELU`):
            Nonlinear activation used in the MLP.
        modulation (`Modulation`):
            Produces scale/shift/gating parameters for modulated layers.

        Methods:
            The forward method performs cross-attention and the MLP with modulation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.hidden_size = hidden_size

        # Pre-attention normalization for image tokens
        self.img_pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # PRXAttention module with built-in projections and norms
        self.attention = PRXAttention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=self.head_dim,
            bias=False,
            out_bias=False,
            eps=1e-6,
            processor=PRXAttnProcessor2_0(),
        )

        # mlp
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.mlp_act = nn.GELU(approximate="tanh")

        self.modulation = Modulation(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        r"""
        Runs modulation-gated cross-attention and MLP, with residual connections.

        Args:
            hidden_states (`torch.Tensor`):
                Image tokens of shape `(B, L_img, hidden_size)`.
            encoder_hidden_states (`torch.Tensor`):
                Text tokens of shape `(B, L_txt, hidden_size)`.
            temb (`torch.Tensor`):
                Conditioning vector used by `Modulation` to produce scale/shift/gates, shape `(B, hidden_size)` (or
                broadcastable).
            image_rotary_emb (`torch.Tensor`):
                Rotary positional embeddings applied inside attention.
            attention_mask (`torch.Tensor`, *optional*):
                Boolean mask for text tokens of shape `(B, L_txt)`, where `0` marks padding.
            **kwargs:
                Additional keyword arguments for API compatibility.

        Returns:
            `torch.Tensor`:
                Updated image tokens of shape `(B, L_img, hidden_size)`.
        """

        mod_attn, mod_mlp = self.modulation(temb)
        attn_shift, attn_scale, attn_gate = mod_attn
        mlp_shift, mlp_scale, mlp_gate = mod_mlp

        hidden_states_mod = (1 + attn_scale) * self.img_pre_norm(hidden_states) + attn_shift

        attn_out = self.attention(
            hidden_states=hidden_states_mod,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + attn_gate * attn_out

        x = (1 + mlp_scale) * self.post_attention_layernorm(hidden_states) + mlp_shift
        hidden_states = hidden_states + mlp_gate * (self.down_proj(self.mlp_act(self.gate_proj(x)) * self.up_proj(x)))
        return hidden_states


class FinalLayer(nn.Module):
    r"""
    Final projection layer with adaptive LayerNorm modulation.

    This layer applies a normalized and modulated transformation to input tokens and projects them into patch-level
    outputs.

    Args:
        hidden_size (`int`):
            Dimensionality of the input tokens.
        patch_size (`int`):
            Size of the square image patches.
        out_channels (`int`):
            Number of output channels per pixel (e.g. RGB = 3).

    Forward Inputs:
        x (`torch.Tensor`):
            Input tokens of shape `(B, L, hidden_size)`, where `L` is the number of patches.
        vec (`torch.Tensor`):
            Conditioning vector of shape `(B, hidden_size)` used to generate shift and scale parameters for adaptive
            LayerNorm.

    Returns:
        `torch.Tensor`:
            Projected patch outputs of shape `(B, L, patch_size * patch_size * out_channels)`.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def img2seq(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    r"""
    Flattens an image tensor into a sequence of non-overlapping patches.

    Args:
        img (`torch.Tensor`):
            Input image tensor of shape `(B, C, H, W)`.
        patch_size (`int`):
            Size of each square patch. Must evenly divide both `H` and `W`.

    Returns:
        `torch.Tensor`:
            Flattened patch sequence of shape `(B, L, C * patch_size * patch_size)`, where `L = (H // patch_size) * (W
            // patch_size)` is the number of patches.
    """
    return unfold(img, kernel_size=patch_size, stride=patch_size).transpose(1, 2)


def seq2img(seq: torch.Tensor, patch_size: int, shape: torch.Tensor) -> torch.Tensor:
    r"""
    Reconstructs an image tensor from a sequence of patches (inverse of `img2seq`).

    Args:
        seq (`torch.Tensor`):
            Patch sequence of shape `(B, L, C * patch_size * patch_size)`, where `L = (H // patch_size) * (W //
            patch_size)`.
        patch_size (`int`):
            Size of each square patch.
        shape (`tuple` or `torch.Tensor`):
            The original image spatial shape `(H, W)`. If a tensor is provided, the first two values are interpreted as
            height and width.

    Returns:
        `torch.Tensor`:
            Reconstructed image tensor of shape `(B, C, H, W)`.
    """
    if isinstance(shape, tuple):
        shape = shape[-2:]
    elif isinstance(shape, torch.Tensor):
        shape = (int(shape[0]), int(shape[1]))
    else:
        raise NotImplementedError(f"shape type {type(shape)} not supported")
    return fold(seq.transpose(1, 2), shape, kernel_size=patch_size, stride=patch_size)


class PRXTransformer2DModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    Transformer-based 2D model for text to image generation.

    Args:
        in_channels (`int`, *optional*, defaults to 16):
            Number of input channels in the latent image.
        patch_size (`int`, *optional*, defaults to 2):
            Size of the square patches used to flatten the input image.
        context_in_dim (`int`, *optional*, defaults to 2304):
            Dimensionality of the text conditioning input.
        hidden_size (`int`, *optional*, defaults to 1792):
            Dimension of the hidden representation.
        mlp_ratio (`float`, *optional*, defaults to 3.5):
            Expansion ratio for the hidden dimension inside MLP blocks.
        num_heads (`int`, *optional*, defaults to 28):
            Number of attention heads.
        depth (`int`, *optional*, defaults to 16):
            Number of transformer blocks.
        axes_dim (`list[int]`, *optional*):
            List of dimensions for each positional embedding axis. Defaults to `[32, 32]`.
        theta (`int`, *optional*, defaults to 10000):
            Frequency scaling factor for rotary embeddings.
        time_factor (`float`, *optional*, defaults to 1000.0):
            Scaling factor applied in timestep embeddings.
        time_max_period (`int`, *optional*, defaults to 10000):
            Maximum frequency period for timestep embeddings.

    Attributes:
        pe_embedder (`EmbedND`):
            Multi-axis rotary embedding generator for positional encodings.
        img_in (`nn.Linear`):
            Projection layer for image patch tokens.
        time_in (`MLPEmbedder`):
            Embedding layer for timestep embeddings.
        txt_in (`nn.Linear`):
            Projection layer for text conditioning.
        blocks (`nn.ModuleList`):
            Stack of transformer blocks (`PRXBlock`).
        final_layer (`LastLayer`):
            Projection layer mapping hidden tokens back to patch outputs.

    Methods:
        attn_processors:
            Returns a dictionary of all attention processors in the model.
        set_attn_processor(processor):
            Replaces attention processors across all attention layers.
        process_inputs(image_latent, txt):
            Converts inputs into patch tokens, encodes text, and produces positional encodings.
        compute_timestep_embedding(timestep, dtype):
            Creates a timestep embedding of dimension 256, scaled and projected.
        forward_transformers(image_latent, cross_attn_conditioning, timestep, time_embedding, attention_mask,
        **block_kwargs):
            Runs the sequence of transformer blocks over image and text tokens.
        forward(image_latent, timestep, cross_attn_conditioning, micro_conditioning, cross_attn_mask=None,
        attention_kwargs=None, return_dict=True):
            Full forward pass from latent input to reconstructed output image.

    Returns:
        `Transformer2DModelOutput` if `return_dict=True` (default), otherwise a tuple containing:
            - `sample` (`torch.Tensor`): Reconstructed image of shape `(B, C, H, W)`.
    """

    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        patch_size: int = 2,
        context_in_dim: int = 2304,
        hidden_size: int = 1792,
        mlp_ratio: float = 3.5,
        num_heads: int = 28,
        depth: int = 16,
        axes_dim: list = None,
        theta: int = 10000,
        time_factor: float = 1000.0,
        time_max_period: int = 10000,
    ):
        super().__init__()

        if axes_dim is None:
            axes_dim = [32, 32]

        # Store parameters directly
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.out_channels = self.in_channels * self.patch_size**2

        self.time_factor = time_factor
        self.time_max_period = time_max_period

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads

        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = PRXEmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.blocks = nn.ModuleList(
            [
                PRXBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer(self.hidden_size, 1, self.out_channels)

        self.gradient_checkpointing = False

    def _compute_timestep_embedding(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.time_in(
            get_timestep_embedding(
                timesteps=timestep,
                embedding_dim=256,
                max_period=self.time_max_period,
                scale=self.time_factor,
                flip_sin_to_cos=True,  # Match original cos, sin order
            ).to(dtype)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        r"""
        Forward pass of the PRXTransformer2DModel.

        The latent image is split into patch tokens, combined with text conditioning, and processed through a stack of
        transformer blocks modulated by the timestep. The output is reconstructed into the latent image space.

        Args:
            hidden_states (`torch.Tensor`):
                Input latent image tensor of shape `(B, C, H, W)`.
            timestep (`torch.Tensor`):
                Timestep tensor of shape `(B,)` or `(1,)`, used for temporal conditioning.
            encoder_hidden_states (`torch.Tensor`):
                Text conditioning tensor of shape `(B, L_txt, context_in_dim)`.
            attention_mask (`torch.Tensor`, *optional*):
                Boolean mask of shape `(B, L_txt)`, where `0` marks padding in the text sequence.
            attention_kwargs (`dict`, *optional*):
                Additional arguments passed to attention layers.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `Transformer2DModelOutput` or a tuple.

        Returns:
            `Transformer2DModelOutput` if `return_dict=True`, otherwise a tuple:

                - `sample` (`torch.Tensor`): Output latent image of shape `(B, C, H, W)`.
        """
        # Process text conditioning
        txt = self.txt_in(encoder_hidden_states)

        # Convert image to sequence and embed
        img = img2seq(hidden_states, self.patch_size)
        img = self.img_in(img)

        # Generate positional embeddings
        bs, _, h, w = hidden_states.shape
        img_ids = get_image_ids(bs, h, w, patch_size=self.patch_size, device=hidden_states.device)
        pe = self.pe_embedder(img_ids)

        # Compute time embedding
        vec = self._compute_timestep_embedding(timestep, dtype=img.dtype)

        # Apply transformer blocks
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img = self._gradient_checkpointing_func(
                    block.__call__,
                    img,
                    txt,
                    vec,
                    pe,
                    attention_mask,
                )
            else:
                img = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    temb=vec,
                    image_rotary_emb=pe,
                    attention_mask=attention_mask,
                )

        # Final layer and convert back to image
        img = self.final_layer(img, vec)
        output = seq2img(img, self.patch_size, hidden_states.shape)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
