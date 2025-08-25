# Copyright 2025 The MAGI Team and The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, is_kernels_available, logging, scale_lora_layers, unscale_lora_layers
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import FP32LayerNorm
from .modeling_common import FeedForward


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_qkv_projections(attn: "Magi1Attention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def range_mod_pytorch(x, c_mapping, gatings):
    """
    PyTorch implementation of range_mod_triton. # TODO: Ensure that this implementation is correct and matches the
    range_mod_triton implementation.

    Inputs:
        x: (s, b, h). Tensor of inputs embedding (images or latent representations of images) c_mapping: (s, b). Tensor
        of condition map gatings: (b, denoising_range_num, h). Tensor of condition embedding
    """
    s, b, h = x.shape

    # Flatten x and c_mapping to 2D for easier indexing
    x_flat = x.transpose(0, 1).flatten(0, 1)  # (s*b, h)
    c_mapping_flat = c_mapping.transpose(0, 1).flatten(0, 1)  # (s*b,)
    gatings_flat = gatings.flatten(0, 1)  # (b*denoising_range_num, h)

    # Use advanced indexing to select the appropriate gating for each row
    # c_mapping_flat contains indices into gatings_flat
    selected_gatings = gatings_flat[c_mapping_flat]  # (s*b, h)

    # Element-wise multiplication
    y_flat = x_flat * selected_gatings  # (s*b, h)

    # Reshape back to original dimensions
    y = y_flat.reshape(b, s, h).transpose(0, 1)  # (s, b, h)

    return y


if is_kernels_available():
    from kernels import use_kernel_forward_from_hub

    range_mod_pytorch = use_kernel_forward_from_hub("range_mod_triton")(range_mod_pytorch)


class Magi1AttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Magi1AttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: "Magi1Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = query.reshape(query.size(0), query.size(1), -1, attn.kv_inner_dim)
        key = key.reshape(key.size(0), key.size(1), -1, attn.kv_inner_dim)

        query = attn.norm_q(query)
        query = query.transpose(0, 1)
        key = attn.norm_k(key)
        key = key.transpose(0, 1)

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # batch_size, seq_len, num_heads, head_dim
        query = query.contiguous()
        key = key.contiguous()
        value = value.unflatten(2, (-1, attn.kv_inner_dim)).transpose(0, 1).contiguous()

        # Perform Grouped-Query Attention (GQA)
        kv_heads = attn.kv_heads if attn.kv_heads is not None else attn.heads
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
            backend=self._attention_backend,
        )
        # Convert [b, nh, sq, hd] -> [sq, b, (hn hd)]
        hidden_states = hidden_states.permute(2, 0, 1, 3)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class Magi1Attention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Magi1AttnProcessor
    _available_processors = [Magi1AttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 24,
        kv_heads: Optional[int] = None,
        dim_head: int = 128,
        eps: float = 1e-6,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = dim_head * self.kv_heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=False)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=False)
        # TODO: Verify here: 2*to_out =? linear_proj of magi1
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(2 * self.inner_dim, dim, bias=False),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = FP32LayerNorm(dim_head, eps)
        self.norm_k = FP32LayerNorm(dim_head, eps)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class Magi1ImageEmbedding(torch.nn.Module):
    """
    Image embedding layer for the MAGI-1 model.

    This module processes image conditioning features for image-to-video generation tasks. It applies layer
    normalization, a feed-forward transformation, and optional positional embeddings to prepare image features for
    cross-attention.

    Args:
        in_features (`int`): Input feature dimension.
        out_features (`int`): Output feature dimension.
        pos_embed_seq_len (`int`, optional): Sequence length for positional embeddings.
            If provided, learnable positional embeddings will be added to the input.
    """

    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        # if pos_embed_seq_len is not None:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        # else:
        #     self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        # if self.pos_embed is not None:
        #     batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
        #     encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
        #     encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class Magi1TextProjection(nn.Module):
    """
    Projects caption embeddings.
    """

    def __init__(self, in_features, hidden_size, adaln_dim):
        super().__init__()
        self.y_proj_xattn = nn.Sequential(nn.Linear(in_features, hidden_size), nn.SiLU())
        self.y_proj_adaln = nn.Linear(in_features, adaln_dim)

    def forward(self, caption):
        caption_xattn = self.y_proj_xattn(caption)
        caption_adaln = self.y_proj_adaln(caption)
        return caption_xattn, caption_adaln


class Magi1TimeTextImageEmbedding(nn.Module):
    """
    Combined time, text, and image embedding module for the MAGI-1 model.

    This module handles the encoding of three types of conditioning inputs:
    1. Timestep embeddings for diffusion process control
    2. Text embeddings for text-to-video generation
    3. Optional image embeddings for image-to-video generation

    Args:
        dim (`int`): Hidden dimension of the transformer model.
        time_freq_dim (`int`): Dimension for sinusoidal time embeddings.
        text_embed_dim (`int`): Input dimension of text embeddings.
        pos_embed_seq_len (`int`, optional): Sequence length for image positional embeddings.
        enable_distillation (`bool`, optional): Enable distillation timestep adjustments.
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        pos_embed_seq_len: Optional[int] = None,
        enable_distillation: bool = False,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=int(dim * 0.25))
        self.text_embedder = Magi1TextProjection(text_embed_dim, dim, adaln_dim=int(dim * 0.25))
        self.enable_distillation = enable_distillation

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        distill_interval: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = get_parameter_dtype(self.time_embedder)
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        y_xattn, y_adaln = self.text_embedder(encoder_hidden_states)

        # Apply distillation logic if enabled
        if self.enable_distillation and num_steps is not None:
            distill_dt_scalar = 2
            if num_steps == 12 and distill_interval is not None:
                base_chunk_step = 4
                distill_dt_factor = base_chunk_step / distill_interval * distill_dt_scalar
            else:
                distill_dt_factor = num_steps / 4 * distill_dt_scalar

            distill_dt = torch.ones_like(timestep) * distill_dt_factor
            distill_dt_embed = self.time_embedder(distill_dt)
            temb = temb + distill_dt_embed

        return temb, y_xattn, y_adaln


# TODO: Verify this.
class Magi1RotaryPosEmbed(nn.Module):
    """
    Rotary Position Embedding for MAGI-1 model.

    Args:
        dim (`int`): The embedding dimension.
        attention_head_dim (`int`): The attention head dimension.
        patch_size (`Tuple[int, int, int]`): Patch size for temporal, height, width dimensions.
        theta (`float`, *optional*, defaults to 10000.0): Base for the geometric progression.
    """

    def __init__(
        self,
        dim: int,
        attention_head_dim: int = 128,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        theta: float = 10000.0,
    ):
        super().__init__()

        self.dim = dim
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.theta = theta

        self.bands = nn.Parameter(self._get_default_bands())

    def _get_default_bands(self) -> torch.Tensor:
        num_bands = self.dim // 8

        exp = torch.arange(0, num_bands, dtype=torch.float32) / num_bands
        bands = 1.0 / (self.theta**exp)

        return bands

    def _build_3d_grid(
        self, feat_shape: List[int], ref_feat_shape: Optional[List[int]] = None, device: torch.device = None
    ) -> torch.Tensor:
        """Build 3D spatial grid matching original MAGI-1 implementation."""
        T, H, W = feat_shape

        t = torch.arange(T, device=device, dtype=torch.float32)
        h = torch.arange(H, device=device, dtype=torch.float32)
        w = torch.arange(W, device=device, dtype=torch.float32)

        # Center alignment for H/W dimensions
        h = h - (H - 1) / 2
        w = w - (W - 1) / 2

        # Handle reference shape rescaling (for fine-tuning)
        if ref_feat_shape is not None:
            ref_T, ref_H, ref_W = ref_feat_shape
            t_rescaled = []
            coords = [t, h, w]
            shapes = [T, H, W]
            ref_shapes = [ref_T, ref_H, ref_W]

            for coord, shape, ref_shape in zip(coords, shapes, ref_shapes):
                if shape == 1:
                    assert ref_shape == 1, "ref_feat_shape must be 1 when feat_shape is 1"
                    t_rescaled.append(coord)
                else:
                    t_rescaled.append(coord / (shape - 1) * (ref_shape - 1))
            t, h, w = t_rescaled

        # Create 3D meshgrid
        try:
            grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")
        except TypeError:
            # Fallback for older PyTorch versions
            grid_t, grid_h, grid_w = torch.meshgrid(t, h, w)

        # Stack into grid tensor: [T, H, W, 3]
        grid = torch.stack([grid_t, grid_h, grid_w], dim=-1)
        return grid

    def forward(self, hidden_states: torch.Tensor, ref_feat_shape: Optional[List[int]] = None) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        device = hidden_states.device
        feat_shape = [ppf, pph, ppw]

        # Build 3D spatial grid
        grid = self._build_3d_grid(feat_shape, ref_feat_shape, device)  # [T, H, W, 3]
        grid = grid.unsqueeze(-1)  # [T, H, W, 3, 1]

        # Apply frequency bands
        bands = self.bands.to(device)  # [num_bands]
        pos = grid * bands  # [T, H, W, 3, num_bands]

        # Compute sin/cos embeddings
        pos_sin = pos.sin()
        pos_cos = pos.cos()

        # Concatenate sin and cos
        embeddings = torch.cat([pos_sin, pos_cos], dim=-1)  # [T, H, W, 3, 2*num_bands]

        # Reshape to sequence format: [seq_len, embed_dim]
        seq_len = ppf * pph * ppw
        embed_dim = 3 * 2 * self.bands.shape[0]  # 3 spatial dims * 2 (sin/cos) * num_bands
        embeddings = embeddings.reshape(seq_len, embed_dim)

        return embeddings


class Magi1TransformerBlock(nn.Module):
    """
    A transformer block used in the MAGI-1 model.

    Args:
        dim (`int`): The number of channels in the input and output.
        ffn_dim (`int`): The number of channels in the feed-forward layer.
        num_heads (`int`): The number of attention heads.
        num_kv_heads (`int`): The number of key-value attention heads.
        eps (`float`): The epsilon value for layer normalization.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps)
        self.attn1 = Magi1Attention(
            dim=dim,
            heads=num_heads,
            kv_heads=num_kv_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=Magi1AttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = Magi1Attention(
            dim=dim,
            heads=num_heads,
            kv_heads=num_kv_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_kv_heads,
            processor=Magi1AttnProcessor(),
        )

        self.ada_modulate_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                int(dim * 0.25),
                int(dim * 2),
            ),
        )
        self.norm2 = FP32LayerNorm(dim, eps)
        self.norm3 = FP32LayerNorm(dim, eps)

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu", bias=False)

        self.norm4 = FP32LayerNorm(dim, eps)
        with torch.no_grad():
            self.norm2.weight.add_(1.0)
            self.norm4.weight.add_(1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        condition_map: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states.float()

        # [bs, sq, h] --> [sq, bs, q + qx + k + v]
        mixed_qqkv = self.norm1(hidden_states.transpose(0, 1))

        # 1. Self-attention
        self_attn_output = self.attn1(mixed_qqkv, None, None, rotary_emb)

        # 2. Cross-attention
        cross_attn_output = self.attn2(mixed_qqkv, encoder_hidden_states, None, None)

        hidden_states = torch.concat([self_attn_output, cross_attn_output], dim=2)

        gate_output = self.ada_modulate_layer(temb)
        # Softcap with 1.0
        gate_output = torch.tanh(gate_output.float()).to(gate_output.dtype)
        gate_msa, gate_mlp = gate_output.chunk(2, dim=-1)

        # Residual connection for self-attention
        original_dtype = hidden_states.dtype
        hidden_states = range_mod_pytorch(hidden_states.float(), condition_map, gate_msa)
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)

        # Residual connection for MLP
        original_dtype = hidden_states.dtype
        hidden_states = range_mod_pytorch(hidden_states.float(), condition_map, gate_mlp)
        hidden_states = self.norm4(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states.to(original_dtype)
        return hidden_states


class Magi1Transformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Magi1 model.

    This model implements a 3D transformer architecture for video generation with support for text conditioning and
    optional image conditioning. The model uses rotary position embeddings and adaptive layer normalization for
    temporal and spatial modeling.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `16`):
            The number of attention heads in each transformer block.
        attention_head_dim (`int`, defaults to `64`):
            The dimension of each attention head.
        in_channels (`int`, defaults to `16`):
            The number of input channels (from VAE latent space).
        out_channels (`int`, defaults to `16`):
            The number of output channels (to VAE latent space).
        cross_attention_dim (`int`, defaults to `4096`):
            The dimension of cross-attention (text encoder hidden size).
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `4096`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `34`):
            The number of transformer layers to use.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        image_embed_dim (`Optional[int]`, defaults to `None`):
            Dimension of image embeddings for image-to-video tasks.
        # pos_embed_seq_len (`Optional[int]`, defaults to `None`):
        #     Sequence length for positional embeddings in image conditioning.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "rope"]
    _no_split_modules = ["Magi1TransformerBlock", "norm_out"]
    _keep_in_fp32_modules = ["condition_embedder", "scale_shift_table", "norm_out", "norm_q", "norm_k"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["Magi1TransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_kv_heads: int = 8,
        in_channels: int = 16,
        out_channels: int = 16,
        cross_attention_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 12288,
        num_layers: int = 34,
        eps: float = 1e-6,
        # pos_embed_seq_len: Optional[int] = None,
        enable_distillation: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = Magi1RotaryPosEmbed(inner_dim // num_attention_heads, attention_head_dim, patch_size)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        # 2. Condition embeddings
        self.condition_embedder = Magi1TimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            text_embed_dim=cross_attention_dim,
            # pos_embed_seq_len=pos_embed_seq_len,
            enable_distillation=enable_distillation,
        )

        # 3. Transformer blocks
        # For image-to-video tasks, we may need additional projections
        # added_kv_proj_dim = image_embed_dim if image_embed_dim is not None else None

        self.blocks = nn.ModuleList(
            [
                Magi1TransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    num_kv_heads,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size), bias=False)

        self.gradient_checkpointing = False

    def _generate_condition_map(
        self, batch_size: int, seq_len: int, denoising_range_num: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate condition map that determines which condition to use for each token.

        Args:
            batch_size: Batch size
            seq_len: Sequence length (T*H*W after patching)
            denoising_range_num: Number of denoising ranges
            device: Device to create tensor on

        Returns:
            condition_map: (seq_len, batch_size) tensor mapping tokens to conditions
        """
        seqlen_per_chunk = seq_len // denoising_range_num
        condition_map = torch.arange(batch_size * denoising_range_num, device=device)
        condition_map = torch.repeat_interleave(condition_map, seqlen_per_chunk)

        # Handle remainder if seq_len is not perfectly divisible
        remainder = seq_len % denoising_range_num
        if remainder > 0:
            # Extend the last chunk indices for remaining tokens
            last_indices = torch.full((remainder,), batch_size * denoising_range_num - 1, device=device)
            condition_map = torch.cat([condition_map, last_indices])

        condition_map = condition_map.reshape(batch_size, -1).transpose(0, 1).contiguous()
        return condition_map

    def _apply_attention_mask(self, y_xattn: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mask to cross-attention inputs, flattening variable-length sequences.

        Args:
            y_xattn: Cross-attention embeddings (B, seq_len, dim)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            y_xattn_flat: Flattened valid tokens (total_valid_tokens, dim)
        """
        # Ensure mask is boolean
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        # Flatten and select valid tokens
        y_xattn_flat = torch.masked_select(y_xattn, attention_mask.unsqueeze(-1)).reshape(-1, y_xattn.shape[-1])

        return y_xattn_flat

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        condition_map: Optional[torch.Tensor] = None,
        denoising_range_num: Optional[int] = None,
        num_steps: Optional[int] = None,
        distill_interval: Optional[int] = None,
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

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, T*H*W, C)

        # Generate condition_map if not provided
        if condition_map is None:
            condition_map = self._generate_condition_map(
                batch_size,
                post_patch_num_frames * post_patch_height * post_patch_width,
                denoising_range_num or 1,
                hidden_states.device,
            )

        timestep_shape = timestep.shape
        temb, y_xattn, y_adaln, encoder_hidden_states_image = self.condition_embedder(
            timestep.flatten(),
            encoder_hidden_states,
            encoder_hidden_states_image,
            num_steps=num_steps,
            distill_interval=distill_interval,
        )

        temb = temb.reshape(*timestep_shape, -1) + y_adaln

        # Process cross-attention inputs with proper masking
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Handle attention masking for variable-length sequences
        if encoder_attention_mask is not None:
            # Apply masking to flatten variable-length sequences
            encoder_hidden_states = self._apply_attention_mask(encoder_hidden_states, encoder_attention_mask)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, rotary_emb, condition_map
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary_emb, condition_map)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify: convert from sequence back to video format
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1
        )

        # Rearrange patches: (B, T//p_t, H//p_h, W//p_w, C*p_t*p_h*p_w) -> (B, C, T, H, W)
        p_t, p_h, p_w = self.config.patch_size
        hidden_states = hidden_states.view(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            self.config.out_channels,
            p_t,
            p_h,
            p_w,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        output = hidden_states.contiguous().view(
            batch_size,
            self.config.out_channels,
            post_patch_num_frames * p_t,
            post_patch_height * p_h,
            post_patch_width * p_w,
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
