# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class MagiTransformerOutput(BaseOutput):
    """
    The output of [`MagiTransformer3DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch_size, num_channels, frames, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class MagiAttention(nn.Module):
    """
    A cross attention layer for MAGI-1.

    This implements the specialized attention mechanism from the MAGI-1 model, including query/key normalization and
    proper handling of rotary embeddings.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        # Projection layers
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        # Normalization layers for query and key - important part of MAGI-1's attention mechanism
        self.norm_q = nn.LayerNorm(dim_head, eps=1e-5)
        self.norm_k = nn.LayerNorm(dim_head, eps=1e-5)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_pos_emb=None,
        **cross_attention_kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # Project to query, key, value
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = query.reshape(batch_size, sequence_length, self.heads, self.dim_head)
        key = key.reshape(batch_size, -1, self.heads, self.dim_head)
        value = value.reshape(batch_size, -1, self.heads, self.dim_head)

        # Apply layer normalization to query and key (as in MAGI-1)
        # Convert to float32 for better numerical stability during normalization
        orig_dtype = query.dtype
        query = self.norm_q(query.float()).to(orig_dtype)
        key = self.norm_k(key.float()).to(orig_dtype)

        # Transpose for attention
        # [batch_size, seq_len, heads, dim_head] -> [batch_size, heads, seq_len, dim_head]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply rotary position embeddings if provided
        if rotary_pos_emb is not None:
            # Apply rotary embeddings using the same method as in MAGI-1
            def apply_rotary_emb(hidden_states, freqs):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                # Convert to complex numbers
                x_complex = torch.view_as_complex(hidden_states.to(dtype).unflatten(-1, (-1, 2)))
                # Apply rotation in complex space
                x_rotated = x_complex * freqs
                # Convert back to real
                x_out = torch.view_as_real(x_rotated).flatten(-2)
                return x_out.type_as(hidden_states)

            # Apply rotary embeddings
            query = apply_rotary_emb(query, rotary_pos_emb)
            key = apply_rotary_emb(key, rotary_pos_emb)

        # Use scaled_dot_product_attention if available (PyTorch 2.0+)
        if hasattr(F, "scaled_dot_product_attention"):
            # Apply scaled dot product attention
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            # [batch_size, heads, seq_len, dim_head] -> [batch_size, seq_len, heads*dim_head]
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        else:
            # Manual implementation of attention
            # Reshape for bmm
            query = query.reshape(batch_size * self.heads, sequence_length, self.dim_head)
            key = key.reshape(batch_size * self.heads, -1, self.dim_head)
            value = value.reshape(batch_size * self.heads, -1, self.dim_head)

            # Compute attention scores
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            attention_scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            # Compute output
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = hidden_states.reshape(batch_size, self.heads, sequence_length, self.dim_head)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1)

        # Project to output
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class MagiTransformerBlock(nn.Module):
    """
    A transformer block for MAGI-1.

    This is a simplified version of the MAGI-1 transformer block.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "gelu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.attn1 = MagiAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        # Cross-attention
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
            self.attn2 = MagiAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        # Choose activation function
        if activation_fn == "gelu":
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout) if final_dropout else nn.Identity(),
                nn.Linear(dim * 4, dim),
            )
        elif activation_fn == "gelu-approximate":
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout) if final_dropout else nn.Identity(),
                nn.Linear(dim * 4, dim),
            )
        elif activation_fn == "silu":
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout) if final_dropout else nn.Identity(),
                nn.Linear(dim * 4, dim),
            )
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.final_dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        rotary_pos_emb=None,
        **kwargs,
    ):
        # Self-attention
        norm_hidden_states = self.norm1(hidden_states)

        if self.only_cross_attention:
            hidden_states = hidden_states + self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
            )
        else:
            hidden_states = hidden_states + self.attn1(
                norm_hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
            )

        # Cross-attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states + self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                **(cross_attention_kwargs if cross_attention_kwargs is not None else {}),
            )

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + self.final_dropout(ff_output)

        return hidden_states


class LearnableRotaryEmbedding(nn.Module):
    """
    Learnable rotary position embeddings similar to the one used in MAGI-1.

    This implementation is based on MAGI-1's LearnableRotaryEmbeddingCat class, which creates rotary embeddings for 3D
    data (frames, height, width).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands

        # Initialize frequency bands
        self.register_buffer("freqs", self._get_default_bands())

    def _get_default_bands(self):
        """Generate default frequency bands"""
        if self.linear_bands:
            # Linear spacing
            bands = torch.linspace(1.0, self.max_seq_len / 2, self.dim // 2, dtype=torch.float32)
        else:
            # Log spacing (as in original RoPE)
            bands = 1.0 / (self.temperature ** (torch.arange(0, self.dim // 2, dtype=torch.float32) / (self.dim // 2)))

        return bands * torch.pi

    def get_embed(self, shape: List[int]) -> torch.Tensor:
        """
        Generate rotary position embeddings for the given shape.

        Args:
            shape: List of dimensions [frames, height, width]

        Returns:
            Rotary position embeddings (sin and cos components)
        """
        frames, height, width = shape
        seq_len = frames * height * width

        # Generate position indices
        if self.in_pixels:
            # Normalize positions to [-1, 1]
            t = torch.linspace(-1.0, 1.0, steps=frames, device=self.freqs.device)
            h = torch.linspace(-1.0, 1.0, steps=height, device=self.freqs.device)
            w = torch.linspace(-1.0, 1.0, steps=width, device=self.freqs.device)
        else:
            # Use integer positions
            t = torch.arange(frames, device=self.freqs.device, dtype=torch.float32)
            h = torch.arange(height, device=self.freqs.device, dtype=torch.float32)
            w = torch.arange(width, device=self.freqs.device, dtype=torch.float32)

            # Center spatial dimensions (as in MAGI-1)
            h = h - (height - 1) / 2
            w = w - (width - 1) / 2

        # Create position grid
        grid = torch.stack(torch.meshgrid(t, h, w, indexing="ij"), dim=-1)
        grid = grid.reshape(-1, 3)  # [seq_len, 3]

        # Get frequency bands
        freqs = self.freqs.to(grid.device)

        # Compute embeddings for each dimension
        # Temporal dimension
        t_emb = torch.outer(grid[:, 0], freqs[: self.dim // 6])
        t_sin = torch.sin(t_emb)
        t_cos = torch.cos(t_emb)

        # Height dimension
        h_emb = torch.outer(grid[:, 1], freqs[: self.dim // 6])
        h_sin = torch.sin(h_emb)
        h_cos = torch.cos(h_emb)

        # Width dimension
        w_emb = torch.outer(grid[:, 2], freqs[: self.dim // 6])
        w_sin = torch.sin(w_emb)
        w_cos = torch.cos(w_emb)

        # Concatenate all embeddings
        sin_emb = torch.cat([t_sin, h_sin, w_sin], dim=-1)
        cos_emb = torch.cat([t_cos, h_cos, w_cos], dim=-1)

        # Pad or trim to match expected dimension
        target_dim = self.dim // 2
        if sin_emb.shape[1] < target_dim:
            pad_size = target_dim - sin_emb.shape[1]
            sin_emb = F.pad(sin_emb, (0, pad_size))
            cos_emb = F.pad(cos_emb, (0, pad_size))
        elif sin_emb.shape[1] > target_dim:
            sin_emb = sin_emb[:, :target_dim]
            cos_emb = cos_emb[:, :target_dim]

        # Combine sin and cos for rotary embeddings
        return torch.cat([cos_emb.unsqueeze(-1), sin_emb.unsqueeze(-1)], dim=-1).reshape(seq_len, target_dim, 2)


class MagiTransformer3DModel(ModelMixin, ConfigMixin):
    """
    Transformer model for MAGI-1.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods implemented for
    all models (downloading, saving, loading, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        num_layers (`int`, *optional*, defaults to 24): Number of transformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 16): Number of attention heads.
        attention_head_dim (`int`, *optional*, defaults to 64): Dimension of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 1280): Dimension of cross-attention conditioning.
        activation_fn (`str`, *optional*, defaults to `"gelu"`): Activation function.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`): Type of normalization.
        norm_eps (`float`, *optional*, defaults to 1e-5): Epsilon for normalization.
        attention_bias (`bool`, *optional*, defaults to `False`): Whether to use bias in attention.
        num_embeds_ada_norm (`int`, *optional*, defaults to `None`): Number of embeddings for AdaLayerNorm.
        only_cross_attention (`bool`, *optional*, defaults to `False`): Whether to only use cross-attention.
        upcast_attention (`bool`, *optional*, defaults to `False`): Whether to upcast attention operations.
        dropout (`float`, *optional*, defaults to 0.0): Dropout probability.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1280,
        activation_fn: str = "gelu",
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        attention_bias: bool = False,
        num_embeds_ada_norm: Optional[int] = None,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        dropout: float = 0.0,
        patch_size: Tuple[int, int, int] = (1, 1, 1),
        max_seq_len: int = 1024,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        # Input embedding
        self.in_channels = in_channels
        time_embed_dim = attention_head_dim * num_attention_heads
        self.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(time_embed_dim, time_embed_dim)

        # Input projection
        self.input_proj = nn.Conv3d(in_channels, time_embed_dim, kernel_size=patch_size, stride=patch_size)

        # Rotary position embeddings
        self.rotary_embedding = LearnableRotaryEmbedding(
            dim=attention_head_dim,
            max_seq_len=max_seq_len,
            temperature=10000.0,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MagiTransformerBlock(
                    dim=time_embed_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.out_channels = out_channels
        self.output_proj = nn.Conv3d(time_embed_dim, out_channels, kernel_size=1)

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        """
        Enable sliced attention computation.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        logger.warning(
            "Calling `set_attention_slice` is deprecated and will be removed in a future version. Use"
            " `set_attention_processor` instead."
        )

        # Not implemented for MAGI-1 yet
        pass

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MagiTransformerBlock):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[MagiTransformerOutput, Tuple]:
        """
        Forward pass of the model.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of shape `(batch_size, in_channels, frames, height, width)`.
            timesteps (`torch.Tensor`, *optional*):
                Timesteps tensor of shape `(batch_size,)`.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Encoder hidden states for cross-attention.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for cross-attention.
            cross_attention_kwargs (`dict`, *optional*):
                Additional arguments for cross-attention.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dictionary.

        Returns:
            `MagiTransformerOutput` or `tuple`:
                If `return_dict` is `True`, a `MagiTransformerOutput` is returned, otherwise a tuple `(sample,)` is
                returned where `sample` is the output tensor.
        """
        # 1. Input processing
        batch_size, channels, frames, height, width = hidden_states.shape
        residual = hidden_states

        # 2. Time embedding
        if timesteps is not None:
            timesteps = timesteps.to(hidden_states.device)
            time_embeds = self.time_proj(timesteps)
            time_embeds = self.time_embedding(time_embeds)
        else:
            time_embeds = None

        # 3. Project input
        hidden_states = self.input_proj(hidden_states)

        # Get patched dimensions
        p_t, p_h, p_w = self.patch_size
        patched_frames = frames // p_t
        patched_height = height // p_h
        patched_width = width // p_w

        # 4. Reshape for transformer blocks
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)  # [B, C, F, H, W] -> [B, F, H, W, C]
        hidden_states = hidden_states.reshape(
            batch_size, patched_frames * patched_height * patched_width, -1
        )  # [B, F*H*W, C]

        # 5. Add time embeddings if provided
        if time_embeds is not None:
            time_embeds = time_embeds.unsqueeze(1)  # [B, 1, C]
            hidden_states = hidden_states + time_embeds

        # 6. Generate rotary position embeddings
        rotary_pos_emb = self.rotary_embedding.get_embed([patched_frames, patched_height, patched_width])
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)

        # Convert to complex representation for the attention mechanism
        # This matches MAGI-1's approach to applying rotary embeddings
        cos_emb = rotary_pos_emb[..., 0]
        sin_emb = rotary_pos_emb[..., 1]
        rotary_pos_emb = torch.complex(cos_emb, sin_emb).unsqueeze(0)  # [1, seq_len, dim//2]

        # 7. Process with transformer blocks
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timesteps,
                    attention_mask,
                    None,  # cross_attention_kwargs
                    rotary_pos_emb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    rotary_pos_emb=rotary_pos_emb,
                )

        # 8. Reshape back to video format
        hidden_states = hidden_states.reshape(batch_size, patched_frames, patched_height, patched_width, -1)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3)  # [B, F, H, W, C] -> [B, C, F, H, W]

        # 9. Project output
        hidden_states = self.output_proj(hidden_states)

        # 10. Add residual connection
        hidden_states = hidden_states + residual

        if not return_dict:
            return (hidden_states,)

        return MagiTransformerOutput(sample=hidden_states)
