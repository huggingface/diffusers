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
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import FeedForward
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Magi1AttnProcessor2_0:
    r"""
    Processor for implementing MAGI-1 attention mechanism.

    This processor handles both self-attention and cross-attention for the MAGI-1 model, following diffusers' standard
    attention processor interface. It supports image conditioning for image-to-video generation tasks.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Magi1AttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Handle image conditioning if present for cross-attention
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            # Extract image conditioning from the concatenated encoder states
            # The text encoder context length is typically 512 tokens
            text_context_length = getattr(attn, "text_context_length", 512)
            image_context_length = encoder_hidden_states.shape[1] - text_context_length
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # For self-attention, use hidden_states as encoder_hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Standard attention computation
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply normalization if available
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape for multi-head attention
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Apply rotary embeddings if provided
        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        # Handle image conditioning (I2V task) for cross-attention
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            attn_output_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            attn_output_img = attn_output_img.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states + attn_output_img

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


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
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class CaptionEmbedder(nn.Module):
    """
    Embeds caption text into vector representations for cross-attention and AdaLN.
    """

    def __init__(self, caption_channels: int, hidden_size: int):
        super().__init__()
        self.y_proj_xattn = nn.Sequential(nn.Linear(caption_channels, hidden_size), nn.SiLU())
        self.y_proj_adaln = nn.Linear(caption_channels, int(hidden_size * 0.25))

    def forward(self, caption):
        caption_xattn = self.y_proj_xattn(caption)
        caption_adaln = self.y_proj_adaln(caption)
        return caption_xattn, caption_adaln


class Magi1TimeTextCaptionEmbedding(nn.Module):
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
        image_embed_dim (`int`, optional): Input dimension of image embeddings.
        pos_embed_seq_len (`int`, optional): Sequence length for image positional embeddings.
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
        caption_channels: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.caption_embedder = CaptionEmbedder(caption_channels=caption_channels, hidden_size=dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = Magi1ImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        y_xattn, y_adaln = self.caption_embedder(encoder_hidden_states)

        # Combine time and text embeddings for AdaLN
        timestep_proj = y_adaln + temb

        encoder_hidden_states = self.text_embedder(y_xattn)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class Magi1RotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=freqs_dtype
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
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
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class Magi1TransformerBlock(nn.Module):
    """
    A transformer block used in the MAGI-1 model.

    This block follows diffusers' design philosophy with separate self-attention (attn1) and cross-attention (attn2)
    modules, while faithfully implementing the original MAGI-1 logic through appropriate parameter mapping during
    conversion.

    Args:
        dim (`int`): The number of channels in the input and output.
        ffn_dim (`int`): The number of channels in the feed-forward layer.
        num_heads (`int`): The number of attention heads.
        qk_norm (`str`): The type of normalization to apply to query and key projections.
        cross_attn_norm (`bool`): Whether to apply normalization in cross-attention.
        eps (`float`): The epsilon value for layer normalization.
        added_kv_proj_dim (`Optional[int]`): Additional key-value projection dimension for image conditioning.
    """

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
            processor=Magi1AttnProcessor2_0(),
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
            processor=Magi1AttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # Scale and shift table for AdaLN - 6 components for gating
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
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


class Magi1Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
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
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`Optional[str]`, defaults to `"rms_norm_across_heads"`):
            Type of query/key normalization to use.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        use_linear_projection (`bool`, defaults to `True`):
            Whether to use linear projection for patch embedding.
        upcast_attention (`bool`, defaults to `False`):
            Whether to upcast attention computation to float32.
        image_embed_dim (`Optional[int]`, defaults to `None`):
            Dimension of image embeddings for image-to-video tasks.
        rope_max_seq_len (`int`, defaults to `1024`):
            Maximum sequence length for rotary position embeddings.
        pos_embed_seq_len (`Optional[int]`, defaults to `None`):
            Sequence length for positional embeddings in image conditioning.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "rope"]
    _no_split_modules = ["Magi1TransformerBlock"]
    _keep_in_fp32_modules = ["condition_embedder", "scale_shift_table", "norm_out"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["Magi1TransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int = 16,
        cross_attention_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 12288,
        num_layers: int = 34,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        use_linear_projection: bool = True,
        upcast_attention: bool = False,
        image_embed_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        caption_channels: Optional[int] = None,
        caption_max_length: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        out_channels = out_channels or in_channels

        # Validate configuration
        if inner_dim != num_attention_heads * attention_head_dim:
            raise ValueError(
                f"inner_dim ({inner_dim}) should be equal to num_attention_heads ({num_attention_heads}) * "
                f"attention_head_dim ({attention_head_dim})"
            )

        if any(p <= 0 for p in patch_size):
            raise ValueError(f"All patch_size values must be positive, got {patch_size}")

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        if freq_dim <= 0:
            raise ValueError(f"freq_dim must be positive, got {freq_dim}")

        if image_embed_dim is not None and image_embed_dim <= 0:
            raise ValueError(f"image_embed_dim must be positive when provided, got {image_embed_dim}")

        # 1. Patch & position embedding
        self.rope = Magi1RotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        if use_linear_projection:
            self.patch_embedding = nn.Linear(in_channels * math.prod(patch_size), inner_dim)
        else:
            self.patch_embedding = nn.Conv3d(
                in_channels, inner_dim, kernel_size=patch_size, stride=patch_size, bias=False
            )

        # 2. Condition embeddings
        self.condition_embedder = Magi1TimeTextCaptionEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            #time_proj_dim=inner_dim * 6,
            text_embed_dim=cross_attention_dim,
            image_embed_dim=image_embed_dim,
            pos_embed_seq_len=pos_embed_seq_len,
            caption_channels=caption_channels,
            caption_max_length=caption_max_length,
        )

        # 3. Transformer blocks
        # For image-to-video tasks, we may need additional projections
        added_kv_proj_dim = image_embed_dim if image_embed_dim is not None else None

        self.blocks = nn.ModuleList(
            [
                Magi1TransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size), bias=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
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

        # Patch embedding - handle both conv3d and linear projection
        if self.config.use_linear_projection:
            # For linear projection, we need to patchify first
            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.config.patch_size

            # Patchify: (B, C, T, H, W) -> (B, T//p_t, H//p_h, W//p_w, C*p_t*p_h*p_w)
            hidden_states = hidden_states.unfold(2, p_t, p_t).unfold(3, p_h, p_h).unfold(4, p_w, p_w)
            hidden_states = hidden_states.contiguous().view(
                batch_size, num_frames // p_t, height // p_h, width // p_w, num_channels * p_t * p_h * p_w
            )
            # Reshape to sequence: (B, T*H*W, C*p_t*p_h*p_w)
            hidden_states = hidden_states.flatten(1, 3)
            # Apply linear projection: (B, T*H*W, inner_dim)
            hidden_states = self.patch_embedding(hidden_states)
        else:
            # For conv3d projection
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
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
