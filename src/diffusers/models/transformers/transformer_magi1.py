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
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.transformers.transformer_wan._get_qkv_projections
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

# Copied from diffusers.models.transformers.transformer_wan._get_added_kv_projections
def _get_added_kv_projections(attn: "Magi1Attention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


def range_mod_pytorch(x, c_mapping, gatings):
    """
    PyTorch implementation of range_mod_triton.
    # TODO: Ensure that this implementation is correct and matches the range_mod_triton implementation.
    
    Inputs:
        x: (s, b, h). Tensor of inputs embedding (images or latent representations of images)
        c_mapping: (s, b). Tensor of condition map
        gatings: (b, denoising_range_num, h). Tensor of condition embedding
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
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # Perform Grouped-query Attention (GQA)
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class Magi1Attention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Magi1AttnProcessor
    _available_processors = [Magi1AttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.fuse_projections
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
    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.unfuse_projections
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


class Magi1TextProjection(nn.Module):
    """
    Projects caption embeddings.
    """

    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.y_proj_xattn = nn.Sequential(nn.Linear(in_features, hidden_size, bias=True), nn.SiLU())
        self.y_proj_adaln = nn.Linear(in_features, hidden_size, bias=True)

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
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.text_embedder = Magi1TextProjection(text_embed_dim, dim)

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

        time_embedder_dtype = get_parameter_dtype(self.time_embedder)
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        y_xattn, y_adaln = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, y_xattn, y_adaln, encoder_hidden_states_image


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
        num_kv_heads (`int`): The number of key-value attention heads.
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
        num_kv_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Magi1Attention(
            dim=dim,
            heads=num_heads,
            kv_heads=num_kv_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=Magi1AttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = Magi1Attention(
            dim=dim,
            heads=num_heads,
            kv_heads=num_kv_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=Magi1AttnProcessor(),
        )
        self.linear_proj = nn.Linear(dim * 2, dim, bias=False)
        self.ada_modulate_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                int(dim * 0.25),
                int(dim * 2),
            )
        )
        self.self_attn_post_norm = FP32LayerNorm(dim, eps)
        self.self_attn_post_norm.weight += 1
        self.norm2 = FP32LayerNorm(dim, eps)

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu")

        self.mlp_post_norm = FP32LayerNorm(dim, eps)
        self.mlp_post_norm.weight += 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        condition_map: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states.float()

        # 1. Self-attention
        self_attn_output = self.attn1(hidden_states=hidden_states, rotary_emb=rotary_emb)

        # 2. Cross-attention
        cross_attn_output = self.attn2(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

        attn_out = torch.concat([self_attn_output, cross_attn_output], dim=2)
        hidden_states = self.linear_proj(attn_out)

        gate_output = self.ada_modulate_layer(temb)
        # Softcap with 1.0
        gate_output = torch.tanh(gate_output.float()).to(gate_output.dtype)
        gate_msa, gate_mlp = gate_output.chunk(2, dim=-1)

        # Residual connection for self-attention
        original_dtype = hidden_states.dtype
        hidden_states = range_mod_pytorch(hidden_states.float(), condition_map, gate_msa)
        hidden_states = self.self_attn_post_norm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)

        # Residual connection for MLP
        original_dtype = hidden_states.dtype
        hidden_states = range_mod_pytorch(hidden_states.float(), condition_map, gate_mlp)
        hidden_states = self.mlp_post_norm(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states.to(original_dtype)
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
    _no_split_modules = ["Magi1TransformerBlock", "norm_out"]
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
        upcast_attention: bool = False,
        image_embed_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = Magi1RotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        # 2. Condition embeddings
        self.condition_embedder = Magi1TimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            text_embed_dim=cross_attention_dim,
            image_embed_dim=image_embed_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        # For image-to-video tasks, we may need additional projections
        added_kv_proj_dim = image_embed_dim if image_embed_dim is not None else None

        self.blocks = nn.ModuleList(
            [
                Magi1TransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, num_kv_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size), bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        condition_map: Optional[torch.Tensor] = None,
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
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        timestep_shape = timestep.shape
        temb, y_xattn, y_adaln, encoder_hidden_states_image = self.condition_embedder(
            timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image
        )

        temb = temb.reshape(*timestep_shape, -1) + y_adaln

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, rotary_emb, condition_map
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary_emb, condition_map)

        hidden_states = self.norm_out(hidden_states, temb=temb)
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
