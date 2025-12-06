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
from ...utils import USE_PEFT_BACKEND, is_kernels_available, logging, scale_lora_layers, unscale_lora_layers
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import AttentionBackendName, dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import FP32LayerNorm


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
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Magi1AttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: "Magi1Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        attention_kwargs = attention_kwargs or {}

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        kv_h = attn.kv_heads if attn.kv_heads is not None else attn.heads
        key = key.unflatten(2, (kv_h, -1))
        value = value.unflatten(2, (kv_h, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None and attn.cross_attention_dim_head is None:

            def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
                # x: [B, S, H, D]
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)  # [B, S, H, D/2]
                # Broadcast cos/sin to [1, S, 1, D/2]
                cos = freqs_cos.view(1, -1, 1, freqs_cos.shape[-1])[..., 0::2]
                sin = freqs_sin.view(1, -1, 1, freqs_sin.shape[-1])[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        kv_heads = kv_h
        n_rep = attn.heads // kv_heads
        if n_rep > 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # Use MAGI backend if varlen parameters are provided
        backend = self._attention_backend
        if attention_kwargs.get("q_ranges") is not None:
            backend = AttentionBackendName.MAGI

        out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
            backend=backend,
            parallel_config=self._parallel_config,
            attention_kwargs=attention_kwargs,
        )
        out = out.flatten(2, 3).type_as(hidden_states)
        return out


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
        # Note: Output projection is handled in Magi1TransformerBlock to match original architecture
        # where [self_attn, cross_attn] outputs are concatenated, rearranged, then projected together
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
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=False)
            self.to_qkv.load_state_dict({"weight": concatenated_weights}, strict=True, assign=True)
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=False)
            self.to_kv.load_state_dict({"weight": concatenated_weights}, strict=True, assign=True)

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


class Magi1TimeTextEmbedding(nn.Module):
    """
    Combined time, text embedding module for the MAGI-1 model.

    This module handles the encoding of two types of conditioning inputs:
    1. Timestep embeddings for diffusion process control
    2. Text embeddings for text-to-video generation

    Args:
        dim (`int`): Hidden dimension of the transformer model.
        time_freq_dim (`int`): Dimension for sinusoidal time embeddings.
        text_embed_dim (`int`): Input dimension of text embeddings.
        enable_distillation (`bool`, optional): Enable distillation timestep adjustments.
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        enable_distillation: bool = False,
    ):
        super().__init__()

        # NOTE: timestep_rescale_factor=1000 to match original implementation (dit_module.py:71)
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=int(dim * 0.25))
        self.text_embedder = Magi1TextProjection(text_embed_dim, dim, adaln_dim=int(dim * 0.25))

        self.enable_distillation = enable_distillation

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
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


class Magi1RotaryPosEmbed(nn.Module):
    """
    Rotary Position Embedding for MAGI-1 model.

    Args:
        dim (`int`): The embedding dimension.
        theta (`float`, *optional*, defaults to 10000.0): Base for the geometric progression.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        num_bands = dim // 8
        exp = torch.arange(0, num_bands, dtype=torch.float32) / num_bands
        bands = 1.0 / (theta**exp)
        self.bands = nn.Parameter(bands)

    def forward(self, hidden_states: torch.Tensor, T_total: int) -> torch.Tensor:
        # Rebuild bands and embeddings every call, use if target shape changes
        device = hidden_states.device
        batch_size, num_channels, num_frames, post_patch_height, post_patch_width = hidden_states.shape
        feat_shape = [T_total, post_patch_height, post_patch_width]

        # Calculate rescale_factor for multi-resolution & multi aspect-ratio training
        # the base_size [16*16] is A predefined size based on data:(256x256)  vae: (8,8,4) patch size: (1,1,2)
        # This definition do not have any relationship with the actual input/model/setting.
        # ref_feat_shape is used to calculate innner rescale factor, so it can be float.
        rescale_factor = math.sqrt((post_patch_height * post_patch_width) / (16 * 16))
        ref_feat_shape = [T_total, post_patch_height / rescale_factor, post_patch_width / rescale_factor]

        f = torch.arange(num_frames, device=device, dtype=torch.float32)
        h = torch.arange(post_patch_height, device=device, dtype=torch.float32)
        w = torch.arange(post_patch_width, device=device, dtype=torch.float32)

        # Align spatial center (H/2, W/2) to (0,0)
        h = h - (post_patch_height - 1) / 2
        w = w - (post_patch_width - 1) / 2

        if ref_feat_shape is not None:
            # eva's scheme for resizing rope embeddings (ref shape = pretrain)
            # aligning to the endpoint e.g [0,1,2] -> [0, 0.4, 0.8, 1.2, 1.6, 2]
            fhw_rescaled = []
            fhw = [f, h, w]
            for x, shape, ref_shape in zip(fhw, feat_shape, ref_feat_shape):
                if shape == 1:  # Deal with image input
                    if ref_shape != 1:
                        raise ValueError("ref_feat_shape must be 1 when feat_shape is 1")
                    fhw_rescaled.append(x)
                else:
                    fhw_rescaled.append(x / (shape - 1) * (ref_shape - 1))
            f, h, w = fhw_rescaled

        # Create 3D meshgrid & stack into grid tensor: [T, H, W, 3]
        grid = torch.stack(torch.meshgrid(f, h, w, indexing="ij"), dim=-1)
        grid = grid.unsqueeze(-1)  # [T, H, W, 3, 1]

        # Apply frequency bands
        freqs = grid * self.bands  # [T, H, W, 3, num_bands]

        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()

        # This would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
        num_spatial_dim = 1
        for x in feat_shape:
            num_spatial_dim *= x

        freqs_cos = freqs_cos.reshape(num_spatial_dim, -1)
        freqs_sin = freqs_sin.reshape(num_spatial_dim, -1)

        return freqs_cos, freqs_sin


class Magi1TransformerBlock(nn.Module):
    """
    A transformer block used in the MAGI-1 model.

    Args:
        dim (`int`): The number of channels in the input and output.
        ffn_dim (`int`): The number of channels in the feed-forward layer.
        num_heads (`int`): The number of attention heads.
        num_kv_heads (`int`): The number of key-value attention heads.
        eps (`float`): The epsilon value for layer normalization.
        gated_linear_unit (`bool`, defaults to `False`):
            Whether to use gated linear units (SwiGLU) in the feed-forward network.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float = 1e-6,
        gated_linear_unit: bool = False,
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

        # Combined output projection for concatenated [self_attn, cross_attn] outputs
        # Matches original architecture: concat -> rearrange -> project
        self.attn_proj = nn.Linear(2 * dim, dim, bias=False)

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
        # Use SwiGLU activation for gated linear units, GELU otherwise
        activation_fn = "swiglu" if gated_linear_unit else "gelu"
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn=activation_fn, bias=False)

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
        temb_map: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        residual = hidden_states.float()

        mixed_qqkv = self.norm1(hidden_states)

        self_attn_output = self.attn1(mixed_qqkv, None, None, rotary_emb, attention_kwargs=self_attention_kwargs)

        cross_attn_output = self.attn2(
            mixed_qqkv, encoder_hidden_states, encoder_attention_mask, None, attention_kwargs=encoder_attention_kwargs
        )

        # 3. Concatenate attention outputs
        # Shape: [sq, b, num_heads * head_dim + num_heads * head_dim] = [sq, b, 2 * dim]
        hidden_states = torch.concat([self_attn_output, cross_attn_output], dim=2)

        # 4. Rearrange to interleave query groups from self and cross attention
        # This matches the original: rearrange(attn_out, "sq b (n hn hd) -> sq b (hn n hd)", n=2, hn=num_query_groups)
        # The interleaving is done at the query group level (not per-head level)
        # For 48 heads with 8 query groups: each group has 6 heads = 768 dims
        # Interleaving pattern: [self_g0, cross_g0, self_g1, cross_g1, ..., self_g7, cross_g7]
        batch_size, seq_len, _ = hidden_states.shape
        num_query_groups = self.attn1.kv_heads if self.attn1.kv_heads is not None else self.attn1.heads
        group_dim = self_attn_output.shape[2] // num_query_groups
        hidden_states = hidden_states.reshape(batch_size, seq_len, 2, num_query_groups, group_dim)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4)
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)

        # 5. Apply combined projection
        hidden_states = self.attn_proj(hidden_states)

        gate_output = self.ada_modulate_layer(temb)
        # Softcap with 1.0
        gate_output = torch.tanh(gate_output.float()).to(gate_output.dtype)
        gate_msa, gate_mlp = gate_output.chunk(2, dim=-1)

        # Residual connection for self-attention
        original_dtype = hidden_states.dtype
        hidden_states = range_mod_pytorch(hidden_states.float().transpose(0, 1), temb_map, gate_msa).transpose(0, 1)
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)

        # Residual connection for MLP
        original_dtype = hidden_states.dtype
        hidden_states = range_mod_pytorch(hidden_states.float().transpose(0, 1), temb_map, gate_mlp).transpose(0, 1)
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
        gated_linear_unit (`bool`, defaults to `False`):
            Whether to use gated linear units (SwiGLU activation) in the feed-forward network. If True, uses SwiGLU
            activation; if False, uses GELU activation.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "rope"]
    _no_split_modules = ["Magi1TransformerBlock"]
    _keep_in_fp32_modules = [
        "condition_embedder",
        "scale_shift_table",
        "norm_out",
        "norm_q",
        "norm_k",
        "patch_embedding",
        "rope",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["Magi1TransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

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
        x_rescale_factor: int = 1,
        half_channel_vae: bool = False,
        enable_distillation: bool = False,
        gated_linear_unit: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.rope = Magi1RotaryPosEmbed(inner_dim // num_attention_heads)

        # 2. Condition embeddings
        self.condition_embedder = Magi1TimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            text_embed_dim=cross_attention_dim,
            enable_distillation=enable_distillation,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Magi1TransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    num_kv_heads,
                    eps,
                    gated_linear_unit,
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
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        denoising_range_num: Optional[int] = None,
        range_num: Optional[int] = None,
        slice_point: Optional[int] = 0,
        kv_range: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None,
        distill_interval: Optional[int] = None,
        extract_prefix_video_feature: Optional[bool] = False,
        fwd_extra_1st_chunk: Optional[bool] = False,
        distill_nearly_clean_chunk: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the MAGI-1 transformer.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of shape `(batch_size, num_channels, num_frames, height, width)`.
            timestep (`torch.LongTensor`):
                Timesteps for diffusion process. Shape: `(batch_size, denoising_range_num)`.
            encoder_hidden_states (`torch.Tensor`):
                Text embeddings from the text encoder for cross-attention conditioning.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for text embeddings to handle variable-length sequences.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dictionary or a tuple.
            attention_kwargs (`dict`, *optional*):
                Additional keyword arguments for attention processors (e.g., LoRA scale).
            denoising_range_num (`int`, *optional*):
                Number of denoising ranges for autoregressive video generation. Each range represents a chunk of video
                frames being denoised in parallel.
            range_num (`int`, *optional*):
                Total number of ranges in the video generation process.
            slice_point (`int`, *optional*, defaults to 0):
                Index indicating how many clean (already generated) frames precede the current denoising chunks. Used
                for autoregressive context.
            kv_range (`Tuple[int, int]`, *optional*):
                Key-value attention ranges for each denoising chunk, defining which frames each chunk can attend to.
                Required for MAGI-1's autoregressive attention pattern.
            num_steps (`int`, *optional*):
                Number of diffusion sampling steps. Used for distillation timestep adjustments.
            distill_interval (`int`, *optional*):
                Interval for distillation when using distilled models. Used with `num_steps`.
            extract_prefix_video_feature (`bool`, *optional*, defaults to `False`):
                Whether to extract features from prefix (clean) video frames.
            fwd_extra_1st_chunk (`bool`, *optional*, defaults to `False`):
                Whether to forward an extra first chunk in the current iteration.
            distill_nearly_clean_chunk (`bool`, *optional*, defaults to `False`):
                Whether to apply distillation to nearly clean chunks during generation.

        Returns:
            `Transformer2DModelOutput` or `tuple`:
                If `return_dict` is True, returns a `Transformer2DModelOutput` containing the sample. Otherwise,
                returns a tuple with the sample as the first element.

        Note:
            MAGI-1 uses an autoregressive video generation approach where video frames are generated in chunks. The
            `denoising_range_num`, `kv_range`, and related parameters control this autoregressive pattern, allowing
            each chunk to attend to previously generated (clean) frames while maintaining causal constraints.
        """
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

        # `kv_range` is optional when not using MAGI varlen backend. No requirement here.

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        frame_in_range = post_patch_num_frames // denoising_range_num
        prev_clean_T = frame_in_range * slice_point
        T_total = post_patch_num_frames + prev_clean_T

        hidden_states = hidden_states * self.config.x_rescale_factor

        if self.config.half_channel_vae:
            if hidden_states.shape[1] != 16:
                raise ValueError(
                    "When `config.half_channel_vae` is True, the input `hidden_states` must have 16 channels."
                )
            hidden_states = torch.cat([hidden_states, hidden_states], dim=1)

        # Patch & position embedding
        hidden_states = self.patch_embedding(hidden_states)
        freqs_cos, freqs_sin = self.rope(hidden_states, T_total)
        # The shape of freqs_* is (total_seq_length, head_dim). Keep only the last tokens corresponding to current window.
        keep = post_patch_num_frames * post_patch_height * post_patch_width
        freqs_cos = freqs_cos[-keep:]
        freqs_sin = freqs_sin[-keep:]
        rotary_emb = (freqs_cos, freqs_sin)

        hidden_states = hidden_states.flatten(2).transpose(
            1, 2
        )  # (B, post_patch_num_frames * post_patch_height * post_patch_width, C)

        temb, y_encoder_attention, y_adaln = self.condition_embedder(
            timestep.flatten(),
            encoder_hidden_states,
            num_steps,
            distill_interval,
        )

        # Pool AdaLN text conditioning over valid tokens (mask) to get per-batch vector, then broadcast per range
        if encoder_attention_mask is not None:
            mask_2d = encoder_attention_mask.squeeze(1).squeeze(1).to(torch.bool)  # [B, L]
            denom = mask_2d.sum(dim=1, keepdim=True).clamp(min=1)
            y_adaln_pooled = (y_adaln * mask_2d.unsqueeze(-1)).sum(dim=1) / denom
        else:
            mask_2d = None
            y_adaln_pooled = y_adaln.mean(dim=1)

        temb = temb.reshape(batch_size, denoising_range_num, -1) + y_adaln_pooled.unsqueeze(1).expand(
            -1, denoising_range_num, -1
        )

        seqlen_per_chunk = (post_patch_num_frames * post_patch_height * post_patch_width) // denoising_range_num
        temb_map = torch.arange(batch_size * denoising_range_num, device=hidden_states.device)
        temb_map = torch.repeat_interleave(temb_map, seqlen_per_chunk)
        temb_map = temb_map.reshape(batch_size, -1).transpose(0, 1).contiguous()

        # Build varlen metadata for MAGI backend
        clip_token_nums = post_patch_height * post_patch_width * frame_in_range

        self_attention_kwargs = None
        if kv_range is not None:
            cu_seqlens_q = (
                torch.tensor(
                    [0] + ([clip_token_nums] * denoising_range_num * batch_size),
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
                .cumsum(-1)
                .to(torch.int32)
            )
            # q_ranges pairs from cu_seqlens_q
            q_ranges = torch.cat([cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1)
            flat_kv = torch.unique(kv_range, sorted=True)
            max_seqlen_k = int((flat_kv[-1] - flat_kv[0]).item())
            self_attention_kwargs = {
                "q_ranges": q_ranges,
                "k_ranges": kv_range,
                "max_seqlen_q": clip_token_nums,
                "max_seqlen_k": max_seqlen_k,
            }

        encoder_attention_kwargs = None
        if mask_2d is not None:
            y_index = mask_2d.sum(dim=-1).to(torch.int32)
            cu_seqlens_q = (
                torch.tensor(
                    [0] + ([clip_token_nums] * denoising_range_num * batch_size),
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
                .cumsum(-1)
                .to(torch.int32)
            )
            cu_seqlens_k = torch.cat([y_index.new_zeros(1, dtype=torch.int32), y_index.to(torch.int32)]).cumsum(-1)
            q_ranges = torch.cat([cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1)
            k_ranges = torch.cat([cu_seqlens_k[:-1].unsqueeze(1), cu_seqlens_k[1:].unsqueeze(1)], dim=1)
            max_seqlen_kv = int(y_index.max().item()) if y_index.numel() > 0 else 0
            encoder_attention_kwargs = {
                "q_ranges": q_ranges,
                "k_ranges": k_ranges,
                "max_seqlen_q": clip_token_nums,
                "max_seqlen_k": max_seqlen_kv,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
            }

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    y_encoder_attention,
                    temb,
                    rotary_emb,
                    temb_map,
                    mask_2d,
                    self_attention_kwargs,
                    encoder_attention_kwargs,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    y_encoder_attention,
                    temb,
                    rotary_emb,
                    temb_map,
                    mask_2d,
                    self_attention_kwargs,
                    encoder_attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
