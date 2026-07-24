# Copyright 2025 The Mage Team and The HuggingFace Team. All rights reserved.
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

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _apply_rotary_emb_complex(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex rotary embeddings to ``x`` using MageFlow's adjacent-pair convention.

    Args:
        x: Query or key tensor of shape ``[B, S, H, D]``.
        freqs_cis: Complex frequency tensor of shape ``[S, D_rope // 2]`` where
            ``D_rope = sum(axes_dim)``. When ``D_rope < D`` only the first
            ``D_rope`` dimensions are rotated; the rest pass through unchanged.

    Returns:
        Tensor of same shape and dtype as *x* with rotary embeddings applied.
    """
    rope_dim = freqs_cis.shape[-1] * 2  # complex dim -> real dim
    head_dim = x.shape[-1]

    if rope_dim < head_dim:
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
    else:
        x_rope = x
        x_pass = None

    # [B, S, H, rope_dim] -> [B, S, H, rope_dim/2] complex
    x_complex = torch.view_as_complex(x_rope.float().reshape(*x_rope.shape[:-1], -1, 2))
    # freqs_cis: [S, D_rope/2] -> [1, S, 1, D_rope/2] for broadcasting
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    x_rotated = x_rotated.to(x.dtype)

    if x_pass is not None:
        return torch.cat([x_rotated, x_pass], dim=-1)
    return x_rotated


class MageFlowPosEmbed(nn.Module):
    """Complex RoPE with symmetric positive/negative frequency scaling for MageFlow.

    Computes multi-scale rotary positional embeddings for video/image tokens using
    three axes (frame, height, width). When ``scale_rope=True``, height and width
    axes use symmetric positive/negative frequency indices centered around the
    spatial midpoint.
    """

    def __init__(self, theta: int = 10000, axes_dim: list[int] = None, scale_rope: bool = True):
        super().__init__()
        if axes_dim is None:
            axes_dim = [16, 48, 48]
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        self.pos_freqs = torch.cat(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),
                self._rope_params(pos_index, self.axes_dim[1], self.theta),
                self._rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(neg_index, self.axes_dim[0], self.theta),
                self._rope_params(neg_index, self.axes_dim[1], self.theta),
                self._rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        # Complex tensors cannot be stored via register_buffer (imaginary part gets dropped).
        self._video_freq_cache: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
        """Compute complex RoPE frequencies for a 1-D position index."""
        freqs = torch.outer(
            index.float(),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim)),
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_len = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_len, -1)
        return freqs.clone().contiguous()

    def forward(self, img_ids: torch.Tensor) -> torch.Tensor:
        """Compute RoPE frequencies from image position ids.

        Args:
            img_ids: ``[seq_len, 3]`` tensor with (frame, height, width) position
                indices for each image token.

        Returns:
            Complex frequency tensor of shape ``[seq_len, head_dim // 2]``.
        """
        device = img_ids.device
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        # Determine spatial extents from the ids.
        frame = int(img_ids[:, 0].max().item()) + 1
        height = int(img_ids[:, 1].max().item()) + 1
        width = int(img_ids[:, 2].max().item()) + 1

        key = (frame, height, width, 0)
        if key not in self._video_freq_cache:
            self._video_freq_cache[key] = self._compute_video_freqs(frame, height, width, idx=0)
        freqs = self._video_freq_cache[key].to(device)

        # If img_ids has more tokens than the grid (padding), pad with zeros.
        if freqs.shape[0] < img_ids.shape[0]:
            pad_len = img_ids.shape[0] - freqs.shape[0]
            freqs = F.pad(freqs, (0, 0, 0, pad_len))

        return freqs


class MageFlowTimestepProjEmbeddings(nn.Module):
    """Timestep projection embeddings for MageFlow.

    Applies sinusoidal projection (scaled by 1000) followed by an MLP to produce
    the conditioning embedding.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        return timesteps_emb


class MageFlowAttnProcessor:
    """Attention processor for MageFlow double-stream (MMDiT) architecture.

    Implements joint attention over concatenated ``[text, image]`` tokens. RoPE is
    applied only to image query/key, not text.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "MageFlowAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Reshape to multi-head: [B, S, inner_dim] -> [B, S, H, D]
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)

        # Apply RoPE to image Q/K only (not text)
        if image_rotary_emb is not None:
            img_query = _apply_rotary_emb_complex(img_query, image_rotary_emb)
            img_key = _apply_rotary_emb_complex(img_key, image_rotary_emb)

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            # Compute QKV for text stream
            txt_query = attn.add_q_proj(encoder_hidden_states)
            txt_key = attn.add_k_proj(encoder_hidden_states)
            txt_value = attn.add_v_proj(encoder_hidden_states)

            txt_query = txt_query.unflatten(-1, (attn.heads, -1))
            txt_key = txt_key.unflatten(-1, (attn.heads, -1))
            txt_value = txt_value.unflatten(-1, (attn.heads, -1))

            txt_query = attn.norm_added_q(txt_query)
            txt_key = attn.norm_added_k(txt_key)

            # No RoPE on text — concatenate [text, image] for joint attention
            query = torch.cat([txt_query, img_query], dim=1)
            key = torch.cat([txt_key, img_key], dim=1)
            value = torch.cat([txt_value, img_value], dim=1)
        else:
            query = img_query
            key = img_key
            value = img_value

        # Joint attention via dispatch
        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split back into text and image parts
            txt_seq_len = encoder_hidden_states.shape[1]
            txt_attn_output, img_attn_output = attn_output.split_with_sizes(
                [txt_seq_len, attn_output.shape[1] - txt_seq_len], dim=1
            )
            img_attn_output = attn.to_out[0](img_attn_output)
            img_attn_output = attn.to_out[1](img_attn_output)
            txt_attn_output = attn.to_add_out(txt_attn_output)
            return img_attn_output, txt_attn_output

        return attn_output


class MageFlowAttention(nn.Module, AttentionModuleMixin):
    """Multi-head attention module for MageFlow with support for dual-stream (MMDiT) attention.

    Follows the diffusers attention pattern with ``_default_processor_cls`` and
    ``_available_processors`` for backend dispatch.
    """

    _default_processor_cls = MageFlowAttnProcessor
    _available_processors = [MageFlowAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int | None = None,
        elementwise_affine: bool = True,
        processor: "MageFlowAttnProcessor | None" = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

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
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class MageFlowTransformerBlock(nn.Module):
    """Double-stream MMDiT transformer block for MageFlow.

    Each block processes image and text streams with separate modulation (AdaLN),
    joint attention, and separate feed-forward networks.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image stream modulation and layers
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = MageFlowAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=MageFlowAttnProcessor(),
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text stream modulation and layers
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute modulation parameters for both streams
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        # Split into norm1 and norm2 modulation parameters (each has shift, scale, gate)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Image stream: norm1 + modulation
        img_shift1, img_scale1, img_gate1 = img_mod1.chunk(3, dim=-1)
        img_normed = self.img_norm1(hidden_states)
        img_modulated = img_normed * (1 + img_scale1.unsqueeze(1)) + img_shift1.unsqueeze(1)

        # Text stream: norm1 + modulation
        txt_shift1, txt_scale1, txt_gate1 = txt_mod1.chunk(3, dim=-1)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated = txt_normed * (1 + txt_scale1.unsqueeze(1)) + txt_shift1.unsqueeze(1)

        # Joint attention
        joint_attention_kwargs = joint_attention_kwargs or {}
        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Apply gates and residuals
        hidden_states = hidden_states + img_gate1.unsqueeze(1) * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1.unsqueeze(1) * txt_attn_output

        # Image stream: norm2 + MLP
        img_shift2, img_scale2, img_gate2 = img_mod2.chunk(3, dim=-1)
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2 = img_normed2 * (1 + img_scale2.unsqueeze(1)) + img_shift2.unsqueeze(1)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2.unsqueeze(1) * img_mlp_output

        # Text stream: norm2 + MLP
        txt_shift2, txt_scale2, txt_gate2 = txt_mod2.chunk(3, dim=-1)
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2 = txt_normed2 * (1 + txt_scale2.unsqueeze(1)) + txt_shift2.unsqueeze(1)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2.unsqueeze(1) * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class MageFlowTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """Transformer model for MageFlow image generation.

    A dual-stream (MMDiT) Transformer that processes image and text tokens jointly.
    Uses complex multi-scale RoPE for image positional encoding and Qwen3-VL text
    embeddings as conditioning.

    Args:
        in_channels (`int`, defaults to ``128``):
            Number of channels in the input latent (MageVAE latent channels).
        out_channels (`int`, defaults to ``128``):
            Number of channels in the output.
        context_in_dim (`int`, defaults to ``3584``):
            Dimension of the text encoder hidden states (Qwen3-VL hidden size).
        hidden_size (`int`, defaults to ``3072``):
            Inner dimension of the transformer (num_attention_heads * attention_head_dim).
        num_attention_heads (`int`, defaults to ``24``):
            Number of attention heads.
        num_layers (`int`, defaults to ``32``):
            Number of dual-stream transformer blocks.
        axes_dim (`list[int]``, defaults to ``[16, 48, 48]``):
            RoPE dimension split across axes (frame, height, width). Must sum to
            ``hidden_size // num_attention_heads``.
        patch_size (`int`, defaults to ``1``):
            Patch size for the output projection.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MageFlowTransformerBlock"]
    _repeated_blocks = ["MageFlowTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        context_in_dim: int = 3584,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        num_layers: int = 32,
        axes_dim: list[int] = [16, 48, 48],
        patch_size: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = hidden_size
        self.num_attention_heads = num_attention_heads
        attention_head_dim = hidden_size // num_attention_heads

        self.pos_embed = MageFlowPosEmbed(theta=10000, axes_dim=axes_dim, scale_rope=True)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim)
        self.context_embedder_norm = nn.RMSNorm(context_in_dim, eps=1e-6)
        self.context_embedder = nn.Linear(context_in_dim, self.inner_dim)

        self.time_text_embed = MageFlowTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                MageFlowTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.Tensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`MageFlowTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, img_seq_len, in_channels)`):
                Flattened image latent tokens.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, txt_seq_len, context_in_dim)`):
                Text encoder hidden states (Qwen3-VL embeddings).
            timestep (`torch.Tensor`):
                Raw sigma value in ``[0, 1]``. Internally multiplied by 1000.
            img_ids (`torch.Tensor` of shape `(img_seq_len, 3)`):
                Image position ids ``(frame, height, width)`` for RoPE computation.
            txt_ids (`torch.Tensor`, *optional*):
                Text position ids (unused, kept for API compatibility with pipelines).
            joint_attention_kwargs (`dict`, *optional*):
                Additional keyword arguments passed to the attention processor.
            return_dict (`bool`, defaults to ``True``):
                Whether to return a :class:`Transformer2DModelOutput` or a plain tuple.

        Returns:
            :class:`Transformer2DModelOutput` or ``tuple``.
        """
        # Embed image tokens
        hidden_states = self.x_embedder(hidden_states)

        # Embed text tokens: RMSNorm then linear projection
        encoder_hidden_states = self.context_embedder_norm(encoder_hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Timestep embedding (Timesteps module handles the 1000x scaling internally via scale=1000)
        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)

        # Add zero text vector (MageFlow does not use pooled text embeddings)
        txt_vec = torch.zeros(
            encoder_hidden_states.shape[0],
            self.inner_dim,
            dtype=temb.dtype,
            device=temb.device,
        )
        temb = temb + txt_vec

        # Compute image RoPE (text tokens are not rotated)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        image_rotary_emb = self.pos_embed(img_ids)

        # Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # Final norm and projection (image stream only)
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
