# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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

import inspect
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, Gemma3ForConditionalGeneration

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ...models.attention_dispatch import dispatch_attention_fn
from ...models.embeddings import get_1d_rotary_pos_embed
from ...models.modeling_utils import ModelMixin
from ...utils import is_torch_version, logging
from ..pipeline_loading_utils import _fetch_class_library_tuple


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def apply_rotary_emb(x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


# Copied from diffusers.models.transformers.transformer_ltx2.LTX2AudioVideoAttnProcessor
class LTX2AudioVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0) for the LTX-2.0 model.
    Compared to the LTX-1.0 model, we allow the RoPE embeddings for the queries and keys to be separate so that we can
    support audio-to-video (a2v) and video-to-audio (v2a) cross attention.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTX2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb)
            key = apply_rotary_emb(key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

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
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# Copied from diffusers.models.transformers.transformer_ltx2.LTX2Attention
class LTX2Attention(torch.nn.Module, AttentionModuleMixin):
    r"""
    Attention class for all LTX-2.0 attention layers. Compared to LTX-1.0, this supports specifying the query and key
    RoPE embeddings separately for audio-to-video (a2v) and video-to-audio (v2a) cross-attention.
    """

    _default_processor_cls = LTX2AudioVideoAttnProcessor
    _available_processors = [LTX2AudioVideoAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        norm_eps: float = 1e-6,
        norm_elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise NotImplementedError("Only 'rms_norm_across_heads' is supported as a valid value for `qk_norm`.")

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = query_dim
        self.heads = heads

        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head * kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        hidden_states = self.processor(
            self, hidden_states, encoder_hidden_states, attention_mask, query_rotary_emb, key_rotary_emb, **kwargs
        )
        return hidden_states


class LTX2RotaryPosEmbed1d(nn.Module):
    """
    1D rotary positional embeddings (RoPE) for the LTX 2.0 text encoder connectors.
    """

    def __init__(
        self,
        dim: int,
        base_seq_len: int = 4096,
        theta: float = 10000.0,
        double_precision: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta = theta
        self.double_precision = double_precision

    def forward(
        self,
        batch_size: int,
        pos: int,
        device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Get 1D position ids
        grid_1d = torch.arange(pos, dtype=torch.float32, device=device)
        # Get fractional indices relative to self.base_seq_len
        grid_1d = grid_1d / self.base_seq_len
        grid = grid_1d.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]

        # 2. Calculate 1D RoPE frequencies
        num_rope_elems = 2  # 1 (because 1D) * 2 (for cos, sin) = 2
        start = 1.0
        end = self.theta
        if self.double_precision:
            pow_indices = np.power(
                self.theta,
                np.linspace(
                    np.log(start) / np.log(self.theta),
                    np.log(end) / np.log(self.theta),
                    self.dim // num_rope_elems,
                    dtype=np.float64,
                ),
            )
            freqs = torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32, device=device)
        else:
            freqs = self.theta ** torch.linspace(
                start=math.log(start, self.theta),
                end=math.log(end, self.theta),
                steps=self.dim // num_rope_elems,
                device=device,
                dtype=torch.float32,
            )
            freqs = freqs * math.pi / 2.0

        # 3. Matrix-vector outer product between pos ids of shape (batch_size, seq_len) and freqs vector of shape
        # (self.dim // 2,).
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs  # [B, seq_len, self.dim // 2]

        # 4. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % num_rope_elems != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % num_rope_elems])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % num_rope_elems])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


class LTX2TransformerBlock1d(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "gelu-approximate",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=LTX2AudioVideoAttnProcessor(),
        )

        self.norm2 = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.ff = FeedForward(dim, activation_fn=activation_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, query_rotary_emb=rotary_emb)
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        ff_hidden_states = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_hidden_states

        return hidden_states


class LTX2ConnectorTransformer1d(nn.Module):
    """
    A 1D sequence transformer for modalities such as text.

    In LTX 2.0, this is used to process the text encoder hidden states for each of the video and audio streams.
    """
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: Optional[int] = 128,
        rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        eps: float = 1e-6,
        causal_temporal_positioning: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning

        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            init_registers = torch.rand(num_learnable_registers, self.inner_dim) * 2.0 - 1.0
            self.learnable_registers = torch.nn.Parameter(init_registers)

        self.rope = LTX2RotaryPosEmbed1d(
            self.inner_dim, base_seq_len=rope_base_seq_len, theta=rope_theta, double_precision=rope_double_precision
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                LTX2TransformerBlock1d(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = torch.nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False)

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        # attention_mask shape: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Replace padding with learned registers, if using
        if self.learnable_registers is not None:
            if seq_len % self.num_learnable_registers != 0:
                raise ValueError(
                    f"The `hidden_states` sequence length {hidden_states.shape[1]} should be divisible by the number"
                    f" of learnable registers {self.num_learnable_registers}"
                )

            num_register_repeats = seq_len // self.num_learnable_registers
            registers = torch.tile(self.learnable_registers, (num_register_repeats, 1))  # [seq_len, inner_dim]

            binary_attn_mask = (attention_mask >= -9000.0).int()
            if binary_attn_mask.ndim == 4:
                binary_attn_mask = binary_attn_mask.squeeze(1).squeeze(1)  # [B, 1, 1, L] --> [B, L]

            hidden_states_non_padded = [hidden_states[i, binary_attn_mask[i].bool(), :] for i in range(batch_size)]
            valid_seq_lens = [x.shape[0] for x in hidden_states_non_padded]
            pad_lengths = [seq_len - valid_seq_len for valid_seq_len in valid_seq_lens]
            padded_hidden_states = [
                F.pad(x, pad=(0, 0, 0, p), value=0) for x, p in zip(hidden_states_non_padded, pad_lengths)
            ]
            padded_hidden_states = torch.cat([x.unsqueeze(0) for x in padded_hidden_states], dim=0)  # [B, L, D]

            flipped_mask = torch.flip(binary_attn_mask, dims=[1]).unsqueeze(-1)  # [B, L, 1]
            hidden_states = flipped_mask * padded_hidden_states + (1 - flipped_mask) * registers

            # Overwrite attention_mask with an all-zeros mask if using registers.
            attention_mask = torch.zeros_like(attention_mask)

        # 2. Calculate 1D RoPE positional embeddings
        rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)

        # 3. Run 1D transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, attention_mask, rotary_emb)
            else:
                hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)

        hidden_states = self.norm_out(hidden_states)

        return hidden_states, attention_mask


class LTX2AudioVisualTextEncoder(ModelMixin, ConfigMixin):
    ignore_for_config = ["text_model"]

    @register_to_config
    def __init__(
        self,
        text_model: Optional[Gemma3ForConditionalGeneration] = None,
        text_model_id: str = "google/gemma-3-12b-it-qat-q4_0-unquantized",
        text_encoder_hidden_dim: Optional[int] = 3840,
        text_proj_in_factor: Optional[int] = 49,  # Num layers in text encoder + 1
        video_connector_num_attention_heads: int = 30,
        video_connector_attention_head_dim: int = 128,
        video_connector_num_layers: int = 2,
        video_connector_num_learnable_registers: int = 128,
        audio_connector_num_attention_heads: int = 30,
        audio_connector_attention_head_dim: int = 128,
        audio_connector_num_layers: int = 2,
        audio_connector_num_learnable_registers: Optional[int] = 128,
        rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_temporal_positioning: bool = False,
        config_only: bool = True,
    ):
        super().__init__()
        if text_model is None:
            self.set_base_text_encoder(text_model_id, config_only=config_only)
        else:
            self.base_text_encoder = text_model

        if text_encoder_hidden_dim is None:
            if hasattr(self.base_text_encoder, "config"):
                if hasattr(self.base_text_encoder.config, "hidden_size"):
                    text_encoder_hidden_dim = getattr(self.base_text_encoder.config, "hidden_size", None)
                elif hasattr(self.base_text_encoder.config, "text_config"):
                    text_encoder_hidden_dim = getattr(self.base_text_encoder.config.text_config, "hidden_size", None)
            if text_encoder_hidden_dim is None:
                raise ValueError(
                    "`text_encoder_hidden_dim` is `None` and it cannot be inferred, please provide a value for it."
                )

        if text_proj_in_factor is None:
            num_layers = None
            if hasattr(self.base_text_encoder, "config"):
                if hasattr(self.base_text_encoder.config, "num_hidden_layers"):
                    num_layers = getattr(self.base_text_encoder.config, "num_hidden_layers", None)
                elif hasattr(self.base_text_encoder.config, "text_config"):
                    num_layers = getattr(self.base_text_encoder.config.text_config, "num_hidden_layers", None)
            if num_layers is None:
                raise ValueError(
                    "`text_proj_in_factor` is `None` and it cannot be inferred, please provide a value for it."
                )
            text_proj_in_factor = num_layers + 1

        self.text_proj_in = nn.Linear(
            text_encoder_hidden_dim * text_proj_in_factor, text_encoder_hidden_dim, bias=False
        )

        self.video_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=video_connector_num_attention_heads,
            attention_head_dim=video_connector_attention_head_dim,
            num_layers=video_connector_num_layers,
            num_learnable_registers=video_connector_num_learnable_registers,
            rope_base_seq_len=rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
        )
        self.audio_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=audio_connector_num_attention_heads,
            attention_head_dim=audio_connector_attention_head_dim,
            num_layers=audio_connector_num_layers,
            num_learnable_registers=audio_connector_num_learnable_registers,
            rope_base_seq_len=rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
        )

    def set_base_text_encoder(
        self, base_text_encoder_id: str = "google/gemma-3-12b-it-qat-q4_0-unquantized", config_only: bool = True
    ):
        if config_only:
            base_text_encoder_config = AutoConfig.from_pretrained(base_text_encoder_id)
            base_text_encoder = AutoModel.from_config(base_text_encoder_config)
        else:
            base_text_encoder = AutoModel.from_pretrained(base_text_encoder_id)
        self.base_text_encoder = base_text_encoder

    @staticmethod
    def pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        device: Union[str, torch.device],
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Packs and normalizes text encoder hidden states, respecting padding. Normalization is performed per-batch and
        per-layer in a masked fashion (only over non-padded positions).

        Args:
            text_hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_dim, num_layers)`):
                Per-layer hidden_states from a text encoder (e.g. `Gemma3ForConditionalGeneration`).
            sequence_lengths (`torch.Tensor of shape `(batch_size,)`):
                The number of valid (non-padded) tokens for each batch instance.
            device: (`str` or `torch.device`, *optional*):
                torch device to place the resulting embeddings on
            padding_side: (`str`, *optional*, defaults to `"left"`):
                Whether the text tokenizer performs padding on the `"left"` or `"right"`.
            scale_factor (`int`, *optional*, defaults to `8`):
                Scaling factor to multiply the normalized hidden states by.
            eps (`float`, *optional*, defaults to `1e-6`):
                A small positive value for numerical stability when performing normalization.
        
        Returns:
            `torch.Tensor` of shape `(batch_size, seq_len, hidden_dim * num_layers)`:
                Normed and flattened text encoder hidden states.
        """
        batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
        original_dtype = text_hidden_states.dtype

        # Create padding mask
        token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        if padding_side == "right":
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask = token_indices < sequence_lengths[:, None]  # [batch_size, seq_len]
        elif padding_side == "left":
            # For left padding, valid tokens are from (T - sequence_length) to T-1
            start_indices = seq_len - sequence_lengths[:, None]  # [batch_size, 1]
            mask = token_indices >= start_indices  # [B, T]
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
        mask = mask[:, :, None, None]  # [batch_size, seq_len] --> [batch_size, seq_len, 1, 1]

        # Compute masked mean over non-padding positions of shape (batch_size, 1, 1, seq_len)
        masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
        num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

        # Compute min/max over non-padding positions of shape (batch_size, 1, 1 seq_len)
        x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

        # Normalization
        normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
        normalized_hidden_states = normalized_hidden_states * scale_factor

        # Pack the hidden states to a 3D tensor (batch_size, seq_len, hidden_dim * num_layers)
        normalized_hidden_states = normalized_hidden_states.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
        normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)
        return normalized_hidden_states

    def run_connectors(
        self, text_encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run LTX 2.0-specific text embedding post-processing logic on top of the base text encoder hidden_states.

        Args:
            text_encoder_hidden_states (`torch.Tensor`):
                Text encoder packed hidden_states of shape `(batch_size, seq_len, hidden_dim * (num_layers + 1))`.
            attention_mask (`torch.Tensor`):
                Attention mask of shape `(batch_size, seq_len)`.
        
        Returns:
            `Tuple(torch.Tensor, torch.Tensor, torch.Tensor)]`:
                Returns a 3-tuple of tensors where the first element is the video text embeddings of shape
                `(batch_size, seq_len, hidden_dim)`, the second element is the audio text embeddings of shape
                `(batch_size, seq_len, hidden_dim)`, and the third element is an attention mask of shape
                `(batch_size, seq_len)`.
        """
        # Convert to additive attention mask
        text_dtype = text_encoder_hidden_states.dtype
        connector_attn_mask = (attention_mask - 1).reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        connector_attn_mask = connector_attn_mask.to(text_dtype) * torch.finfo(text_dtype).max

        text_encoder_hidden_states = self.text_proj_in(text_encoder_hidden_states)

        video_text_embedding, new_attn_mask = self.video_connector(
            text_encoder_hidden_states, connector_attn_mask
        )

        attn_mask = (new_attn_mask < 1e-6).to(torch.int64)
        attn_mask = attn_mask.reshape(video_text_embedding.shape[0], video_text_embedding.shape[1], 1)
        video_text_embedding = video_text_embedding * attn_mask
        new_attn_mask = attn_mask.squeeze(-1)

        audio_text_embedding, _ = self.audio_connector(text_encoder_hidden_states, connector_attn_mask)

        return video_text_embedding, audio_text_embedding, new_attn_mask

    def forward(
        self,
        text_input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        padding_side: str = "left",
        scale_factor: int = 8,
    ):
        text_encoder_outputs = self.base_text_encoder(
            input_ids=text_input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)

        text_encoder_hidden_states = self.pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=text_encoder_hidden_states.device,
            padding_side=padding_side,
            scale_factor=scale_factor,
        )

        video_text_embedding, audio_text_embedding, new_attn_mask = self.run_connectors(
            text_encoder_hidden_states, attention_mask
        )

        return video_text_embedding, audio_text_embedding, new_attn_mask
