# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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
#
# This file derives from the FAR architecture (Gu et al., 2025, arXiv:2503.19325) and adds the
# AnyFlow dual-timestep flow-map embedding (AnyFlowDualTimestepTextImageEmbedding) introduced by
# Yuchao Gu, Guian Fang et al. (arXiv:2605.13724). The base 3D DiT structure is adapted from the
# v0.35.1 Wan2.1 transformer (transformer_wan.py); upstream Wan has since been refactored, so
# this file is intentionally self-contained rather than annotated with `# Copied from`.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import apply_lora_scale, logging
from ...utils.torch_utils import maybe_adjust_dtype_for_device
from ..attention import AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
    # MPS / NPU backends do not support complex128 / float64; fall back to float32 on those devices.
    rotary_dtype = maybe_adjust_dtype_for_device(torch.float64, hidden_states.device)
    x_rotated = torch.view_as_complex(hidden_states.to(rotary_dtype).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(hidden_states)


class AnyFlowAttnProcessor:
    """
    Bidirectional self-attention processor for AnyFlow. Routes through
    :func:`~diffusers.models.attention_dispatch.dispatch_attention_fn` so any SDPA-compatible backend is supported
    (SDPA, flash-attn, xformers, flex, …). FAR causal generation lives in
    :class:`~diffusers.models.transformers.transformer_anyflow_far.AnyFlowCausalAttnProcessor`.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AnyFlowAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0 or higher."
            )

    def __call__(
        self,
        attn: "AnyFlowAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[Any] = None,
        rotary_emb: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Layout (B, H, L, D) for rotary application; transposed to (B, L, H, D) before dispatch.
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb["query"])
            key = apply_rotary_emb(key, rotary_emb["key"])

        hidden_states = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AnyFlowCrossAttnProcessor:
    """
    Cross-attention processor for AnyFlow. Always uses the dispatched SDPA-compatible backend; no rotary embedding or
    KV cache is applied to the text→video cross-attention path.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AnyFlowCrossAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0 or higher."
            )

    def __call__(
        self,
        attn: "AnyFlowAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # (B, L, H, D) layout for dispatch_attention_fn.
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
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AnyFlowAttention(torch.nn.Module, AttentionModuleMixin):
    """
    Attention module used by :class:`AnyFlowTransformerBlock`. Layout matches the legacy
    :class:`~diffusers.models.attention_processor.Attention` so existing AnyFlow checkpoints load bit-exactly into this
    class.
    """

    _default_processor_cls = AnyFlowAttnProcessor
    _available_processors = [AnyFlowAttnProcessor, AnyFlowCrossAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-6,
        processor: Optional[Any] = None,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(0.0),
            ]
        )
        # ``rms_norm_across_heads`` per-axis: normalize Q and K across the entire ``heads * dim_head``
        # channel axis. We use diffusers' RMSNorm (rather than ``torch.nn.RMSNorm``) so the numerics
        # match the legacy Attention class that produced the released checkpoints.
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

        self.set_processor(processor if processor is not None else self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


class AnyFlowImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class AnyFlowDualTimestepTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        gate_value: float,
        deltatime_type: str,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.delta_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = AnyFlowImageEmbedding(image_embed_dim, dim)

        self.register_buffer("delta_emb_gate", torch.tensor([gate_value], dtype=torch.float32), persistent=False)
        self.deltatime_type = deltatime_type

    def forward_timestep(
        self, timestep: torch.Tensor, delta_timestep: torch.Tensor, encoder_hidden_states, token_per_frame
    ):
        batch_size, num_frames = timestep.shape
        timestep = timestep.reshape(-1)
        delta_timestep = delta_timestep.reshape(-1)

        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        delta_timestep = self.timesteps_proj(delta_timestep)

        delta_embedder_dtype = next(iter(self.delta_embedder.parameters())).dtype
        if delta_timestep.dtype != delta_embedder_dtype and delta_embedder_dtype != torch.int8:
            delta_timestep = delta_timestep.to(delta_embedder_dtype)
        delta_emb = self.delta_embedder(delta_timestep).type_as(encoder_hidden_states)

        gate = self.delta_emb_gate.to(delta_embedder_dtype)

        rt_emb = (1 - gate) * temb + gate * delta_emb
        timestep_proj = self.time_proj(self.act_fn(rt_emb))

        rt_emb = rt_emb.unflatten(0, (batch_size, num_frames)).repeat_interleave(token_per_frame, dim=1)
        timestep_proj = timestep_proj.unflatten(0, (batch_size, num_frames)).repeat_interleave(token_per_frame, dim=1)

        return rt_emb, timestep_proj

    def forward(
        self,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        layout_cfg=None,
    ):
        if self.deltatime_type == "r":
            delta_timestep = r_timestep
        elif self.deltatime_type == "t-r":
            delta_timestep = timestep - r_timestep
        else:
            raise NotImplementedError

        timestep, timestep_proj = self.forward_timestep(
            timestep, delta_timestep, encoder_hidden_states, layout_cfg["full_token_per_frame"]
        )

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return timestep, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class AnyFlowRotaryPosEmbed(nn.Module):
    """Rotary positional embedding for the bidirectional AnyFlow transformer.

    The FAR causal variant lives in :mod:`~diffusers.models.transformers.transformer_anyflow_far` and additionally
    handles compressed-frame chunks; this bidi class produces frequencies for the single full-resolution token grid
    only.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Frequency table is lazily built per-device in ``_build_freqs``: MPS / NPU don't support
        # complex128, so we downcast to complex64 there.
        self._freqs_cache: Optional[Tuple[Any, torch.Tensor]] = None

    def _build_freqs(self, device: torch.device) -> torch.Tensor:
        cache_key = (device.type, str(device))
        if self._freqs_cache is not None and self._freqs_cache[0] == cache_key:
            return self._freqs_cache[1]

        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, device)

        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim

        freqs_list = []
        for dim in (t_dim, h_dim, w_dim):
            f = get_1d_rotary_pos_embed(
                dim,
                self.max_seq_len,
                self.theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=freqs_dtype,
            )
            freqs_list.append(f.to(device))
        freqs = torch.cat(freqs_list, dim=1)
        self._freqs_cache = (cache_key, freqs)
        return freqs

    def _forward_full_frame(self, num_frames, height, width, device) -> torch.Tensor:
        ppf, pph, ppw = num_frames, height, width

        freqs_full = self._build_freqs(device)
        if min(ppf, pph, ppw) <= 0:
            freq_channels = self.attention_head_dim // 2
            return torch.empty((ppf, pph, ppw, freq_channels), dtype=freqs_full.dtype, device=device)

        freqs = freqs_full.split_with_sizes(
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
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)
        return freqs

    def forward(self, layout_cfg, device):
        freqs = self._forward_full_frame(
            num_frames=layout_cfg["total_frames"],
            height=layout_cfg["full_frame_shape"][0],
            width=layout_cfg["full_frame_shape"][1],
            device=device,
        )
        freqs = freqs.flatten(start_dim=0, end_dim=2)
        freqs = freqs[None, None, ...]
        return {"query": freqs, "key": freqs}


class AnyFlowTransformerBlock(nn.Module):
    """AnyFlow transformer block.

    The self-attention processor is chosen at construction by ``is_causal``: the bidirectional transformer passes
    ``is_causal=False`` (the default), the FAR causal transformer passes ``is_causal=True``. The forward pass is
    identical in both modes — only the processor differs, so all causal-specific machinery (BlockMask, KV cache) lives
    inside the processor.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        is_causal: bool = False,
    ):
        super().__init__()

        self.is_causal = is_causal

        # 1. Self-attention. The causal processor lives in the FAR sibling module; lazy-import to
        # avoid a circular import at module load time.
        if is_causal:
            from .transformer_anyflow_far import AnyFlowCausalAttnProcessor

            self_attn_processor = AnyFlowCausalAttnProcessor()
        else:
            self_attn_processor = AnyFlowAttnProcessor()

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = AnyFlowAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=self_attn_processor,
        )

        # 2. Cross-attention
        self.attn2 = AnyFlowAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=AnyFlowCrossAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache=None,
        kv_cache_flag=None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            shift_msa.squeeze(2),
            scale_msa.squeeze(2),
            gate_msa.squeeze(2),
            c_shift_msa.squeeze(2),
            c_scale_msa.squeeze(2),
            c_gate_msa.squeeze(2),
        )  # noqa: E501

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn1_kwargs = {
            "hidden_states": norm_hidden_states,
            "rotary_emb": rotary_emb,
            "attention_mask": attention_mask,
        }
        # KV cache kwargs are only consumed by the FAR causal processor; the bidi processor
        # doesn't accept them, so we forward them only when they're actually populated.
        if kv_cache is not None:
            attn1_kwargs["kv_cache"] = kv_cache
            attn1_kwargs["kv_cache_flag"] = kv_cache_flag
        attn_output = self.attn1(**attn1_kwargs)
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


class AnyFlowTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    Bidirectional 3D Transformer for AnyFlow flow-map sampling.

    The architecture is the v0.35.1 Wan2.1 3D DiT backbone with one structural change: the timestep embedder is
    replaced by ``AnyFlowDualTimestepTextImageEmbedding`` so that every forward call conditions on both the source
    timestep ``t`` and the target timestep ``r``. This is the embedding required to learn the flow map
    :math:`\Phi_{r\leftarrow t}` introduced in [AnyFlow](https://huggingface.co/papers/2605.13724) by Yuchao Gu, Guian
    Fang et al.

    For frame-level autoregressive (FAR causal) generation, use ``AnyFlowFARTransformer3DModel`` instead; that variant
    adds the FAR causal block-mask and a compressed-frame patch embedding on top of the same backbone.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Number of attention heads.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input latent.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output latent.
        text_dim (`int`, defaults to `4096`):
            Input dimension for text embeddings (UMT5).
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            Number of transformer blocks.
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon for normalization layers.
        image_dim (`Optional[int]`, *optional*, defaults to `None`):
            Image embedding dimension for I2V conditioning (`1280` for the original Wan2.1-I2V model).
        rope_max_seq_len (`int`, defaults to `1024`):
            Maximum sequence length used to precompute rotary position frequencies.
        gate_value (`float`, defaults to `0.25`):
            Mixing gate between source-timestep and delta-timestep embeddings (the AnyFlow paper's :math:`g` parameter,
            fixed at 0.25 in stage-1 distillation).
        deltatime_type (`str`, defaults to `'r'`):
            Either ``"r"`` (delta is the target timestep) or ``"t-r"`` (delta is the absolute interval).
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["AnyFlowTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _repeated_blocks = ["AnyFlowTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        gate_value: float = 0.25,
        deltatime_type: str = "r",
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding (full-frame only).
        self.rope = AnyFlowRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embedding (always dual-timestep for AnyFlow distilled checkpoints).
        self.condition_embedder = AnyFlowDualTimestepTextImageEmbedding(
            dim=inner_dim,
            gate_value=gate_value,
            deltatime_type=deltatime_type,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                AnyFlowTransformerBlock(inner_dim, ffn_dim, num_attention_heads, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def _unpack_latent_sequence(self, latents, num_frames, height, width, patch_size):
        batch_size, num_patches, channels = latents.shape
        height, width = height // patch_size, width // patch_size

        latents = latents.view(
            batch_size * num_frames, height, width, patch_size, patch_size, channels // (patch_size * patch_size)
        )
        latents = latents.permute(0, 5, 1, 3, 2, 4)
        latents = latents.reshape(
            batch_size, num_frames, channels // (patch_size * patch_size), height * patch_size, width * patch_size
        )
        return latents

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple]:
        """
        Bidirectional flow-map forward pass. ``hidden_states`` is laid out as ``(B, F, C, H, W)`` (per-frame latents).
        The input is patchified with the standard ``patch_embedding`` (kernel = stride = ``patch_size``) and denoised
        with global bidirectional self-attention over the resulting flat token sequence.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
                Input video latents.
            timestep (`torch.Tensor`):
                Source (noisier) flow-map timestep `t`.
            r_timestep (`torch.Tensor`):
                Target (cleaner) flow-map timestep `r`; defines the destination of the flow-map step.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_len, embed_dims)`):
                Text-conditioning embeddings.
            encoder_hidden_states_image (`torch.Tensor`, *optional*):
                Image-conditioning embeddings; concatenated before the text tokens when provided.
            attention_kwargs (`dict`, *optional*):
                Kwargs forwarded to the `AttentionProcessor` as defined under `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple` whose
            first element is the predicted velocity tensor.
        """
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        full_token_per_frame = (height * width) // (self.config.patch_size[1] * self.config.patch_size[2])

        layout_cfg = {
            "total_frames": num_frames,
            "full_frame_shape": (height // self.config.patch_size[1], width // self.config.patch_size[2]),
            "full_token_per_frame": full_token_per_frame,
        }

        rotary_emb = self.rope(layout_cfg=layout_cfg, device=hidden_states.device)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            r_timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            layout_cfg=layout_cfg,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        attention_mask = None

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_mask)

        # Output norm, projection & unpatchify.
        # `temb` is always 3D from `condition_embedder.forward()` (broadcast over total tokens).
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)

        # Move shift/scale to hidden_states' device for multi-GPU accelerate inference.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        output = self._unpack_latent_sequence(
            hidden_states,
            num_frames=layout_cfg["total_frames"],
            height=height,
            width=width,
            patch_size=self.config.patch_size[1],
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
