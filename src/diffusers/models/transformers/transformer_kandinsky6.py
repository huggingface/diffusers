# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""Kandinsky 6 Diffusers transformer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from ...configuration_utils import ConfigMixin, register_to_config
from ...hooks import MagCacheConfig
from ...hooks.hooks import HookRegistry, ModelHook, StateManager
from ...hooks.mag_cache import MagCacheState
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import (
    _CAN_USE_FLEX_ATTN,
    AttentionBackendName,
    dispatch_attention_fn,
)
from ..cache_utils import CacheMixin
from ..embeddings import get_timestep_embedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    """Return inverse frequencies for rotary position embeddings.

    Args:
        dim (`int`): Number of frequency values to generate.
        max_period (`float`, *optional*, defaults to 10000.0): Maximum period
            used by the frequency schedule.

    Returns:
        `torch.Tensor`: Frequency values in float32.
    """
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def apply_scale_shift_norm(norm, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """AdaLN-style affine in fp32, cast back to ``x.dtype``."""
    if x.ndim > 2 and scale.ndim == 2:
        shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[-1],)
        scale, shift = scale.reshape(shape), shift.reshape(shape)
    return (norm(x.float()) * (scale.float() + 1.0) + shift.float()).to(dtype=x.dtype)


def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor) -> Tensor:
    """Residual gate in fp32, cast back to ``x.dtype``."""
    if x.ndim > 2 and gate.ndim == 2:
        gate = gate.reshape((gate.shape[0],) + (1,) * (x.ndim - 2) + (gate.shape[-1],))
    return (x.float() + gate.float() * out.float()).to(dtype=x.dtype)


def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """RoPE apply in fp32 (rope tables are fp32), cast back to ``x.dtype``."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope.float() * x_).sum(dim=-1).reshape(*x.shape).to(dtype=x.dtype)


def _local_patch(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, g1, H // g2, g2, W // g3, g3, *x.shape[dim + 3 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 2, d + 4, d + 1, d + 3, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)


def _local_merge(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, H // g2, W // g3, g1, g2, g3, *x.shape[dim + 2 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 3, d + 1, d + 4, d + 2, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)


def nabla_block_mask(
    q: Tensor,
    k: Tensor,
    sta: Tensor,
    thr: float = 0.9,
    block_size: int = 64,
) -> BlockMask:
    """Build a dynamic NABLA block mask from query/key statistics and an STA prior."""
    B, h, S, D = q.shape
    s1 = S // block_size
    qa = q.reshape(B, h, s1, block_size, D).mean(-2)
    ka = k.reshape(B, h, s1, block_size, D).mean(-2).transpose(-2, -1)
    attn_map = torch.softmax((qa @ ka) / math.sqrt(D), dim=-1)

    vals, inds = attn_map.sort(-1)
    mask = (vals.cumsum_(-1) >= 1 - thr).int().gather(-1, inds.argsort(-1))
    mask = torch.logical_or(mask, sta)

    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb),
        kv_inds,
        kv_nb,
        kv_inds,
        BLOCK_SIZE=block_size,
        mask_mod=None,
    )


class RoPE1D(nn.Module):
    """1-D Rotary Position Embedding — used for text and audio sequences."""

    def __init__(
        self,
        dim: int,
        max_pos: int = 2048,
        max_period: float = 10000.0,
        freqs_scaling: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.max_period = max_period
        self.freqs_scaling = freqs_scaling
        freq = get_freqs(dim // 2, max_period) * freqs_scaling
        self.register_buffer("args", torch.outer(torch.arange(max_pos, dtype=freq.dtype), freq), persistent=False)

    def forward(self, pos: Tensor) -> Tensor:
        # RoPE tables are fp32; keep trig in fp32.
        args = self.args[pos]  # (seq_len, dim//2)
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)

    def reset_parameters(self) -> None:
        freq = get_freqs(self.dim // 2, self.max_period).to(self.args.device) * self.freqs_scaling
        self.args = torch.outer(torch.arange(self.max_pos, dtype=freq.dtype, device=freq.device), freq)


class RoPE3D(nn.Module):
    """3-D Rotary Position Embedding — used for video spatial-temporal tokens (T, H, W)."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period
        for i, (d, mp) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(d // 2, max_period)
            self.register_buffer(f"args_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype), freq), persistent=False)

    def forward(
        self,
        shape: tuple,
        pos: list[Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        T, H, W = shape
        args_t = getattr(self, "args_0")[pos[0]] / scale_factor[0]  # (T, d//2)
        args_h = getattr(self, "args_1")[pos[1]] / scale_factor[1]  # (H, d//2)
        args_w = getattr(self, "args_2")[pos[2]] / scale_factor[2]  # (W, d//2)

        args = torch.cat(
            [
                args_t.view(T, 1, 1, -1).expand(T, H, W, -1),
                args_h.view(1, H, 1, -1).expand(T, H, W, -1),
                args_w.view(1, 1, W, -1).expand(T, H, W, -1),
            ],
            dim=-1,
        )
        cos, sin = torch.cos(args), torch.sin(args)
        rope = torch.stack([cos, -sin, sin, cos], dim=-1)  # (T, H, W, total_dim, 4)
        rope = rope.view(*rope.shape[:-1], 2, 2)  # (T, H, W, total_dim, 2, 2)
        return rope.unsqueeze(-4)  # (T, H, W, 1, total_dim, 2, 2)

    def reset_parameters(self) -> None:
        for i, (d, mp) in enumerate(zip(self.axes_dims, self.max_pos)):
            freq = get_freqs(d // 2, self.max_period).to(getattr(self, f"args_{i}").device)
            setattr(self, f"args_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype, device=freq.device), freq))


# Diffusers MagCache adapter for the K6 multimodal transformer.
#
# Diffusers provides the public :class:`MagCacheConfig` and the stateful hook
# infrastructure. K6's fused visual blocks carry video and audio streams
# together, so the stock hook needs a small adapter to preserve both tensors
# when a block is skipped. The adapter keeps the Diffusers ``enable_cache`` /
# ``disable_cache`` API and uses the Diffusers MagCache state and configuration.


_HEAD_HOOK = "kandinsky6_mag_cache_head"
_BLOCK_HOOK = "kandinsky6_mag_cache_block"


def _streams_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Tensor | None, Tensor | None]:
    video = kwargs.get("vis", args[0] if args else None)
    audio = kwargs.get("aud", args[1] if len(args) > 1 else None)
    return video, audio


def _add_residual(
    video: Tensor | None,
    audio: Tensor | None,
    residual: Tensor | tuple[Tensor | None, Tensor | None],
) -> Tensor | tuple[Tensor | None, Tensor | None]:
    if audio is None:
        if not isinstance(residual, Tensor):
            raise RuntimeError("K6 MagCache residual does not match a video-only block")
        return video + residual if video is not None else video
    if not isinstance(residual, tuple):
        raise RuntimeError("K6 MagCache residual does not contain an audio stream")
    video_residual, audio_residual = residual
    return (
        video + video_residual if video is not None and video_residual is not None else video,
        audio + audio_residual if audio_residual is not None else audio,
    )


def _residual(
    output: Tensor | tuple[Tensor | None, Tensor | None],
    input_video: Tensor | None,
    input_audio: Tensor | None,
) -> Tensor | tuple[Tensor | None, Tensor | None]:
    output_video, output_audio = _streams_from_args((output,), {}) if isinstance(output, Tensor) else output
    if input_audio is None:
        if output_video is None or input_video is None:
            return output_video
        return output_video - input_video
    return (
        output_video - input_video if output_video is not None and input_video is not None else None,
        output_audio - input_audio if output_audio is not None else None,
    )


def _should_compute(state: MagCacheState, config: MagCacheConfig, *, lane: int, num_steps: int) -> bool:
    if config.calibrate:
        return True

    ratio_index = state.step_index * 2 + lane
    if config.mag_ratios is None or ratio_index >= len(config.mag_ratios):
        current_scale = 1.0
    else:
        current_scale = float(config.mag_ratios[ratio_index])

    retention_step = int(config.retention_ratio * num_steps + 0.5)
    if state.step_index < retention_step:
        return True

    state.accumulated_ratio *= current_scale
    state.accumulated_steps += 1
    state.accumulated_err += abs(1.0 - state.accumulated_ratio)
    if (
        state.previous_residual is not None
        and state.accumulated_err <= config.threshold
        and state.accumulated_steps <= config.max_skip_steps
    ):
        return False

    state.accumulated_ratio = 1.0
    state.accumulated_steps = 0
    state.accumulated_err = 0.0
    return True


def _advance(state: MagCacheState, config: MagCacheConfig, num_steps: int) -> None:
    state.step_index += 1
    if state.step_index < num_steps:
        return
    state.step_index = 0
    state.accumulated_ratio = 1.0
    state.accumulated_steps = 0
    state.accumulated_err = 0.0
    state.previous_residual = None
    state.head_block_input = None
    state.should_compute = True
    state.calibration_ratios = []


class _Kandinsky6MagCacheHeadHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, config: MagCacheConfig, num_steps: int):
        super().__init__()
        self.state_manager = state_manager
        self.config = config
        self.num_steps = num_steps

    @torch.compiler.disable
    def new_forward(self, module: nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        state: MagCacheState = self.state_manager.get_state()
        video, audio = _streams_from_args(args, kwargs)
        state.head_block_input = (video, audio)
        lane = 1 if self.state_manager._current_context in {"uncond", "negative"} else 0
        state.should_compute = _should_compute(state, self.config, lane=lane, num_steps=self.num_steps)

        if not state.should_compute:
            if state.previous_residual is None:
                raise RuntimeError("K6 MagCache requested a skip before a residual was computed")
            return _add_residual(video, audio, state.previous_residual)
        return self.fn_ref.original_forward(*args, **kwargs)

    def reset_state(self, module: nn.Module):
        self.state_manager.reset()
        return module


class _Kandinsky6MagCacheBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, config: MagCacheConfig, num_steps: int, is_tail: bool):
        super().__init__()
        self.state_manager = state_manager
        self.config = config
        self.num_steps = num_steps
        self.is_tail = is_tail

    @torch.compiler.disable
    def new_forward(self, module: nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        state: MagCacheState = self.state_manager.get_state()
        video, audio = _streams_from_args(args, kwargs)

        if not state.should_compute:
            if self.is_tail:
                _advance(state, self.config, self.num_steps)
            return video if audio is None else (video, audio)

        output = self.fn_ref.original_forward(*args, **kwargs)
        if self.is_tail:
            input_video, input_audio = state.head_block_input
            state.previous_residual = _residual(output, input_video, input_audio)
            _advance(state, self.config, self.num_steps)
        return output

    def reset_state(self, module: nn.Module):
        self.state_manager.reset()
        return module


def _apply_kandinsky6_mag_cache(module: nn.Module, config: MagCacheConfig) -> None:
    blocks = getattr(module, "visual_transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
        raise ValueError("K6 MagCache requires a non-empty transformer.visual_transformer_blocks ModuleList")

    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.remove_hook(_HEAD_HOOK, recurse=True)
    registry.remove_hook(_BLOCK_HOOK, recurse=True)
    state_manager = StateManager(MagCacheState, (), {})
    # K6 stores conditional and unconditional coefficients interleaved. The
    # Diffusers pipeline sets separate cache contexts for the two CFG passes.
    num_steps = config.num_inference_steps // 2

    head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
    head_registry.register_hook(
        _Kandinsky6MagCacheHeadHook(state_manager, config, num_steps),
        _HEAD_HOOK,
    )
    for block in blocks[1:-1]:
        block_registry = HookRegistry.check_if_exists_or_initialize(block)
        block_registry.register_hook(
            _Kandinsky6MagCacheBlockHook(state_manager, config, num_steps, is_tail=False),
            _BLOCK_HOOK,
        )
    if len(blocks) > 1:
        tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
        tail_registry.register_hook(
            _Kandinsky6MagCacheBlockHook(state_manager, config, num_steps, is_tail=True),
            _BLOCK_HOOK,
        )


class Kandinsky6MagCacheMixin(CacheMixin):
    """Expose Diffusers' cache API for K6's dual-stream visual blocks."""

    def enable_cache(self, config) -> None:
        if not isinstance(config, MagCacheConfig):
            return super().enable_cache(config)
        if self.is_cache_enabled:
            raise ValueError(f"Caching has already been enabled with {type(self._cache_config)}")
        if config.num_inference_steps % 2 != 0:
            raise ValueError(
                "K6 MagCacheConfig.num_inference_steps must count interleaved CFG forwards (an even number)"
            )
        _apply_kandinsky6_mag_cache(self, config)
        self._cache_config = config

    def disable_cache(self) -> None:
        if not isinstance(self._cache_config, MagCacheConfig):
            return super().disable_cache()
        registry = HookRegistry.check_if_exists_or_initialize(self)
        registry.remove_hook(_HEAD_HOOK, recurse=True)
        registry.remove_hook(_BLOCK_HOOK, recurse=True)
        self._cache_config = None


# Diffusers-style K6 transformer components.
#
# The classes use the same inner-module vocabulary as the Diffusers Kandinsky5
# transformer (for example ``in_layer``, ``modulation``, ``self_attention`` and
# ``feed_forward``). Checkpoint conversion maps the native K6 names to this
# public Diffusers layout.


_MASKED_ATTENTION_BACKENDS = {
    "flash": "flash_varlen",
    "_flash_3": "_flash_varlen_3",
    "sage": "sage_varlen",
    "native": "native",
}


class Kandinsky6AttnProcessor:
    """Diffusers attention processor used by the TI2VA transformer."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, attention_backend=None, parallel_config=None):
        if not hasattr(functional, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0 or newer.")
        self._masked = False
        self._attention_backend = attention_backend
        self._parallel_config = parallel_config

    @property
    def _attention_backend(self):
        return self.__attention_backend

    @_attention_backend.setter
    def _attention_backend(self, backend):
        if self._masked and backend is not None:
            name = getattr(backend, "value", backend).lower()
            backend = _MASKED_ATTENTION_BACKENDS.get(name, name)
            backend = AttentionBackendName(backend)
        self.__attention_backend = backend

    def __call__(
        self,
        attn: Any,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        rotary_emb: Tensor | None = None,
        rotary_emb_kv: Tensor | None = None,
        sparse_params: dict[str, Any] | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        query = attn.to_query(hidden_states)
        if encoder_hidden_states is None:
            key = attn.to_key(hidden_states)
            value = attn.to_value(hidden_states)
        else:
            key = attn.to_key(encoder_hidden_states)
            value = attn.to_value(encoder_hidden_states)

        query = query.reshape(*query.shape[:-1], attn.num_heads, -1)
        key = key.reshape(*key.shape[:-1], attn.num_heads, -1)
        value = value.reshape(*value.shape[:-1], attn.num_heads, -1)
        query = attn.query_norm(query)
        key = attn.key_norm(key)

        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb).to(dtype=query.dtype)
        if rotary_emb_kv is not None:
            key = apply_rotary(key, rotary_emb_kv).to(dtype=key.dtype)

        if sparse_params is not None:
            q = query.transpose(1, 2).contiguous()
            k = key.transpose(1, 2).contiguous()
            v = value.transpose(1, 2).contiguous()
            block_mask = nabla_block_mask(
                q,
                k,
                sparse_params["sta_mask"],
                thr=sparse_params["P"],
            )
            if not _CAN_USE_FLEX_ATTN:
                raise ValueError("Nabla attention requires PyTorch 2.5 or newer")
            output = (
                torch.nn.attention.flex_attention.flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                )
                .transpose(1, 2)
                .contiguous()
            )
        else:
            output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        return attn.out_layer(output.flatten(-2, -1))


class Kandinsky6TimeEmbeddings(nn.Module):
    """Sinusoidal timestep embedding with a K6-compatible parameter layout."""

    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0):
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even")
        self.model_dim = model_dim
        self.max_period = max_period
        self.in_layer = nn.Linear(model_dim, time_dim)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim)

    def forward(self, time: Tensor) -> Tensor:
        embed = get_timestep_embedding(
            time, self.model_dim, flip_sin_to_cos=True, downscale_freq_shift=0, max_period=self.max_period
        )
        h = functional.linear(embed, self.in_layer.weight.float(), self.in_layer.bias.float())
        return functional.linear(self.activation(h), self.out_layer.weight.float(), self.out_layer.bias.float())


class Kandinsky6TextEmbeddings(nn.Module):
    """Text projection and normalization used by K6 text branches."""

    def __init__(self, text_dim: int, model_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.in_layer(x))


class Kandinsky6VisualEmbeddings(nn.Module):
    """Patch projection for ``[B,T,H,W,C]`` visual tokens."""

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, duration, height, width, channels = x.shape
        p_t, p_h, p_w = self.patch_size
        x = (
            x.view(batch, duration // p_t, p_t, height // p_h, p_h, width // p_w, p_w, channels)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


class Kandinsky6Modulation(nn.Module):
    """Zero-initialized AdaLN modulation projection."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = functional.linear(self.activation(x.float()), self.out_layer.weight.float(), self.out_layer.bias.float())
        return out.to(dtype=x.dtype)


class Kandinsky6FeedForward(nn.Module):
    """K6 bias-free GELU feed-forward network."""

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


class Kandinsky6Attention(nn.Module, AttentionModuleMixin):
    """K6 attention with the Diffusers ``set_processor`` contract."""

    _default_processor_cls = Kandinsky6AttnProcessor
    _available_processors = [Kandinsky6AttnProcessor]

    def __init__(
        self,
        num_channels: int,
        head_dim: int,
        kv_dim: int | None = None,
        text_token_padding: bool = False,
        processor: Kandinsky6AttnProcessor | None = None,
    ):
        super().__init__()
        if num_channels % head_dim:
            raise ValueError("num_channels must be divisible by head_dim")
        kv_dim = kv_dim or num_channels
        self.num_heads = num_channels // head_dim
        self.to_query = nn.Linear(num_channels, num_channels)
        self.to_key = nn.Linear(kv_dim, num_channels)
        self.to_value = nn.Linear(kv_dim, num_channels)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels)
        self.text_token_padding = text_token_padding
        self.set_processor(processor or self._default_processor_cls())
        if self.text_token_padding:
            self.processor._masked = True
            self.processor._attention_backend = AttentionBackendName.NATIVE

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rotary_emb: Tensor | None = None,
        sparse_params: dict[str, Any] | None = None,
        rope_q: Tensor | None = None,
        rope_kv: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        # Native K6 blocks pass ``(hidden, rope, mask_or_sparse)`` for
        # self-attention. Keep that call shape while exposing Diffusers'
        # encoder_hidden_states/rotary_emb keyword boundary.
        if attn_mask is None:
            attn_mask = kwargs.get("attention_mask")
        rotary_emb = rope_q if rope_q is not None else rotary_emb
        rotary_emb_kv = rope_kv if rope_kv is not None else rotary_emb
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=rotary_emb,
            rotary_emb_kv=rotary_emb_kv,
            sparse_params=sparse_params,
            attn_mask=attn_mask,
        )


class Kandinsky6OutLayer(nn.Module):
    """Projects visual hidden states back to packed latent patches."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim)

    def forward(self, visual_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        condition_shape = (scale.shape[0],) + (1,) * (visual_embed.ndim - 2) + (scale.shape[-1],)
        x = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale.reshape(condition_shape),
            shift.reshape(condition_shape),
        )
        x = self.out_layer(x)

        batch, duration, height, width = x.shape[:4]
        p_t, p_h, p_w = self.patch_size
        return (
            x.view(batch, duration, height, width, -1, p_t, p_h, p_w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )


class Kandinsky6OutLayerAudio(nn.Module):
    """Projects audio hidden states back to audio latent channels."""

    def __init__(self, model_dim: int, time_dim: int, audio_dim: int):
        super().__init__()
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, audio_dim)

    def forward(self, audio_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        x = apply_scale_shift_norm(self.norm, audio_embed, scale, shift)
        x = self.norm(x)
        return self.out_layer(x)


class Kandinsky6TransformerEncoderBlock(nn.Module):
    """Text self-attention + feed-forward block in Diffusers style."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        text_token_padding: bool = False,
    ):
        super().__init__()
        self.text_modulation = Kandinsky6Modulation(time_dim, model_dim, 6)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky6Attention(model_dim, head_dim, text_token_padding=text_token_padding)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky6FeedForward(model_dim, ff_dim)

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        sa_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(sa_params, 3, dim=-1)
        x = apply_gate_sum(
            x,
            self.self_attention(
                apply_scale_shift_norm(self.self_attention_norm, x, scale, shift),
                rotary_emb=rope,
                attn_mask=attn_mask,
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        return apply_gate_sum(
            x, self.feed_forward(apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)), gate
        )


class Kandinsky6TransformerDecoderBlock(nn.Module):
    """Visual self-attention, text cross-attention, and feed-forward submodules.

    `Kandinsky6FusedTransformerDecoderBlock` uses this class only as a named submodule container
    (`self_attention`/`cross_attention`/`feed_forward` and their norms/modulation), calling those
    submodules directly rather than this class's own `forward`.
    """

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        text_token_padding: bool = False,
    ):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky6Attention(model_dim, head_dim)
        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            kv_dim=model_dim,
            text_token_padding=text_token_padding,
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky6FeedForward(model_dim, ff_dim)


class Kandinsky6FusedTransformerDecoderBlock(nn.Module):
    """Fused K6 video/audio block with cross-modal attention."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        model_dim_a: int,
        time_dim_a: int,
        ff_dim_a: int,
        head_dim_a: int,
        text_token_padding: bool = False,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
    ):
        super().__init__()
        self.videoT = Kandinsky6TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, text_token_padding)
        self.audioT = Kandinsky6TransformerDecoderBlock(
            model_dim_a, time_dim_a, ff_dim_a, head_dim_a, text_token_padding
        )
        self.va_cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            kv_dim=model_dim_a,
        )
        self.av_cross_attention = Kandinsky6Attention(
            model_dim_a,
            head_dim_a,
            kv_dim=model_dim,
        )
        self.va_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim if not cross_gates else model_dim * 2 + model_dim_a,
            1 if cross_gates else 3,
        )
        self.av_modulation = Kandinsky6Modulation(
            time_dim_a,
            model_dim_a if not cross_gates else model_dim_a * 2 + model_dim,
            1 if cross_gates else 3,
        )
        self.va_normalization = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.av_normalization = nn.LayerNorm(model_dim_a, elementwise_affine=False)
        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def forward(
        self,
        vis: Tensor | None,
        aud: Tensor | None,
        text_v: Tensor,
        text_a: Tensor,
        time_embed: tuple[Tensor, Tensor],
        vis_rope: Tensor | None,
        aud_rope: Tensor | None,
        sparse_params: dict | None,
        attn_mask=None,
    ) -> tuple[Tensor | None, Tensor | None]:
        t_v, t_a = time_embed
        if vis is not None:
            sa_p, ca_p, ff_p = torch.chunk(self.videoT.visual_modulation(t_v), 3, dim=-1)
            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            vis = apply_gate_sum(
                vis,
                self.videoT.self_attention(
                    apply_scale_shift_norm(self.videoT.self_attention_norm, vis, scale, shift),
                    rotary_emb=vis_rope,
                    sparse_params=sparse_params,
                ),
                gate,
            ).type_as(vis)
            shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
            vis_pre_ca = apply_scale_shift_norm(self.videoT.cross_attention_norm, vis, scale, shift)
            vis_out_t = self.videoT.cross_attention(
                vis_pre_ca,
                encoder_hidden_states=text_v,
                attn_mask=attn_mask,
            )

        if aud is not None:
            sa_p, ca_p, ff_p_a = torch.chunk(self.audioT.visual_modulation(t_a), 3, dim=-1)
            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            aud = apply_gate_sum(
                aud,
                self.audioT.self_attention(
                    apply_scale_shift_norm(self.audioT.self_attention_norm, aud, scale, shift),
                    rotary_emb=aud_rope,
                ),
                gate,
            ).type_as(aud)
            shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
            aud_pre_ca = apply_scale_shift_norm(self.audioT.cross_attention_norm, aud, scale, shift)
            aud_out_t = self.audioT.cross_attention(
                aud_pre_ca,
                encoder_hidden_states=text_a,
                attn_mask=attn_mask,
            )
            aud = apply_gate_sum(aud, aud_out_t, gate_a).type_as(aud)

            if vis is not None:
                t_va_mod = t_a if not self.fix_modulation else t_v
                t_av_mod = t_v if not self.fix_modulation else t_a
                va_params = self.va_modulation(t_va_mod)
                av_params = self.av_modulation(t_av_mod)
                if self.cross_gates:
                    va_shift, va_scale, va_gate = torch.split(
                        va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1
                    )
                    av_shift, av_scale, av_gate = torch.split(
                        av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1
                    )
                else:
                    va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
                    av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)
                vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)
                vis_for_va = apply_scale_shift_norm(self.va_normalization, vis, va_scale, va_shift)
                aud_for_av = apply_scale_shift_norm(self.av_normalization, aud, av_scale, av_shift)
                rq_v = vis_rope if self.ca_rope else None
                rk_a = aud_rope if self.ca_rope else None
                vis_from_aud = self.va_cross_attention(
                    vis_for_va,
                    encoder_hidden_states=aud_pre_ca,
                    rope_q=rq_v,
                    rope_kv=rk_a,
                )
                aud_from_vis = self.av_cross_attention(
                    aud_for_av,
                    encoder_hidden_states=vis_pre_ca,
                    rope_q=rk_a,
                    rope_kv=rq_v,
                )
                vis = apply_gate_sum(vis, vis_from_aud, va_gate if not self.cross_gates else av_gate).type_as(vis)
                aud = apply_gate_sum(aud, aud_from_vis, av_gate if not self.cross_gates else va_gate).type_as(aud)
        elif vis is not None:
            vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)

        if vis is not None:
            shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
            vis = apply_gate_sum(
                vis,
                self.videoT.feed_forward(apply_scale_shift_norm(self.videoT.feed_forward_norm, vis, scale, shift)),
                gate,
            ).type_as(vis)
        if aud is not None:
            shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
            aud = apply_gate_sum(
                aud,
                self.audioT.feed_forward(apply_scale_shift_norm(self.audioT.feed_forward_norm, aud, scale, shift)),
                gate,
            ).type_as(aud)
        return vis, aud


class Kandinsky6RoPE1D(RoPE1D):
    """Diffusers-exported name for the K6 one-dimensional RoPE."""


class Kandinsky6RoPE3D(RoPE3D):
    """Diffusers-exported name for the K6 three-dimensional RoPE."""


class Kandinsky6Transformer3DModel(
    Kandinsky6MagCacheMixin,
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """Kandinsky 6 multimodal transformer for text/image-to-video-and-audio generation.

    Video and audio are denoised together through fused, cross-modal transformer blocks, each
    conditioned on its own text branch (Qwen2.5-VL tokens + CLIP pooled embedding). Passing only
    `x_video` or only `x_audio` denoises a single modality while still running through the fused
    block's per-modality self-attention, cross-attention, and feed-forward stages. Inputs and
    outputs use packed token layouts compatible with the native K6 model.

    Args:
        in_visual_dim (`int`, *optional*, defaults to 16): Number of input video latent channels.
        out_visual_dim (`int`, *optional*, defaults to 16): Number of output video latent channels.
        in_text_dim (`int`, *optional*, defaults to 3584): Text token embedding dimension.
        in_text_dim2 (`int`, *optional*, defaults to 768): Pooled text embedding dimension.
        time_dim (`int`, *optional*, defaults to 1024): Time embedding dimension.
        patch_size (`tuple[int, int, int]`, *optional*, defaults to ``(1, 2, 2)``): Video patch size.
        model_dim (`int`, *optional*, defaults to 4096): Video transformer hidden dimension.
        ff_dim (`int`, *optional*, defaults to 16384): Video feed-forward hidden dimension.
        num_text_blocks (`int`, *optional*, defaults to 4): Number of text blocks per modality.
        num_visual_blocks (`int`, *optional*, defaults to 60): Number of fused video/audio blocks.
        axes_dims (`tuple[int, int, int]`, *optional*, defaults to ``(32, 48, 48)``): RoPE dimensions for video.
        visual_cond (`bool`, *optional*, defaults to True): Whether video conditioning channels are present.
        in_audio_dim (`int`, *optional*, defaults to 20): Number of input audio latent channels.
        out_audio_dim (`int`, *optional*, defaults to 20): Number of output audio latent channels.
        model_dim_a (`int`, *optional*): Audio transformer hidden dimension. Defaults to `model_dim`.
        time_dim_a (`int`, *optional*): Audio time embedding dimension. Defaults to `time_dim`.
        ff_dim_a (`int`, *optional*): Audio feed-forward hidden dimension. Defaults to `ff_dim`.
        axes_dims_a (`tuple[int, int, int]`, *optional*): Audio RoPE dimensions. Defaults to `axes_dims`.
        audio_freqs_scaling (`float`, *optional*, defaults to 1.0): Audio RoPE frequency scaling.
        text_token_padding (`bool`, *optional*, defaults to False): Whether text sequences are padded.
        scale_factor (`tuple[float]`, *optional*): Per-axis RoPE frequency scaling.
        ca_rope (`bool`, *optional*, defaults to False): Whether to use cross-modal audio RoPE.
        cross_gates (`bool`, *optional*, defaults to False): Whether to use cross-modal residual gates.
        fix_modulation (`bool`, *optional*, defaults to False): Whether to use the fixed modulation variant.
        visual_token_type_num_embeddings (`int`, *optional*, defaults to 0): Number of visual token type embeddings.
        magcache (`dict`, *optional*): MagCache configuration.
    """

    _repeated_blocks = [
        "Kandinsky6TransformerEncoderBlock",
        "Kandinsky6FusedTransformerDecoderBlock",
    ]
    _no_split_modules = _repeated_blocks
    _keep_in_fp32_modules = ["time_embeddings", "modulation"]
    _supports_gradient_checkpointing = True
    # `Kandinsky6TimeEmbeddings`/`Kandinsky6Modulation` hand their `nn.Linear` weight/bias straight to
    # `functional.linear` (upcast to fp32 first) rather than calling the submodule, so the AdaLN math stays
    # fp32 even after a later blanket `.to(bfloat16)` cast (`_keep_in_fp32_modules` only protects the
    # `from_pretrained(dtype=...)` load, not a subsequent cast). Leaf-level offload hooks only fire on an
    # actual submodule call, so they never see these weights — the same class of gap as
    # `HunyuanDiTAttentionPool` (see testing.md), which opts out of group offloading for the same reason.
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 1024,
        patch_size: tuple = (1, 2, 2),
        model_dim: int = 4096,
        ff_dim: int = 16384,
        num_text_blocks: int = 4,
        num_visual_blocks: int = 60,
        axes_dims: tuple = (32, 48, 48),
        visual_cond: bool = True,
        in_audio_dim: int = 20,
        out_audio_dim: int = 20,
        model_dim_a: int | None = None,
        time_dim_a: int | None = None,
        ff_dim_a: int | None = None,
        axes_dims_a: tuple | None = None,
        audio_freqs_scaling: float = 1.0,
        text_token_padding: bool = False,
        scale_factor: tuple | list[float] = (1.0, 2.0, 2.0),
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        visual_token_type_num_embeddings: int = 0,
        magcache: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.in_visual_dim = in_visual_dim
        self.in_audio_dim = in_audio_dim
        self.text_token_padding = text_token_padding
        self.scale_factor = tuple(float(value) for value in scale_factor)
        self.visual_token_type_num_embeddings = int(visual_token_type_num_embeddings or 0)
        head_dim = sum(axes_dims)
        model_dim_a = model_dim_a or model_dim
        time_dim_a = time_dim_a or time_dim
        ff_dim_a = ff_dim_a or ff_dim
        axes_dims_a = axes_dims_a or axes_dims
        head_dim_a = sum(axes_dims_a)

        vis_in_dim = (2 * in_visual_dim + 1) if visual_cond else in_visual_dim
        self.visual_embeddings = Kandinsky6VisualEmbeddings(vis_in_dim, model_dim, patch_size)
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(self.visual_token_type_num_embeddings, model_dim)
        self.visual_rope_embeddings = Kandinsky6RoPE3D(axes_dims)
        self.out_layer = Kandinsky6OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

        self.audio_embeddings = Kandinsky6TextEmbeddings(in_audio_dim, model_dim_a)
        self.audio_rope_embeddings = Kandinsky6RoPE1D(head_dim_a, freqs_scaling=audio_freqs_scaling)
        self.audio_out_layer = Kandinsky6OutLayerAudio(model_dim_a, time_dim_a, out_audio_dim)
        for prefix, md, td, fd, hd in (
            ("video", model_dim, time_dim, ff_dim, head_dim),
            ("audio", model_dim_a, time_dim_a, ff_dim_a, head_dim_a),
        ):
            setattr(self, f"{prefix}_time_embeddings", Kandinsky6TimeEmbeddings(md, td))
            setattr(self, f"{prefix}_text_embeddings", Kandinsky6TextEmbeddings(in_text_dim, md))
            setattr(self, f"{prefix}_pooled_text_embeddings", Kandinsky6TextEmbeddings(in_text_dim2, td))
            setattr(self, f"{prefix}_text_rope_embeddings", Kandinsky6RoPE1D(hd))
            setattr(
                self,
                f"{prefix}_text_transformer_blocks",
                nn.ModuleList(
                    [
                        Kandinsky6TransformerEncoderBlock(md, td, fd, hd, text_token_padding)
                        for _ in range(num_text_blocks)
                    ]
                ),
            )
        self.visual_transformer_blocks = nn.ModuleList(
            [
                Kandinsky6FusedTransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    model_dim_a,
                    time_dim_a,
                    ff_dim_a,
                    head_dim_a,
                    text_token_padding,
                    ca_rope,
                    cross_gates,
                    fix_modulation,
                )
                for _ in range(num_visual_blocks)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        x_video: Tensor | None = None,
        x_audio: Tensor | None = None,
        text_embed: Tensor | None = None,
        pooled_text_embed: Tensor | None = None,
        time: Tensor | None = None,
        visual_rope: Tensor | None = None,
        audio_rope: Tensor | None = None,
        video_text_rope: Tensor | None = None,
        audio_text_rope: Tensor | None = None,
        sparse_params: dict | None = None,
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Tensor] | Transformer2DModelOutput:
        """Run the K6 transformer on video, audio, or fused video/audio latents.

        Args:
            x_video (`torch.Tensor`, *optional*): Packed video latent tokens.
            x_audio (`torch.Tensor`, *optional*): Packed audio latent tokens.
            text_embed (`torch.Tensor`): Text token embeddings, shared by the video and audio text branches.
            pooled_text_embed (`torch.Tensor`): Pooled text embedding, shared by the video and audio text branches.
            time (`torch.Tensor`): Diffusion timestep, shared by the video and audio branches.
            visual_rope (`torch.Tensor`, *optional*): Video rotary position embeddings. Required with `x_video`.
            audio_rope (`torch.Tensor`, *optional*): Audio rotary position embeddings. Required with `x_audio`.
            video_text_rope (`torch.Tensor`): Rotary position embeddings for the video text branch.
            audio_text_rope (`torch.Tensor`): Rotary position embeddings for the audio text branch.
            sparse_params (`dict`, *optional*): NABLA sparse-attention configuration for the video self-attention.
            attention_mask (`torch.Tensor`, *optional*): Text attention mask.
            visual_token_type_ids (`torch.Tensor`, *optional*): Video token type IDs.
            return_dict (`bool`, *optional*, defaults to False): Whether to return a
                [`Transformer2DModelOutput`] instead of tensors.

        Returns:
            `torch.Tensor`, `tuple[torch.Tensor, torch.Tensor]`, or
            [`Transformer2DModelOutput`]: The denoised single modality, `(video, audio)` when both `x_video`
            and `x_audio` are given, or a model output object.
        """
        if x_video is None and x_audio is None:
            raise ValueError("at least one of `x_video`, `x_audio` must be provided")
        if text_embed is None or pooled_text_embed is None or time is None:
            raise ValueError("`text_embed`, `pooled_text_embed`, and `time` are required")
        if video_text_rope is None or audio_text_rope is None:
            raise ValueError("`video_text_rope` and `audio_text_rope` are required")

        attn_mask = attention_mask
        if attn_mask is not None and attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)
        checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing

        # 1. Encode text tokens through the video and audio text branches
        video_text_embed = self.video_text_embeddings(text_embed)
        video_temb = self.video_time_embeddings(time) + self.video_pooled_text_embeddings(pooled_text_embed)
        for block in self.video_text_transformer_blocks:
            args = (video_text_embed, video_temb, video_text_rope, attn_mask)
            video_text_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

        audio_text_embed = self.audio_text_embeddings(text_embed)
        audio_temb = self.audio_time_embeddings(time) + self.audio_pooled_text_embeddings(pooled_text_embed)
        for block in self.audio_text_transformer_blocks:
            args = (audio_text_embed, audio_temb, audio_text_rope, attn_mask)
            audio_text_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

        # 2. Patchify the video latents and embed the audio latents (only the modalities that were passed)
        visual_embed, visual_shape = None, None
        to_fractal = bool(sparse_params and sparse_params.get("to_fractal"))
        if x_video is not None:
            if x_video.ndim == 4:
                x_video = x_video.unsqueeze(0)
            visual_embed = self.visual_embeddings(x_video)
            if visual_token_type_ids is not None and hasattr(self, "visual_token_type_embeddings"):
                token_type_ids = (
                    visual_token_type_ids.unsqueeze(0) if visual_token_type_ids.ndim == 1 else visual_token_type_ids
                )
                token_types = self.visual_token_type_embeddings(token_type_ids.to(device=visual_embed.device))
                visual_embed = visual_embed + token_types[:, :, None, None, :]
            visual_shape = visual_embed.shape[-4:-1]
            if to_fractal:
                visual_embed = _local_patch(visual_embed, visual_shape, (1, 8, 8), dim=1).flatten(1, 2)
                visual_rope = _local_patch(visual_rope, visual_shape, (1, 8, 8), dim=0).flatten(0, 1)
            else:
                visual_embed = visual_embed.flatten(1, 3)
                visual_rope = visual_rope.flatten(0, 2)

        audio_embed = None
        if x_audio is not None:
            if x_audio.ndim == 2:
                x_audio = x_audio.unsqueeze(0)
            audio_embed = self.audio_embeddings(x_audio)

        # 3. Run the fused video/audio transformer blocks (a block no-ops on whichever modality is absent)
        for block in self.visual_transformer_blocks:
            args = (
                visual_embed,
                audio_embed,
                video_text_embed,
                audio_text_embed,
                (video_temb, audio_temb),
                visual_rope,
                audio_rope,
                sparse_params,
                attn_mask,
            )
            visual_embed, audio_embed = self._gradient_checkpointing_func(block, *args) if checkpoint else block(*args)

        # 4. Project back to patch/latent space
        video_out = None
        if visual_embed is not None:
            if to_fractal:
                visual_embed = _local_merge(
                    visual_embed.reshape(visual_embed.shape[0], -1, 64, visual_embed.shape[-1]),
                    visual_shape,
                    (1, 8, 8),
                    dim=1,
                )
            else:
                visual_embed = visual_embed.reshape(-1, *visual_shape, visual_embed.shape[-1])
            video_out = self.out_layer(visual_embed, video_temb)
        audio_out = self.audio_out_layer(audio_embed, audio_temb) if audio_embed is not None else None

        if video_out is not None and audio_out is not None:
            result = (video_out, audio_out)
        else:
            result = video_out if video_out is not None else audio_out
        return Transformer2DModelOutput(sample=result) if return_dict else result


__all__ = [
    "Kandinsky6Transformer3DModel",
    "Kandinsky6TransformerEncoderBlock",
    "Kandinsky6TransformerDecoderBlock",
    "Kandinsky6FusedTransformerDecoderBlock",
]
