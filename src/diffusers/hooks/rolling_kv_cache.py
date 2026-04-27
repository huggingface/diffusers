# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.attention_dispatch import dispatch_attention_fn
from ..utils import logging
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)


_ROLLING_KV_CACHE_HOOK = "rolling_kv_cache"
_ROLLING_KV_WRITE_MODES = {"append", "overwrite"}
_TESTED_ATTENTION_CLASSES = frozenset({"WanAttention"})


@dataclass
class RollingKVCacheConfig:
    r"""Configuration for rolling self-attention KV caching during autoregressive inference.

    Args:
        window_size (`int`, defaults to `-1`):
            Maximum number of cached self-attention tokens to keep. Set to `-1` to keep the full prefix.
    """

    window_size: int = -1


class RollingKVAttentionProcessor:
    r"""Default attention preprocessor used by the rolling KV cache hook.

    The defaults target Wan-style self-attention modules. The default rotary embedding path mirrors
    WanAttnProcessor while staying local to avoid a hook dependency on Wan private helpers; override
    it for other layouts. To support a model with a different attention layout — most often a
    different rotary embedding form — subclass and override the relevant method, then pass the
    instance via `apply_rolling_kv_cache(..., attention_processor=...)`.
    """

    def prepare_qkv(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = self.apply_rotary_emb(query, *rotary_emb)
            key = self.apply_rotary_emb(key, *rotary_emb)

        return query, key, value

    def apply_rotary_emb(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_complex = torch.view_as_complex(
            hidden_states.to(torch.float64).reshape(*hidden_states.shape[:-1], -1, 2)
        )
        freqs_complex = torch.complex(
            freqs_cos[..., 0::2].to(torch.float64),
            freqs_sin[..., 1::2].to(torch.float64),
        )
        out = torch.view_as_real(hidden_states_complex * freqs_complex).flatten(-2)
        return out.type_as(hidden_states)

    def post_attention(
        self,
        attn: torch.nn.Module,
        attn_output: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        out = attn_output.flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    def get_attention_backend(self, attn: torch.nn.Module):
        processor = getattr(attn, "processor", None)
        return getattr(processor, "_attention_backend", None)


class RollingKVCacheState(BaseState):
    r"""Shared state controlling how the rolling KV cache is updated."""

    def __init__(self):
        self.should_update_cache = True
        self.write_mode = "append"
        self.absolute_token_offset: int | None = None

    def configure_cache_write(self, write_mode: str = "append", absolute_token_offset: int | None = None) -> None:
        if write_mode not in _ROLLING_KV_WRITE_MODES:
            raise ValueError(
                f"`write_mode` must be one of {sorted(_ROLLING_KV_WRITE_MODES)}, but received {write_mode!r}."
            )
        if write_mode == "append" and absolute_token_offset is not None:
            raise ValueError("`absolute_token_offset` is only supported with `write_mode='overwrite'`.")
        if write_mode == "overwrite" and absolute_token_offset is None:
            raise ValueError("`absolute_token_offset` must be provided when `write_mode='overwrite'`.")
        if absolute_token_offset is not None and absolute_token_offset < 0:
            raise ValueError("`absolute_token_offset` must be >= 0.")

        self.write_mode = write_mode
        self.absolute_token_offset = absolute_token_offset

    def clear_cache_write(self) -> None:
        self.write_mode = "append"
        self.absolute_token_offset = None

    def reset(self):
        self.should_update_cache = True
        self.clear_cache_write()


class RollingKVCacheBlockState(BaseState):
    r"""Per-attention-block self-attention cache state."""

    def __init__(self):
        self.cached_key: torch.Tensor | None = None
        self.cached_value: torch.Tensor | None = None
        self.cache_start_token_offset = 0

    def reset(self):
        self.cached_key = None
        self.cached_value = None
        self.cache_start_token_offset = 0


def _ensure_state(state_manager: StateManager):
    if state_manager._current_context is None:
        state_manager.set_context("inference")
    return state_manager.get_state()


def _slice_cache_for_overwrite(
    block_state: RollingKVCacheBlockState,
    absolute_token_offset: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    cached_key = block_state.cached_key
    cached_value = block_state.cached_value
    cache_start = block_state.cache_start_token_offset

    if cached_key is None:
        return None, None, absolute_token_offset

    cache_end = cache_start + cached_key.shape[1]
    if absolute_token_offset > cache_end:
        raise ValueError(
            "`absolute_token_offset` points beyond the retained cache prefix. Reset the cache or prefill the "
            "missing chunks before appending new ones."
        )
    if absolute_token_offset < cache_start:
        return None, None, absolute_token_offset

    prefix_length = absolute_token_offset - cache_start
    return cached_key[:, :prefix_length], cached_value[:, :prefix_length], cache_start


def _trim_cache_to_window(
    key: torch.Tensor,
    value: torch.Tensor,
    cache_start_token_offset: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if window_size > 0 and key.shape[1] > window_size:
        # TODO: support pinned sink frames when rolling the cache window.
        trim = key.shape[1] - window_size
        key = key[:, trim:]
        value = value[:, trim:]
        cache_start_token_offset += trim

    return key.detach(), value.detach(), cache_start_token_offset


def _is_self_attention_module(module: torch.nn.Module) -> bool:
    if getattr(module, "is_cross_attention", False):
        return False

    required_attrs = ("to_q", "to_k", "to_v", "to_out", "heads", "norm_q", "norm_k")
    return all(hasattr(module, attr) for attr in required_attrs)


class RollingKVCacheHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        config: RollingKVCacheConfig,
        state_manager: StateManager,
        block_state_manager: StateManager,
        attention_processor: RollingKVAttentionProcessor,
    ):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.block_state_manager = block_state_manager
        self.attention_processor = attention_processor

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("Rolling KV cache only supports self-attention modules.")

        shared_state: RollingKVCacheState = _ensure_state(self.state_manager)
        block_state: RollingKVCacheBlockState = _ensure_state(self.block_state_manager)
        proc = self.attention_processor

        query, key, value = proc.prepare_qkv(module, hidden_states, rotary_emb)

        if shared_state.write_mode == "overwrite":
            cached_key, cached_value, prefix_start = _slice_cache_for_overwrite(
                block_state, shared_state.absolute_token_offset
            )
        else:
            cached_key = block_state.cached_key
            cached_value = block_state.cached_value
            prefix_start = block_state.cache_start_token_offset

        if cached_key is not None:
            if cached_key.shape[0] != key.shape[0]:
                raise ValueError(
                    f"Rolling KV cache batch size mismatch (cached={cached_key.shape[0]}, current={key.shape[0]}). "
                    "Use `cache_context` to isolate cond/uncond passes or reset the cache before changing batch size."
                )
            full_key = torch.cat([cached_key, key], dim=1)
            full_value = torch.cat([cached_value, value], dim=1)
        else:
            full_key = key
            full_value = value

        if shared_state.should_update_cache:
            (
                block_state.cached_key,
                block_state.cached_value,
                block_state.cache_start_token_offset,
            ) = _trim_cache_to_window(
                full_key,
                full_value,
                prefix_start,
                self.config.window_size,
            )

        attn_output = dispatch_attention_fn(
            query,
            full_key,
            full_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=proc.get_attention_backend(module),
        )
        return proc.post_attention(module, attn_output, query)

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        self.block_state_manager.reset()
        return module


def apply_rolling_kv_cache(
    module: torch.nn.Module,
    config: RollingKVCacheConfig | None = None,
    attention_processor: RollingKVAttentionProcessor | None = None,
) -> None:
    r"""Apply rolling KV cache hooks to compatible self-attention modules.

    The default `attention_processor` targets Wan-style attention modules. Pass a custom
    `RollingKVAttentionProcessor` subclass for models with a different rotary embedding form
    or projection layout.
    """
    if config is None:
        config = RollingKVCacheConfig()
    if attention_processor is None:
        attention_processor = RollingKVAttentionProcessor()

    state_manager = StateManager(RollingKVCacheState)
    HookRegistry.check_if_exists_or_initialize(module)

    warned_classes: set[str] = set()
    for submodule in module.modules():
        if not _is_self_attention_module(submodule):
            continue

        cls_name = type(submodule).__name__
        if cls_name not in _TESTED_ATTENTION_CLASSES and cls_name not in warned_classes:
            warned_classes.add(cls_name)
            logger.warning(
                "apply_rolling_kv_cache: attaching to '%s' which is untested. The default "
                "RollingKVAttentionProcessor targets Wan-style attention; if outputs look wrong, "
                "subclass it (in particular `apply_rotary_emb`) and pass via `attention_processor=`.",
                cls_name,
            )

        block_state_manager = StateManager(RollingKVCacheBlockState)
        hook = RollingKVCacheHook(config, state_manager, block_state_manager, attention_processor)
        registry = HookRegistry.check_if_exists_or_initialize(submodule)
        registry.register_hook(hook, _ROLLING_KV_CACHE_HOOK)


def get_rolling_kv_cache_state(module: torch.nn.Module) -> RollingKVCacheState | None:
    r"""Return the shared rolling KV cache state for a hooked module."""
    for submodule in module.modules():
        if not hasattr(submodule, "_diffusers_hook"):
            continue

        hook = submodule._diffusers_hook.get_hook(_ROLLING_KV_CACHE_HOOK)
        if hook is not None:
            return _ensure_state(hook.state_manager)

    return None
