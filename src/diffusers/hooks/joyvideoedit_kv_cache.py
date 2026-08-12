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

from dataclasses import dataclass
from typing import Dict

import torch

from .hooks import BaseState, HookRegistry, ModelHook, StateManager


_JOYVIDEOEDIT_KV_CACHE_HOOK = "joyvideoedit_kv_cache"


@dataclass
class JoyVideoEditKVCacheConfig:
    """Enable the chunk-wise streaming KV cache used by `JoyVideoEditTransformer3DModel`.

    Chunks of clean video are fed through once (`kv_cache_mode="store"`) to populate a per-layer cache, then reused as
    attention context for later chunks (`kv_cache_mode="reuse"`) without recomputing their key/value projections.
    """

    pass


class JoyVideoEditKVCacheState(BaseState):
    """Holds the per-chunk-per-layer KV cache."""

    def __init__(self):
        self.chunk_cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    def reset(self):
        self.chunk_cache.clear()


class JoyVideoEditKVCacheHook(ModelHook):
    """Routes `JoyVideoEditTransformer3DModel`'s KV-cache reads/writes through a `StateManager`-managed state.

    The hook owns the cache state and ensures a context is active before `forward` runs. Cache selection, assembly, and
    eviction are handled by the model.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        return self.fn_ref.original_forward(*args, **kwargs)

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


def apply_joyvideoedit_kv_cache(module: torch.nn.Module, config: JoyVideoEditKVCacheConfig) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(module)
    state_manager = StateManager(JoyVideoEditKVCacheState)
    hook = JoyVideoEditKVCacheHook(state_manager)
    registry.register_hook(hook, _JOYVIDEOEDIT_KV_CACHE_HOOK)
