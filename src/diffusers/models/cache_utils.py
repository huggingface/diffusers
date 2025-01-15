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

from typing import Union

from ..hooks import HookRegistry, PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast


CacheConfig = Union[PyramidAttentionBroadcastConfig]


class CacheMixin:
    _cache_config: CacheConfig = None

    @property
    def is_cache_enabled(self) -> bool:
        return self._cache_config is not None

    def enable_cache(self, config: CacheConfig) -> None:
        if isinstance(config, PyramidAttentionBroadcastConfig):
            apply_pyramid_attention_broadcast(self.model, config)
        else:
            raise ValueError(f"Cache config {type(config)} is not supported.")
        self._cache_config = config

    def disable_cache(self) -> None:
        if self._cache_config is None:
            raise ValueError("Caching techniques have not been enabled.")
        if isinstance(self._cache_config, PyramidAttentionBroadcastConfig):
            registry = HookRegistry.check_if_exists_or_initialize(self)
            registry.remove_hook("pyramid_attention_broadcast")
        else:
            raise ValueError(f"Cache config {type(self._cache_config)} is not supported.")
        self._cache_config = None

    def reset_stateful_cache(self, recurse: bool = True) -> None:
        HookRegistry.check_if_exists_or_initialize(self).reset_stateful_hooks(recurse=recurse)
