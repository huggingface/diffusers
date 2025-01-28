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

from ..utils.logging import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


class CacheMixin:
    r"""
    A class for enable/disabling caching techniques on diffusion models.

    Supported caching techniques:
        - [Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588)
        - [FasterCache](https://huggingface.co/papers/2410.19355)
    """

    _cache_config = None

    @property
    def is_cache_enabled(self) -> bool:
        return self._cache_config is not None

    def enable_cache(self, config) -> None:
        r"""
        Enable caching techniques on the model.

        Args:
            config (`Union[PyramidAttentionBroadcastConfig]`):
                The configuration for applying the caching technique. Currently supported caching techniques are:
                    - [`~hooks.PyramidAttentionBroadcastConfig`]

        Example:

        ```python
        >>> import torch
        >>> from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> config = PyramidAttentionBroadcastConfig(
        ...     spatial_attention_block_skip_range=2,
        ...     spatial_attention_timestep_skip_range=(100, 800),
        ...     current_timestep_callback=lambda: pipe.current_timestep,
        ... )
        >>> pipe.transformer.enable_cache(config)
        ```
        """

        from ..hooks import (
            FasterCacheConfig,
            PyramidAttentionBroadcastConfig,
            apply_faster_cache,
            apply_pyramid_attention_broadcast,
        )

        if self.is_cache_enabled:
            raise ValueError(
                f"Caching has already been enabled with {type(self._cache_config)}. To apply a new caching technique, please disable the existing one first."
            )

        if isinstance(config, PyramidAttentionBroadcastConfig):
            apply_pyramid_attention_broadcast(self, config)
        elif isinstance(config, FasterCacheConfig):
            apply_faster_cache(self, config)
        else:
            raise ValueError(f"Cache config {type(config)} is not supported.")

        self._cache_config = config

    def disable_cache(self) -> None:
        from ..hooks import FasterCacheConfig, HookRegistry, PyramidAttentionBroadcastConfig
        from ..hooks.faster_cache import _FASTER_CACHE_BLOCK_HOOK, _FASTER_CACHE_DENOISER_HOOK
        from ..hooks.pyramid_attention_broadcast import _PYRAMID_ATTENTION_BROADCAST_HOOK

        if self._cache_config is None:
            logger.warning("Caching techniques have not been enabled, so there's nothing to disable.")
            return

        if isinstance(self._cache_config, PyramidAttentionBroadcastConfig):
            registry = HookRegistry.check_if_exists_or_initialize(self)
            registry.remove_hook(_PYRAMID_ATTENTION_BROADCAST_HOOK, recurse=True)
        elif isinstance(self._cache_config, FasterCacheConfig):
            registry = HookRegistry.check_if_exists_or_initialize(self)
            registry.remove_hook(_FASTER_CACHE_DENOISER_HOOK, recurse=True)
            registry.remove_hook(_FASTER_CACHE_BLOCK_HOOK, recurse=True)
        else:
            raise ValueError(f"Cache config {type(self._cache_config)} is not supported.")

        self._cache_config = None

    def _reset_stateful_cache(self, recurse: bool = True) -> None:
        from ..hooks import HookRegistry

        HookRegistry.check_if_exists_or_initialize(self).reset_stateful_hooks(recurse=recurse)
