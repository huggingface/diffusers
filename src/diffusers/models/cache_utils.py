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

from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from ..utils.logging import get_logger
from ._modeling_parallel import ParallelConfig, gather_size_by_comm


logger = get_logger(__name__)  # pylint: disable=invalid-name


def apply_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_cache,
    cache_mode: str | None,
    cache_write_slice: slice | None,
    *,
    attention_mask: Any | None = None,
    parallel_config: ParallelConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    if layer_cache is None:
        return key, value, attention_mask

    cp_config = None if parallel_config is None else parallel_config.context_parallel_config
    if cp_config is not None and cp_config.ring_degree > 1:
        raise NotImplementedError("KV caching currently supports Ulysses context parallelism only.")

    if cache_mode == "extract" and cache_write_slice is not None:
        if cp_config is None:
            # `clone()`, not `contiguous()`: at batch size 1 the prefix slice already counts as contiguous
            # (size-1 dims are ignored), so `contiguous()` returns the same view and the cache would pin the
            # whole prefill K/V for every step of the denoising loop.
            layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
        else:
            group = cp_config._ulysses_mesh.get_group()
            rank = dist.get_rank(group)
            world_size = dist.get_world_size(group)
            local_sizes = gather_size_by_comm(key.shape[1], group)
            cache_start, cache_stop, cache_step = cache_write_slice.indices(sum(local_sizes))
            if cache_step != 1:
                raise ValueError("Context-parallel KV caching requires a slice with step 1.")
            cache_stop = max(cache_start, cache_stop)
            cache_size = cache_stop - cache_start
            if not cp_config.ulysses_anything and cache_size % world_size != 0:
                raise ValueError(
                    "The cached sequence length must be divisible by the Ulysses degree. Enable "
                    "`ulysses_anything=True` to cache an uneven sequence."
                )
            offsets = [sum(local_sizes[:index]) for index in range(world_size)]
            starts = [min(max(cache_start - offset, 0), size) for offset, size in zip(offsets, local_sizes)]
            stops = [min(max(cache_stop - offset, 0), size) for offset, size in zip(offsets, local_sizes)]
            cache_sizes = [stop - start for start, stop in zip(starts, stops)]
            max_size = max(cache_sizes)
            cached = []
            for tensor in (key, value):
                local = tensor[:, starts[rank] : stops[rank]].contiguous()
                if local.shape[1] < max_size:
                    padding = tensor.new_zeros(tensor.shape[0], max_size - local.shape[1], *tensor.shape[2:])
                    local = torch.cat([local, padding], dim=1)
                gathered = [torch.empty_like(local) for _ in range(world_size)]
                dist.all_gather(gathered, local, group=group)
                full_cache = torch.cat([part[:, :size] for part, size in zip(gathered, cache_sizes)], dim=1)
                cached.append(torch.tensor_split(full_cache, world_size, dim=1)[rank].clone())
            layer_cache.store(*cached)
    elif cache_mode == "cached":
        cached_key, cached_value = layer_cache.get()
        if cp_config is not None and attention_mask is not None:
            group = cp_config._ulysses_mesh.get_group()
            rank = dist.get_rank(group)
            world_size = dist.get_world_size(group)
            target_len = sum(gather_size_by_comm(key.shape[1], group))
            prefix_len = attention_mask.shape[-1] - target_len
            prefix_masks = torch.tensor_split(attention_mask[..., :prefix_len], world_size, dim=-1)
            target_masks = torch.tensor_split(attention_mask[..., prefix_len:], world_size, dim=-1)
            attention_mask = torch.cat([prefix_masks[rank], target_masks[rank]], dim=-1)
        key = torch.cat([cached_key, key], dim=1)
        value = torch.cat([cached_value, value], dim=1)

    return key, value, attention_mask


class CacheMixin:
    r"""
    A class for enable/disabling caching techniques on diffusion models.

    Supported caching techniques:
        - [Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588)
        - [FasterCache](https://huggingface.co/papers/2410.19355)
        - [FirstBlockCache](https://github.com/chengzeyi/ParaAttention/blob/7a266123671b55e7e5a2fe9af3121f07a36afc78/README.md#first-block-cache-our-dynamic-caching)
        - [SeaCache](https://huggingface.co/papers/2602.18993)
    """

    _cache_config = None

    @property
    def is_cache_enabled(self) -> bool:
        return self._cache_config is not None

    def enable_cache(self, config) -> None:
        r"""
        Enable caching techniques on the model.

        Args:
            config (`PyramidAttentionBroadcastConfig | FasterCacheConfig | FirstBlockCacheConfig | SeaCacheConfig | TextKVCacheConfig`):
                The configuration for applying the caching technique. Currently supported caching techniques are:
                    - [`~hooks.PyramidAttentionBroadcastConfig`]
                    - [`~hooks.FasterCacheConfig`]
                    - [`~hooks.FirstBlockCacheConfig`]
                    - [`~hooks.SeaCacheConfig`]
                    - [`~hooks.TextKVCacheConfig`]

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
            FirstBlockCacheConfig,
            HookRegistry,
            MagCacheConfig,
            PyramidAttentionBroadcastConfig,
            SeaCacheConfig,
            TaylorSeerCacheConfig,
            TextKVCacheConfig,
            apply_faster_cache,
            apply_first_block_cache,
            apply_mag_cache,
            apply_pyramid_attention_broadcast,
            apply_sea_cache,
            apply_taylorseer_cache,
            apply_text_kv_cache,
        )

        if self.is_cache_enabled:
            raise ValueError(
                f"Caching has already been enabled with {type(self._cache_config)}. To apply a new caching technique, please disable the existing one first."
            )

        if isinstance(config, FasterCacheConfig):
            apply_faster_cache(self, config)
        elif isinstance(config, FirstBlockCacheConfig):
            apply_first_block_cache(self, config)
        elif isinstance(config, MagCacheConfig):
            apply_mag_cache(self, config)
        elif isinstance(config, TextKVCacheConfig):
            apply_text_kv_cache(self, config)
        elif isinstance(config, PyramidAttentionBroadcastConfig):
            apply_pyramid_attention_broadcast(self, config)
        elif isinstance(config, SeaCacheConfig):
            apply_sea_cache(self, config)
        elif isinstance(config, TaylorSeerCacheConfig):
            apply_taylorseer_cache(self, config)
        else:
            raise ValueError(f"Cache config {type(config)} is not supported.")

        # Applying a cache technique registers hooks on child blocks, which stales any
        # `_child_registries_cache` built earlier (e.g. by a prior `cache_context`). Invalidate it
        # so later context updates reach the freshly-registered block hooks.
        HookRegistry.check_if_exists_or_initialize(self).invalidate_child_registries_cache()

        self._cache_config = config

    def disable_cache(self) -> None:
        from ..hooks import (
            FasterCacheConfig,
            FirstBlockCacheConfig,
            HookRegistry,
            MagCacheConfig,
            PyramidAttentionBroadcastConfig,
            SeaCacheConfig,
            TaylorSeerCacheConfig,
            TextKVCacheConfig,
        )
        from ..hooks.faster_cache import _FASTER_CACHE_BLOCK_HOOK, _FASTER_CACHE_DENOISER_HOOK
        from ..hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
        from ..hooks.mag_cache import _MAG_CACHE_BLOCK_HOOK, _MAG_CACHE_LEADER_BLOCK_HOOK
        from ..hooks.pyramid_attention_broadcast import _PYRAMID_ATTENTION_BROADCAST_HOOK
        from ..hooks.sea_cache import (
            _SEA_CACHE_BLOCK_HOOK,
            _SEA_CACHE_LEADER_BLOCK_HOOK,
            _SEA_CACHE_POST_NORM_HOOK,
            _SEA_CACHE_ROOT_HOOK,
        )
        from ..hooks.taylorseer_cache import _TAYLORSEER_CACHE_HOOK
        from ..hooks.text_kv_cache import _TEXT_KV_CACHE_BLOCK_HOOK, _TEXT_KV_CACHE_TRANSFORMER_HOOK

        if self._cache_config is None:
            logger.warning("Caching techniques have not been enabled, so there's nothing to disable.")
            return

        registry = HookRegistry.check_if_exists_or_initialize(self)
        if isinstance(self._cache_config, FasterCacheConfig):
            registry.remove_hook(_FASTER_CACHE_DENOISER_HOOK, recurse=True)
            registry.remove_hook(_FASTER_CACHE_BLOCK_HOOK, recurse=True)
        elif isinstance(self._cache_config, FirstBlockCacheConfig):
            registry.remove_hook(_FBC_LEADER_BLOCK_HOOK, recurse=True)
            registry.remove_hook(_FBC_BLOCK_HOOK, recurse=True)
        elif isinstance(self._cache_config, MagCacheConfig):
            registry.remove_hook(_MAG_CACHE_LEADER_BLOCK_HOOK, recurse=True)
            registry.remove_hook(_MAG_CACHE_BLOCK_HOOK, recurse=True)
        elif isinstance(self._cache_config, PyramidAttentionBroadcastConfig):
            registry.remove_hook(_PYRAMID_ATTENTION_BROADCAST_HOOK, recurse=True)
        elif isinstance(self._cache_config, TextKVCacheConfig):
            registry.remove_hook(_TEXT_KV_CACHE_TRANSFORMER_HOOK, recurse=True)
            registry.remove_hook(_TEXT_KV_CACHE_BLOCK_HOOK, recurse=True)
        elif isinstance(self._cache_config, SeaCacheConfig):
            registry.remove_hook(_SEA_CACHE_POST_NORM_HOOK, recurse=True)
            registry.remove_hook(_SEA_CACHE_BLOCK_HOOK, recurse=True)
            registry.remove_hook(_SEA_CACHE_LEADER_BLOCK_HOOK, recurse=True)
            registry.remove_hook(_SEA_CACHE_ROOT_HOOK, recurse=True)
        elif isinstance(self._cache_config, TaylorSeerCacheConfig):
            registry.remove_hook(_TAYLORSEER_CACHE_HOOK, recurse=True)
        else:
            raise ValueError(f"Cache config {type(self._cache_config)} is not supported.")

        # Removing the cache hooks stales any `_child_registries_cache` that included them.
        registry.invalidate_child_registries_cache()

        self._cache_config = None

    def _reset_stateful_cache(self, recurse: bool = True) -> None:
        from ..hooks import HookRegistry

        HookRegistry.check_if_exists_or_initialize(self).reset_stateful_hooks(recurse=recurse)

    @contextmanager
    def cache_context(self, name: str, **kwargs):
        r"""Context manager that provides information for cache management.

        `name` is the name of the denoising call, usually `"cond"` or `"uncond"`. The keyword arguments describe where
        the denoising loop is — see `CacheContext` for the accepted fields, e.g. `cache_context("cond", step_index=i,
        sigma=sigma)`.
        """
        from ..hooks.hooks import CacheContext, _set_cache_context

        _set_cache_context(self, CacheContext(name, **kwargs))

        try:
            yield
        finally:
            _set_cache_context(self, None)
