from collections.abc import Iterator
from functools import wraps


def patch_triton_autotuner_prune_configs() -> bool:
    """
    Triton's autotuner assumes `prune_configs()` returns a reusable collection.
    Some GemLite kernels provide a generator-backed config pruner, which can be
    consumed once and later appear empty during autotune result selection.
    """
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return False

    original = Autotuner.prune_configs
    if getattr(original, "_diffusers_materializes_iterators", False):
        return False

    @wraps(original)
    def prune_configs(self, kwargs):
        pruned_configs = original(self, kwargs)
        if isinstance(pruned_configs, Iterator):
            return list(pruned_configs)
        return pruned_configs

    prune_configs._diffusers_materializes_iterators = True
    Autotuner.prune_configs = prune_configs
    return True
