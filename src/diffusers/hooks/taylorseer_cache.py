import math
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn

from .hooks import ModelHook, StateManager, HookRegistry
from ..utils import logging


logger = logging.get_logger(__name__)
_TAYLORSEER_CACHE_HOOK = "taylorseer_cache"
_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "^blocks.*attn",
    "^transformer_blocks.*attn",
    "^single_transformer_blocks.*attn",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("^temporal_transformer_blocks.*attn",)
_TRANSFORMER_BLOCK_IDENTIFIERS = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS + _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS
_BLOCK_IDENTIFIERS = (
    "^[^.]*block[^.]*\\.[^.]+$",
)
_PROJ_OUT_IDENTIFIERS = ("^proj_out$",)

@dataclass
class TaylorSeerCacheConfig:
    """
    Configuration for TaylorSeer cache.
    See: https://huggingface.co/papers/2503.06923

    Attributes:
        warmup_steps (`int`, defaults to `3`):
            Number of denoising steps to run with full computation
            before enabling caching. During warmup, the Taylor series factors
            are still updated, but no predictions are used.

        predict_steps (`int`, defaults to `5`):
            Number of prediction (cached) steps to take between two full
            computations. That is, once a module state is refreshed, it will
            be reused for `predict_steps` subsequent denoising steps, then a new
            full forward will be computed on the next step.

        stop_predicts (`int`, *optional*, defaults to `None`):
            Denoising step index at which caching is disabled.
            If provided, for `self.current_step >= stop_predicts` all modules are
            evaluated normally (no predictions, no state updates).

        max_order (`int`, defaults to `1`):
            Maximum order of Taylor series expansion to approximate the
            features. Higher order gives closer approximation but more compute.

        taylor_factors_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type for computing Taylor series expansion factors.
            Use lower precision to reduce memory usage.
            Use higher precision to improve numerical stability.

        skip_identifiers (`List[str]`, *optional*, defaults to `None`):
            Regex patterns (fullmatch) for module names to be placed in
            "skip" mode, where the module is evaluated during warmup /
            refresh, but then replaced by a cheap dummy tensor during
            prediction steps.

        cache_identifiers (`List[str]`, *optional*, defaults to `None`):
            Regex patterns (fullmatch) for module names to be placed in
            Taylor-series caching mode.

        lite (`bool`, *optional*, defaults to `False`):
            Whether to use a TaylorSeer Lite variant that reduces memory usage. This option overrides
            any user-provided `skip_identifiers` or `cache_identifiers` patterns.
    Notes:
        - Patterns are applied with `re.fullmatch` on `module_name`.
        - If either `skip_identifiers` or `cache_identifiers` is provided, only modules matching at least
          one of those patterns will be hooked.
        - If neither is provided, all attention-like modules will be hooked.
    """

    warmup_steps: int = 3
    predict_steps: int = 5
    stop_predicts: Optional[int] = None
    max_order: int = 1
    taylor_factors_dtype: Optional[torch.dtype] = torch.bfloat16
    skip_identifiers: Optional[List[str]] = None
    cache_identifiers: Optional[List[str]] = None
    lite: bool = False

    def __repr__(self) -> str:
        return (
            "TaylorSeerCacheConfig("
            f"warmup_steps={self.warmup_steps}, "
            f"predict_steps={self.predict_steps}, "
            f"stop_predicts={self.stop_predicts}, "
            f"max_order={self.max_order}, "
            f"taylor_factors_dtype={self.taylor_factors_dtype}, "
            f"skip_identifiers={self.skip_identifiers}, "
            f"cache_identifiers={self.cache_identifiers}, "
            f"lite={self.lite})"
        )


class TaylorSeerState:
    def __init__(
        self,
        taylor_factors_dtype: Optional[torch.dtype] = torch.bfloat16,
        max_order: int = 1,
    ):
        self.taylor_factors_dtype = taylor_factors_dtype
        self.max_order = max_order

        self.module_dtypes: Tuple[torch.dtype, ...] = ()
        self.last_update_step: Optional[int] = None
        self.taylor_factors: Dict[int, Dict[int, torch.Tensor]] = {}

        # For skip-mode modules
        self.device: Optional[torch.device] = None
        self.dummy_tensors: Optional[Tuple[torch.Tensor, ...]] = None

        self.current_step = -1

    def reset(self) -> None:
        self.last_update_step = None
        self.taylor_factors = {}
        self.device = None
        self.dummy_tensors = None
        self.current_step = -1

    def update(
        self,
        outputs: Tuple[torch.Tensor, ...],
        current_step: int,
    ) -> None:
        self.module_dtypes = tuple(output.dtype for output in outputs)
        for i in range(len(outputs)):
            features = outputs[i].to(self.taylor_factors_dtype)
            new_factors: Dict[int, torch.Tensor] = {0: features}
            is_first_update = self.last_update_step is None
            if not is_first_update:
                delta_step = current_step - self.last_update_step
                if delta_step == 0:
                    raise ValueError("Delta step cannot be zero for TaylorSeer update.")

                # Recursive divided differences up to max_order
                for j in range(self.max_order):
                    prev = self.taylor_factors[i].get(j)
                    if prev is None:
                        break
                    new_factors[j + 1] = (new_factors[j] - prev.to(self.taylor_factors_dtype)) / delta_step
            self.taylor_factors[i] = new_factors
        self.last_update_step = current_step

    def predict(self, current_step: int) -> torch.Tensor:
        if self.last_update_step is None:
            raise ValueError("Cannot predict without prior initialization/update.")

        step_offset = current_step - self.last_update_step

        if not self.taylor_factors:
            raise ValueError("Taylor factors empty during prediction.")

        outputs = []
        for i in range(len(self.module_dtypes)):
            taylor_factors = self.taylor_factors[i]
            # Accumulate Taylor series: f(t0 + Δt) ≈ Σ f^{(n)}(t0) * (Δt^n / n!)
            output = torch.zeros_like(taylor_factors[0])
            for order, factor in taylor_factors.items():
                # Note: order starts at 0
                coeff = (step_offset**order) / math.factorial(order)
                output = output + factor * coeff
            outputs.append(output.to(self.module_dtypes[i]))

        return outputs


class TaylorSeerCacheHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        module_name: str,
        predict_steps: int,
        warmup_steps: int,
        taylor_factors_dtype: torch.dtype,
        state_manager: StateManager,
        stop_predicts: Optional[int] = None,
        is_skip: bool = False,
    ):
        super().__init__()
        self.module_name = module_name
        self.predict_steps = predict_steps
        self.warmup_steps = warmup_steps
        self.stop_predicts = stop_predicts
        self.taylor_factors_dtype = taylor_factors_dtype
        self.state_manager = state_manager
        self.is_skip = is_skip

        self.dummy_outputs = None

    def initialize_hook(self, module: torch.nn.Module):
        return module

    def reset_state(self, module: torch.nn.Module) -> None:
        """
        Reset state between sampling runs.
        """
        self.dummy_outputs = None
        self.current_step = -1
        self.state_manager.reset()

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: TaylorSeerState = self.state_manager.get_state()
        state.current_step += 1
        current_step = state.current_step
        is_warmup_phase = current_step < self.warmup_steps
        should_compute = (
            is_warmup_phase
            or ((current_step - self.warmup_steps - 1) % self.predict_steps == 0)
            or (self.stop_predicts is not None and current_step >= self.stop_predicts)
        )
        if should_compute:
            outputs = self.fn_ref.original_forward(*args, **kwargs)
            if not self.is_skip:
                state.update((outputs,) if isinstance(outputs, torch.Tensor) else outputs, current_step)
            else:
                self.dummy_outputs = outputs
            return outputs

        if self.is_skip:
            return self.dummy_outputs

        outputs = state.predict(current_step)
        return outputs[0] if len(outputs) == 1 else outputs


def _resolve_patterns(config: TaylorSeerCacheConfig) -> Tuple[List[str], List[str]]:
    """
    Resolve effective skip and cache pattern lists from config + templates.
    """

    skip_patterns = config.skip_identifiers if config.skip_identifiers is not None else None
    cache_patterns = config.cache_identifiers if config.cache_identifiers is not None else None

    return skip_patterns or [], cache_patterns or []


def apply_taylorseer_cache(module: torch.nn.Module, config: TaylorSeerCacheConfig):
    """
    Applies the TaylorSeer cache to a given pipeline (typically the transformer / UNet).

    Args:
        module (torch.nn.Module): The model subtree to apply the hooks to.
        config (TaylorSeerCacheConfig): Configuration for the cache.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import FluxPipeline, TaylorSeerCacheConfig
    >>>
    >>> pipe = FluxPipeline.from_pretrained(
    ...     "black-forest-labs/FLUX.1-dev",
    ...     torch_dtype=torch.bfloat16,
    ... )
    >>> pipe.to("cuda")
    >>>
    >>> config = TaylorSeerCacheConfig(
    ...     predict_steps=5,
    ...     max_order=1,
    ...     warmup_steps=3,
    ...     taylor_factors_dtype=torch.float32,
    ... )
    >>> pipe.transformer.enable_cache(config)
    ```
    """
    skip_patterns, cache_patterns = _resolve_patterns(config)

    logger.debug("TaylorSeer skip identifiers: %s", skip_patterns)
    logger.debug("TaylorSeer cache identifiers: %s", cache_patterns)

    cache_patterns = cache_patterns or _TRANSFORMER_BLOCK_IDENTIFIERS

    if config.lite:
        logger.info("Using TaylorSeer Lite variant for cache.")
        cache_patterns = _PROJ_OUT_IDENTIFIERS
        skip_patterns = _BLOCK_IDENTIFIERS
        if config.skip_identifiers or config.cache_identifiers:
            logger.warning("Lite mode overrides user patterns.")

    for name, submodule in module.named_modules():
        matches_skip = any(re.fullmatch(pattern, name) for pattern in skip_patterns)
        matches_cache = any(re.fullmatch(pattern, name) for pattern in cache_patterns)
        if not (matches_skip or matches_cache):
            continue
        logger.debug(
            "Applying TaylorSeer cache to %s (mode=%s)",
            name,
            "skip" if matches_skip else "cache",
        )
        state_manager = StateManager(
            TaylorSeerState,
            init_kwargs={
                "taylor_factors_dtype": config.taylor_factors_dtype,
                "max_order": config.max_order,
            },
        )
        _apply_taylorseer_cache_hook(
            name=name,
            module=submodule,
            config=config,
            is_skip=matches_skip,
            state_manager=state_manager,
        )


def _apply_taylorseer_cache_hook(
    name: str,
    module: nn.Module,
    config: TaylorSeerCacheConfig,
    is_skip: bool,
    state_manager: StateManager,
):
    """
    Registers the TaylorSeer hook on the specified nn.Module.

    Args:
        name: Name of the module.
        module: The nn.Module to be hooked.
        config: Cache configuration.
        is_skip: Whether this module should operate in "skip" mode.
        state_manager: The state manager for managing hook state.
    """
    registry = HookRegistry.check_if_exists_or_initialize(module)

    hook = TaylorSeerCacheHook(
        module_name=name,
        predict_steps=config.predict_steps,
        warmup_steps=config.warmup_steps,
        taylor_factors_dtype=config.taylor_factors_dtype,
        stop_predicts=config.stop_predicts,
        is_skip=is_skip,
        state_manager=state_manager,
    )

    registry.register_hook(hook, _TAYLORSEER_CACHE_HOOK)
