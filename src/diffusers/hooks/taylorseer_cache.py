import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils import logging
from .hooks import HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)
_TAYLORSEER_CACHE_HOOK = "taylorseer_cache"
_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "^blocks.*attn",
    "^transformer_blocks.*attn",
    "^single_transformer_blocks.*attn",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("^temporal_transformer_blocks.*attn",)
_TRANSFORMER_BLOCK_IDENTIFIERS = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS + _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS
_BLOCK_IDENTIFIERS = ("^[^.]*block[^.]*\\.[^.]+$",)
_PROJ_OUT_IDENTIFIERS = ("^proj_out$",)


@dataclass
class TaylorSeerCacheConfig:
    """
    Configuration for TaylorSeer cache. See: https://huggingface.co/papers/2503.06923

    Attributes:
        cache_interval (`int`, defaults to `5`):
            The interval between full computation steps. After a full computation, the cached (predicted) outputs are
            reused for this many subsequent denoising steps before refreshing with a new full forward pass.

        disable_cache_before_step (`int`, defaults to `3`):
            The denoising step index before which caching is disabled, meaning full computation is performed for the
            initial steps (0 to disable_cache_before_step - 1) to gather data for Taylor series approximations. During
            these steps, Taylor factors are updated, but caching/predictions are not applied. Caching begins at this
            step.

        disable_cache_after_step (`int`, *optional*, defaults to `None`):
            The denoising step index after which caching is disabled. If set, for steps >= this value, all modules run
            full computations without predictions or state updates, ensuring accuracy in later stages if needed.

        max_order (`int`, defaults to `1`):
            The highest order in the Taylor series expansion for approximating module outputs. Higher orders provide
            better approximations but increase computation and memory usage.

        taylor_factors_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type used for storing and computing Taylor series factors. Lower precision reduces memory but may
            affect stability; higher precision improves accuracy at the cost of more memory.

        skip_predict_identifiers (`List[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names to place as "skip" in "cache" mode. In this mode,
            the module computes fully during initial or refresh steps but returns a zero tensor (matching recorded
            shape) during prediction steps to skip computation cheaply.

        cache_identifiers (`List[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names to place in Taylor-series caching mode, where
            outputs are approximated and cached for reuse.

        use_lite_mode (`bool`, *optional*, defaults to `False`):
            Enables a lightweight TaylorSeer variant that minimizes memory usage by applying predefined patterns for
            skipping and caching (e.g., skipping blocks and caching projections). This overrides any custom
            `inactive_identifiers` or `active_identifiers`.

    Notes:
        - Patterns are matched using `re.fullmatch` on the module name.
        - If `skip_predict_identifiers` or `cache_identifiers` are provided, only matching modules are hooked.
        - If neither is provided, all attention-like modules are hooked by default.

    Example of inactive and active usage:

    ```py
    def forward(x):
        x = self.module1(x)  # inactive module: returns zeros tensor based on shape recorded during full compute
        x = self.module2(x)  # active module: caches output here, avoiding recomputation of prior steps
        return x
    ```
    """

    cache_interval: int = 5
    disable_cache_before_step: int = 3
    disable_cache_after_step: Optional[int] = None
    max_order: int = 1
    taylor_factors_dtype: Optional[torch.dtype] = torch.bfloat16
    skip_predict_identifiers: Optional[List[str]] = None
    cache_identifiers: Optional[List[str]] = None
    use_lite_mode: bool = False

    def __repr__(self) -> str:
        return (
            "TaylorSeerCacheConfig("
            f"cache_interval={self.cache_interval}, "
            f"disable_cache_before_step={self.disable_cache_before_step}, "
            f"disable_cache_after_step={self.disable_cache_after_step}, "
            f"max_order={self.max_order}, "
            f"taylor_factors_dtype={self.taylor_factors_dtype}, "
            f"skip_predict_identifiers={self.skip_predict_identifiers}, "
            f"cache_identifiers={self.cache_identifiers}, "
            f"use_lite_mode={self.use_lite_mode})"
        )


class TaylorSeerState:
    def __init__(
        self,
        taylor_factors_dtype: Optional[torch.dtype] = torch.bfloat16,
        max_order: int = 1,
        is_inactive: bool = False,
    ):
        self.taylor_factors_dtype = taylor_factors_dtype
        self.max_order = max_order
        self.is_inactive = is_inactive

        self.module_dtypes: Tuple[torch.dtype, ...] = ()
        self.last_update_step: Optional[int] = None
        self.taylor_factors: Dict[int, Dict[int, torch.Tensor]] = {}
        self.inactive_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None
        self.device: Optional[torch.device] = None
        self.current_step: int = -1

    def reset(self) -> None:
        self.current_step = -1
        self.last_update_step = None
        self.taylor_factors = {}
        self.inactive_shapes = None
        self.device = None

    def update(
        self,
        outputs: Tuple[torch.Tensor, ...],
    ) -> None:
        self.module_dtypes = tuple(output.dtype for output in outputs)
        self.device = outputs[0].device

        if self.is_inactive:
            self.inactive_shapes = tuple(output.shape for output in outputs)
        else:
            for i, features in enumerate(outputs):
                new_factors: Dict[int, torch.Tensor] = {0: features}
                is_first_update = self.last_update_step is None
                if not is_first_update:
                    delta_step = self.current_step - self.last_update_step
                    if delta_step == 0:
                        raise ValueError("Delta step cannot be zero for TaylorSeer update.")

                    # Recursive divided differences up to max_order
                    prev_factors = self.taylor_factors.get(i, {})
                    for j in range(self.max_order):
                        prev = prev_factors.get(j)
                        if prev is None:
                            break
                        new_factors[j + 1] = (new_factors[j] - prev.to(features.dtype)) / delta_step
                self.taylor_factors[i] = {
                    order: factor.to(self.taylor_factors_dtype) for order, factor in new_factors.items()
                }

        self.last_update_step = self.current_step

    @torch.compiler.disable
    def predict(self) -> List[torch.Tensor]:
        if self.last_update_step is None:
            raise ValueError("Cannot predict without prior initialization/update.")

        step_offset = self.current_step - self.last_update_step

        outputs = []
        if self.is_inactive:
            if self.inactive_shapes is None:
                raise ValueError("Inactive shapes not set during prediction.")
            for i in range(len(self.module_dtypes)):
                outputs.append(
                    torch.zeros(
                        self.inactive_shapes[i],
                        dtype=self.module_dtypes[i],
                        device=self.device,
                    )
                )
        else:
            if not self.taylor_factors:
                raise ValueError("Taylor factors empty during prediction.")
            num_outputs = len(self.taylor_factors)
            num_orders = len(self.taylor_factors[0])
            for i in range(num_outputs):
                output_dtype = self.module_dtypes[i]
                taylor_factors = self.taylor_factors[i]
                output = torch.zeros_like(taylor_factors[0], dtype=output_dtype)
                for order in range(num_orders):
                    coeff = (step_offset**order) / math.factorial(order)
                    factor = taylor_factors[order]
                    output = output + factor.to(output_dtype) * coeff
                outputs.append(output)
        return outputs


class TaylorSeerCacheHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        cache_interval: int,
        disable_cache_before_step: int,
        taylor_factors_dtype: torch.dtype,
        state_manager: StateManager,
        disable_cache_after_step: Optional[int] = None,
    ):
        super().__init__()
        self.cache_interval = cache_interval
        self.disable_cache_before_step = disable_cache_before_step
        self.disable_cache_after_step = disable_cache_after_step
        self.taylor_factors_dtype = taylor_factors_dtype
        self.state_manager = state_manager

    def initialize_hook(self, module: torch.nn.Module):
        return module

    def reset_state(self, module: torch.nn.Module) -> None:
        """
        Reset state between sampling runs.
        """
        self.state_manager.reset()

    @torch.compiler.disable
    def _measure_should_compute(self) -> bool:
        state: TaylorSeerState = self.state_manager.get_state()
        state.current_step += 1
        current_step = state.current_step
        is_warmup_phase = current_step < self.disable_cache_before_step
        is_compute_interval = (current_step - self.disable_cache_before_step - 1) % self.cache_interval == 0
        is_cooldown_phase = self.disable_cache_after_step is not None and current_step >= self.disable_cache_after_step
        should_compute = is_warmup_phase or is_compute_interval or is_cooldown_phase
        return should_compute, state

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        should_compute, state = self._measure_should_compute()
        if should_compute:
            outputs = self.fn_ref.original_forward(*args, **kwargs)
            wrapped_outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
            state.update(wrapped_outputs)
            return outputs

        outputs_list = state.predict()
        return outputs_list[0] if len(outputs_list) == 1 else tuple(outputs_list)


def _resolve_patterns(config: TaylorSeerCacheConfig) -> Tuple[List[str], List[str]]:
    """
    Resolve effective inactive and active pattern lists from config + templates.
    """

    inactive_patterns = config.skip_predict_identifiers if config.skip_predict_identifiers is not None else None
    active_patterns = config.cache_identifiers if config.cache_identifiers is not None else None

    return inactive_patterns or [], active_patterns or []


def apply_taylorseer_cache(module: torch.nn.Module, config: TaylorSeerCacheConfig):
    """
    Applies the TaylorSeer cache to a given pipeline (typically the transformer / UNet).

    This function hooks selected modules in the model to enable caching or skipping based on the provided
    configuration, reducing redundant computations in diffusion denoising loops.

    Args:
        module (torch.nn.Module): The model subtree to apply the hooks to.
        config (TaylorSeerCacheConfig): Configuration for the cache.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import FluxPipeline, TaylorSeerCacheConfig

    >>> pipe = FluxPipeline.from_pretrained(
    ...     "black-forest-labs/FLUX.1-dev",
    ...     torch_dtype=torch.bfloat16,
    ... )
    >>> pipe.to("cuda")

    >>> config = TaylorSeerCacheConfig(
    ...     cache_interval=5,
    ...     max_order=1,
    ...     disable_cache_before_step=3,
    ...     taylor_factors_dtype=torch.float32,
    ... )
    >>> pipe.transformer.enable_cache(config)
    ```
    """
    inactive_patterns, active_patterns = _resolve_patterns(config)

    active_patterns = active_patterns or _TRANSFORMER_BLOCK_IDENTIFIERS

    if config.use_lite_mode:
        logger.info("Using TaylorSeer Lite variant for cache.")
        active_patterns = _PROJ_OUT_IDENTIFIERS
        inactive_patterns = _BLOCK_IDENTIFIERS
        if config.skip_predict_identifiers or config.cache_identifiers:
            logger.warning("Lite mode overrides user patterns.")

    for name, submodule in module.named_modules():
        matches_inactive = any(re.fullmatch(pattern, name) for pattern in inactive_patterns)
        matches_active = any(re.fullmatch(pattern, name) for pattern in active_patterns)
        if not (matches_inactive or matches_active):
            continue
        _apply_taylorseer_cache_hook(
            module=submodule,
            config=config,
            is_inactive=matches_inactive,
        )


def _apply_taylorseer_cache_hook(
    module: nn.Module,
    config: TaylorSeerCacheConfig,
    is_inactive: bool,
):
    """
    Registers the TaylorSeer hook on the specified nn.Module.

    Args:
        name: Name of the module.
        module: The nn.Module to be hooked.
        config: Cache configuration.
        is_inactive: Whether this module should operate in "inactive" mode.
    """
    state_manager = StateManager(
        TaylorSeerState,
        init_kwargs={
            "taylor_factors_dtype": config.taylor_factors_dtype,
            "max_order": config.max_order,
            "is_inactive": is_inactive,
        },
    )

    registry = HookRegistry.check_if_exists_or_initialize(module)

    hook = TaylorSeerCacheHook(
        cache_interval=config.cache_interval,
        disable_cache_before_step=config.disable_cache_before_step,
        taylor_factors_dtype=config.taylor_factors_dtype,
        disable_cache_after_step=config.disable_cache_after_step,
        state_manager=state_manager,
    )

    registry.register_hook(hook, _TAYLORSEER_CACHE_HOOK)
