import math
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch

from .hooks import ModelHook
from ..models.attention import Attention
from ..models.attention import AttentionModuleMixin
from ._common import _ATTENTION_CLASSES
from ..hooks import HookRegistry
from ..utils import logging


logger = logging.get_logger(__name__)

_TAYLORSEER_ATTENTION_CACHE_HOOK = "taylorseer_attention_cache"

# Predefined cache templates for optimized architectures
_CACHE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "flux": {
        "cache": [
            r"transformer_blocks\.\d+\.attn",
            r"transformer_blocks\.\d+\.ff",
            r"transformer_blocks\.\d+\.ff_context",
            r"single_transformer_blocks\.\d+\.proj_out",
        ],
        "skip": [
            r"single_transformer_blocks\.\d+\.attn",
            r"single_transformer_blocks\.\d+\.proj_mlp",
            r"single_transformer_blocks\.\d+\.act_mlp",
        ],
    },
}


@dataclass
class TaylorSeerCacheConfig:
    """
    Configuration for TaylorSeer cache.
    See: https://huggingface.co/papers/2503.06923

    Attributes:
        warmup_steps (`int`, defaults to `3`):
            Number of *outer* diffusion steps to run with full computation
            before enabling caching. During warmup, the Taylor series factors
            are still updated, but no predictions are used.

        predict_steps (`int`, defaults to `5`):
            Number of prediction (cached) steps to take between two full
            computations. That is, once a module state is refreshed, it will
            be reused for `predict_steps` subsequent outer steps, then a new
            full forward will be computed on the next step.

        stop_predicts (`int`, *optional*, defaults to `None`):
            Outer diffusion step index at which caching is disabled.
            If provided, for `true_step >= stop_predicts` all modules are
            evaluated normally (no predictions, no state updates).

        max_order (`int`, defaults to `1`):
            Maximum order of Taylor series expansion to approximate the
            features. Higher order gives closer approximation but more compute.

        num_inner_loops (`int`, defaults to `1`):
            Number of inner loops per outer diffusion step. For example,
            with classifier-free guidance (CFG) you typically have 2 inner
            loops: unconditional and conditional branches.

        taylor_factors_dtype (`torch.dtype`, defaults to `torch.float32`):
            Data type for computing Taylor series expansion factors.

        architecture (`str`, *optional*, defaults to `None`):
            If provided, will look up default `cache` and `skip` regex
            patterns in `_CACHE_TEMPLATES[architecture]`. These can be
            overridden by `skip_identifiers` and `cache_identifiers`.

        skip_identifiers (`List[str]`, *optional*, defaults to `None`):
            Regex patterns (fullmatch) for module names to be placed in
            "skip" mode, where the module is evaluated during warmup /
            refresh, but then replaced by a cheap dummy tensor during
            prediction steps.

        cache_identifiers (`List[str]`, *optional*, defaults to `None`):
            Regex patterns (fullmatch) for module names to be placed in
            Taylor-series caching mode.

    Notes:
        - Patterns are applied with `re.fullmatch` on `module_name`.
        - If either `skip_identifiers` or `cache_identifiers` is provided
          (or inferred from `architecture`), only modules matching at least
          one of those patterns will be hooked.
        - If neither is provided, all attention-like modules will be hooked.
    """

    warmup_steps: int = 3
    predict_steps: int = 5
    stop_predicts: Optional[int] = None
    max_order: int = 1
    num_inner_loops: int = 1
    taylor_factors_dtype: torch.dtype = torch.float32
    architecture: str | None = None
    skip_identifiers: Optional[List[str]] = None
    cache_identifiers: Optional[List[str]] = None

    def __repr__(self) -> str:
        return (
            "TaylorSeerCacheConfig("
            f"warmup_steps={self.warmup_steps}, "
            f"predict_steps={self.predict_steps}, "
            f"stop_predicts={self.stop_predicts}, "
            f"max_order={self.max_order}, "
            f"num_inner_loops={self.num_inner_loops}, "
            f"taylor_factors_dtype={self.taylor_factors_dtype}, "
            f"architecture={self.architecture}, "
            f"skip_identifiers={self.skip_identifiers}, "
            f"cache_identifiers={self.cache_identifiers})"
        )

    @classmethod
    def get_identifiers_template(cls) -> Dict[str, Dict[str, List[str]]]:
        return _CACHE_TEMPLATES


class TaylorSeerOutputState:
    """
    Manages the state for Taylor series-based prediction of a single attention output.

    Tracks Taylor expansion factors, last update step, and remaining prediction steps.
    The Taylor expansion uses the (outer) timestep as the independent variable.

    This class is designed to handle state for a single inner loop index and a single
    output (in cases where the module forward returns multiple tensors).
    """

    def __init__(
        self,
        module_name: str,
        taylor_factors_dtype: torch.dtype,
        module_dtype: torch.dtype,
        is_skip: bool = False,
    ):
        self.module_name = module_name
        self.taylor_factors_dtype = taylor_factors_dtype
        self.module_dtype = module_dtype
        self.is_skip = is_skip

        self.remaining_predictions: int = 0
        self.last_update_step: Optional[int] = None
        self.taylor_factors: Dict[int, torch.Tensor] = {}

        # For skip-mode modules
        self.dummy_shape: Optional[Tuple[int, ...]] = None
        self.device: Optional[torch.device] = None
        self.dummy_tensor: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.remaining_predictions = 0
        self.last_update_step = None
        self.taylor_factors = {}
        self.dummy_shape = None
        self.device = None
        self.dummy_tensor = None

    def update(
        self,
        features: torch.Tensor,
        current_step: int,
        max_order: int,
        predict_steps: int,
    ) -> None:
        """
        Update Taylor factors based on the current features and (outer) timestep.

        For non-skip modules, finite difference approximations for derivatives are
        computed using recursive divided differences.

        Args:
            features: Attention output features to update with.
            current_step: Current outer timestep (true diffusion step).
            max_order: Maximum Taylor expansion order.
            predict_steps: Number of prediction steps to allow after this update.
        """
        if self.is_skip:
            # For skip modules we only need shape & device and a dummy tensor.
            self.dummy_shape = features.shape
            self.device = features.device
            # zero is safer than uninitialized values for a "skipped" module
            self.dummy_tensor = torch.zeros(
                self.dummy_shape,
                dtype=self.module_dtype,
                device=self.device,
            )
            self.taylor_factors = {}
            self.last_update_step = current_step
            self.remaining_predictions = predict_steps
            return

        features = features.to(self.taylor_factors_dtype)
        new_factors: Dict[int, torch.Tensor] = {0: features}

        is_first_update = self.last_update_step is None

        if not is_first_update:
            delta_step = current_step - self.last_update_step
            if delta_step == 0:
                raise ValueError("Delta step cannot be zero for TaylorSeer update.")

            # Recursive divided differences up to max_order
            for i in range(max_order):
                prev = self.taylor_factors.get(i)
                if prev is None:
                    break
                new_factors[i + 1] = (new_factors[i] - prev.to(self.taylor_factors_dtype)) / delta_step

        # Keep factors in taylor_factors_dtype
        self.taylor_factors = new_factors
        self.last_update_step = current_step
        self.remaining_predictions = predict_steps

        if self.module_name == "proj_out":
            logger.debug(
                "[UPDATE] module=%s remaining_predictions=%d current_step=%d is_first_update=%s",
                self.module_name,
                self.remaining_predictions,
                current_step,
                is_first_update,
            )

    def predict(self, current_step: int) -> torch.Tensor:
        """
        Predict features using the Taylor series at the given (outer) timestep.

        Args:
            current_step: Current outer timestep for prediction.

        Returns:
            Predicted features in the module's dtype.
        """
        if self.is_skip:
            if self.dummy_tensor is None:
                raise ValueError("Cannot predict for skip module without prior update.")
            self.remaining_predictions -= 1
            return self.dummy_tensor

        if self.last_update_step is None:
            raise ValueError("Cannot predict without prior initialization/update.")

        step_offset = current_step - self.last_update_step

        output: torch.Tensor
        if not self.taylor_factors:
            raise ValueError("Taylor factors empty during prediction.")

        # Accumulate Taylor series: f(t0 + Δt) ≈ Σ f^{(n)}(t0) * (Δt^n / n!)
        output = torch.zeros_like(self.taylor_factors[0])
        for order, factor in self.taylor_factors.items():
            # Note: order starts at 0
            coeff = (step_offset**order) / math.factorial(order)
            output = output + factor * coeff

        self.remaining_predictions -= 1
        out = output.to(self.module_dtype)

        if self.module_name == "proj_out":
            logger.debug(
                "[PREDICT] module=%s remaining_predictions=%d current_step=%d last_update_step=%s",
                self.module_name,
                self.remaining_predictions,
                current_step,
                self.last_update_step,
            )

        return out


class TaylorSeerAttentionCacheHook(ModelHook):
    """
    Hook for caching and predicting attention outputs using Taylor series approximations.

    Applies to attention modules in diffusion models (e.g., Flux).
    Performs full computations during warmup, then alternates between blocks of
    predictions and refreshes.

    The hook maintains separate states for each inner loop index (e.g., for
    classifier-free guidance). Each inner loop has its own list of
    `TaylorSeerOutputState` instances, one per output tensor from the module's
    forward (typically one).

    The `step_counter` increments on every forward call of this module.
    We define:
        - `inner_index  = step_counter % num_inner_loops`
        - `true_step    = step_counter // num_inner_loops`

    Warmup, prediction, and updates are handled per inner loop, but use the
    shared `true_step` (outer diffusion step).
    """

    _is_stateful = True

    def __init__(
        self,
        module_name: str,
        predict_steps: int,
        max_order: int,
        warmup_steps: int,
        taylor_factors_dtype: torch.dtype,
        num_inner_loops: int = 1,
        stop_predicts: Optional[int] = None,
        is_skip: bool = False,
    ):
        super().__init__()
        if num_inner_loops <= 0:
            raise ValueError("num_inner_loops must be >= 1")

        self.module_name = module_name
        self.predict_steps = predict_steps
        self.max_order = max_order
        self.warmup_steps = warmup_steps
        self.stop_predicts = stop_predicts
        self.num_inner_loops = num_inner_loops
        self.taylor_factors_dtype = taylor_factors_dtype
        self.is_skip = is_skip

        self.step_counter: int = -1
        self.states: Optional[List[Optional[List[TaylorSeerOutputState]]]] = None
        self.num_outputs: Optional[int] = None

    def initialize_hook(self, module: torch.nn.Module):
        self.step_counter = -1
        self.states = None
        self.num_outputs = None
        return module

    def reset_state(self, module: torch.nn.Module) -> None:
        """
        Reset state between sampling runs.
        """
        self.step_counter = -1
        self.states = None
        self.num_outputs = None

    @staticmethod
    def _listify(outputs):
        if isinstance(outputs, torch.Tensor):
            return [outputs]
        return list(outputs)

    def _delistify(self, outputs_list):
        if self.num_outputs == 1:
            return outputs_list[0]
        return tuple(outputs_list)

    def _ensure_states_initialized(
        self,
        module: torch.nn.Module,
        inner_index: int,
        true_step: int,
        *args,
        **kwargs,
    ) -> Optional[List[torch.Tensor]]:
        """
        Ensure per-inner-loop states exist. If this is the first call for this
        inner_index, perform a full forward, initialize states, and return the
        outputs. Otherwise, return None.
        """
        if self.states is None:
            self.states = [None for _ in range(self.num_inner_loops)]

        if self.states[inner_index] is not None:
            return None

        if self.module_name == "proj_out":
            logger.debug(
                "[FIRST STEP] Initializing states for %s (inner_index=%d, true_step=%d)",
                self.module_name,
                inner_index,
                true_step,
            )

        # First step for this inner loop: always full compute and initialize.
        attention_outputs = self._listify(self.fn_ref.original_forward(*args, **kwargs))
        module_dtype = attention_outputs[0].dtype

        if self.num_outputs is None:
            self.num_outputs = len(attention_outputs)
        elif self.num_outputs != len(attention_outputs):
            raise ValueError("Output count mismatch across inner loops.")

        self.states[inner_index] = [
            TaylorSeerOutputState(
                self.module_name,
                self.taylor_factors_dtype,
                module_dtype,
                is_skip=self.is_skip,
            )
            for _ in range(self.num_outputs)
        ]

        for i, features in enumerate(attention_outputs):
            self.states[inner_index][i].update(
                features=features,
                current_step=true_step,
                max_order=self.max_order,
                predict_steps=self.predict_steps,
            )

        return attention_outputs

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        self.step_counter += 1
        inner_index = self.step_counter % self.num_inner_loops
        true_step = self.step_counter // self.num_inner_loops
        is_warmup_phase = true_step < self.warmup_steps

        if self.module_name == "proj_out":
            logger.debug(
                "[FORWARD] module=%s step_counter=%d inner_index=%d true_step=%d is_warmup=%s",
                self.module_name,
                self.step_counter,
                inner_index,
                true_step,
                is_warmup_phase,
            )

        # First-time initialization for this inner loop
        maybe_outputs = self._ensure_states_initialized(module, inner_index, true_step, *args, **kwargs)
        if maybe_outputs is not None:
            return self._delistify(maybe_outputs)

        assert self.states is not None
        states = self.states[inner_index]
        assert states is not None and len(states) > 0

        # If stop_predicts is set and we are past that step, always run full forward
        if self.stop_predicts is not None and true_step >= self.stop_predicts:
            attention_outputs = self._listify(self.fn_ref.original_forward(*args, **kwargs))
            return self._delistify(attention_outputs)

        # Decide between prediction vs refresh
        # - Never predict during warmup.
        # - Otherwise, predict while we still have remaining_predictions.
        should_predict = (not is_warmup_phase) and (states[0].remaining_predictions > 0)

        if should_predict:
            predicted_outputs = [state.predict(true_step) for state in states]
            return self._delistify(predicted_outputs)

        # Full compute: warmup or refresh
        attention_outputs = self._listify(self.fn_ref.original_forward(*args, **kwargs))
        for i, features in enumerate(attention_outputs):
            states[i].update(
                features=features,
                current_step=true_step,
                max_order=self.max_order,
                predict_steps=self.predict_steps,
            )
        return self._delistify(attention_outputs)


def _resolve_patterns(config: TaylorSeerCacheConfig) -> Tuple[List[str], List[str]]:
    """
    Resolve effective skip and cache pattern lists from config + templates.
    """
    template = _CACHE_TEMPLATES.get(config.architecture or "", {})
    default_skip = template.get("skip", [])
    default_cache = template.get("cache", [])

    skip_patterns = config.skip_identifiers if config.skip_identifiers is not None else default_skip
    cache_patterns = config.cache_identifiers if config.cache_identifiers is not None else default_cache

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
    ...     architecture="flux",
    ...     num_inner_loops=2,  # e.g. CFG
    ... )
    >>> pipe.transformer.enable_cache(config)
    ```
    """
    skip_patterns, cache_patterns = _resolve_patterns(config)

    logger.debug("TaylorSeer skip identifiers: %s", skip_patterns)
    logger.debug("TaylorSeer cache identifiers: %s", cache_patterns)

    use_patterns = bool(skip_patterns or cache_patterns)

    for name, submodule in module.named_modules():
        matches_skip = any(re.fullmatch(pattern, name) for pattern in skip_patterns)
        matches_cache = any(re.fullmatch(pattern, name) for pattern in cache_patterns)

        if use_patterns:
            # If patterns are configured (either skip or cache), only touch modules
            # that explicitly match at least one pattern.
            if not (matches_skip or matches_cache):
                continue

            logger.debug(
                "Applying TaylorSeer cache to %s (mode=%s)",
                name,
                "skip" if matches_skip else "cache",
            )
            _apply_taylorseer_cache_hook(
                name=name,
                module=submodule,
                config=config,
                is_skip=matches_skip,
            )
        else:
            # No patterns configured: fall back to "all attention modules".
            if isinstance(submodule, (*_ATTENTION_CLASSES, AttentionModuleMixin)):
                logger.debug("Applying TaylorSeer cache to %s (fallback attention mode)", name)
                _apply_taylorseer_cache_hook(
                    name=name,
                    module=submodule,
                    config=config,
                    is_skip=False,
                )


def _apply_taylorseer_cache_hook(
    name: str,
    module: Attention,
    config: TaylorSeerCacheConfig,
    is_skip: bool,
):
    """
    Registers the TaylorSeer hook on the specified attention module.

    Args:
        name: Name of the module.
        module: The attention-like module to be hooked.
        config: Cache configuration.
        is_skip: Whether this module should operate in "skip" mode.
    """
    registry = HookRegistry.check_if_exists_or_initialize(module)

    hook = TaylorSeerAttentionCacheHook(
        module_name=name,
        predict_steps=config.predict_steps,
        max_order=config.max_order,
        warmup_steps=config.warmup_steps,
        taylor_factors_dtype=config.taylor_factors_dtype,
        num_inner_loops=config.num_inner_loops,
        stop_predicts=config.stop_predicts,
        is_skip=is_skip,
    )

    registry.register_hook(hook, _TAYLORSEER_ATTENTION_CACHE_HOOK)
