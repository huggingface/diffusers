import torch
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Tuple
from .hooks import ModelHook
import math
from ..models.attention import Attention
from ..models.attention import AttentionModuleMixin
from ._common import (
    _ATTENTION_CLASSES,
)
from ..hooks import HookRegistry
from ..utils import logging
import re
from collections import defaultdict


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_TAYLORSEER_ATTENTION_CACHE_HOOK = "taylorseer_attention_cache"

# Predefined cache templates for optimized architectures
_CACHE_TEMPLATES = {
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
        warmup_steps (`int`, defaults to `3`): Calculate normal computations `N` times before applying this caching strategy. Higher `N` gives more closed outputs.
        predict_steps (`int`, defaults to `5`): Calculate the module states every `N` iterations. If this is set to `N`, the module computation will be skipped `N - 1` times before computing the new module states again.
        stop_predicts (`int`, *optional*, defaults to `None`): Disable caching strategy after this step, this feature helps produce fine-grained outputs. If not provided, the caching strategy will be applied until the end of the inference.
        max_order (`int`, defaults to `1`): Maximum order of Taylor series expansion to approximate the features. In theory, the higher the order, the more closed the output is to the actual value but also the more computation is required.
        taylor_factors_dtype (`torch.dtype`, defaults to `torch.float32`): Data type for calculating Taylor series expansion factors.
        architecture (`str`, *optional*, defaults to `None`): Option to use cache strategy optimized for specific architectures. By default, this cache strategy will be applied to all `Attention` modules.
        skip_identifiers (`List[str]`, *optional*, defaults to `[]`): Regex patterns to identify modules to skip computation.
        cache_identifiers (`List[str]`, *optional*, defaults to `[]`): Regex patterns to identify modules to cache.
    """

    warmup_steps: int = 3
    predict_steps: int = 5
    stop_predicts: Optional[int] = None
    max_order: int = 1
    taylor_factors_dtype: torch.dtype = torch.float32
    architecture: str | None = None
    skip_identifiers: List[str] = None
    cache_identifiers: List[str] = None

    def __repr__(self) -> str:
        return f"TaylorSeerCacheConfig(warmup_steps={self.warmup_steps}, predict_steps={self.predict_steps}, stop_predicts={self.stop_predicts}, max_order={self.max_order}, taylor_factors_dtype={self.taylor_factors_dtype}, architecture={self.architecture}, skip_identifiers={self.skip_identifiers}, cache_identifiers={self.cache_identifiers})"

    @classmethod
    def get_identifiers_template(self) -> Dict[str, Dict[str, List[str]]]:
        return _CACHE_TEMPLATES

class TaylorSeerOutputState:
    """
    Manages the state for Taylor series-based prediction of a single attention output.
    Tracks Taylor expansion factors, last update step, and remaining prediction steps.
    The Taylor expansion uses the timestep as the independent variable for approximation.
    """

    def __init__(
        self, module_name: str, taylor_factors_dtype: torch.dtype, module_dtype: torch.dtype, is_skip: bool = False
    ):
        self.module_name = module_name
        self.remaining_predictions: int = 0
        self.last_update_step: Optional[int] = None
        self.taylor_factors: Dict[int, torch.Tensor] = {}
        self.taylor_factors_dtype = taylor_factors_dtype
        self.module_dtype = module_dtype
        self.is_skip = is_skip
        self.dummy_shape: Optional[Tuple[int, ...]] = None
        self.device: Optional[torch.device] = None
        self.dummy_tensor: Optional[torch.Tensor] = None

    def reset(self):
        self.remaining_predictions = 0
        self.last_update_step = None
        self.taylor_factors = {}
        self.dummy_shape = None
        self.device = None
        self.dummy_tensor = None

    def update(
        self, features: torch.Tensor, current_step: int, max_order: int, predict_steps: int, is_first_update: bool
    ):
        """
        Updates the Taylor factors based on the current features and timestep.
        Computes finite difference approximations for derivatives using recursive divided differences.

        Args:
            features (torch.Tensor): The attention output features to update with.
            current_step (int): The current timestep or step number from the diffusion model.
            max_order (int): Maximum order of the Taylor expansion.
            predict_steps (int): Number of prediction steps to set after update.
            is_first_update (bool): Whether this is the initial update (skips difference computation).
        """
        if self.is_skip:
            self.dummy_shape = features.shape
            self.device = features.device
            self.taylor_factors = {}
            self.last_update_step = current_step
            self.remaining_predictions = predict_steps
        else:
            features = features.to(self.taylor_factors_dtype)
            new_factors = {0: features}
            if not is_first_update:
                if self.last_update_step is None:
                    raise ValueError("Cannot update without prior initialization.")
                delta_step = current_step - self.last_update_step
                if delta_step == 0:
                    raise ValueError("Delta step cannot be zero for updates.")
                for i in range(max_order):
                    if i in self.taylor_factors:
                        new_factors[i + 1] = (
                            new_factors[i] - self.taylor_factors[i].to(self.taylor_factors_dtype)
                        ) / delta_step
                    else:
                        break

            # taylor factors will be kept in the taylor_factors_dtype
            self.taylor_factors = new_factors
            self.last_update_step = current_step
            self.remaining_predictions = predict_steps

    def predict(self, current_step: int) -> torch.Tensor:
        """
        Predicts the features using the Taylor series expansion at the given timestep.

        Args:
            current_step (int): The current timestep for prediction.

        Returns:
            torch.Tensor: The predicted features in the module's dtype.
        """
        if self.is_skip:
            if self.dummy_shape is None or self.device is None:
                raise ValueError("Cannot predict for skip module without prior update.")
            self.remaining_predictions -= 1
            return torch.empty(self.dummy_shape, dtype=self.module_dtype, device=self.device)
        else:
            if self.last_update_step is None:
                raise ValueError("Cannot predict without prior update.")
            step_offset = current_step - self.last_update_step
            output = 0
            for order in range(len(self.taylor_factors)):
                output += self.taylor_factors[order] * (step_offset**order) * (1 / math.factorial(order))
            self.remaining_predictions -= 1
            # output will be converted to the module's dtype
            return output.to(self.module_dtype)


class TaylorSeerAttentionCacheHook(ModelHook):
    """
    Hook for caching and predicting attention outputs using Taylor series approximations.
    Applies to attention modules in diffusion models (e.g., Flux).
    Performs full computations during warmup, then alternates between predictions and refreshes.
    """

    _is_stateful = True

    def __init__(
        self,
        module_name: str,
        predict_steps: int,
        max_order: int,
        warmup_steps: int,
        taylor_factors_dtype: torch.dtype,
        stop_predicts: Optional[int] = None,
        is_skip: bool = False,
    ):
        super().__init__()
        self.module_name = module_name
        self.predict_steps = predict_steps
        self.max_order = max_order
        self.warmup_steps = warmup_steps
        self.stop_predicts = stop_predicts
        self.step_counter = -1
        self.states: Optional[List[TaylorSeerOutputState]] = None
        self.num_outputs: Optional[int] = None
        self.taylor_factors_dtype = taylor_factors_dtype
        self.is_skip = is_skip

    def initialize_hook(self, module: torch.nn.Module):
        self.step_counter = -1
        self.states = None
        self.num_outputs = None
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        self.step_counter += 1
        is_warmup_phase = self.step_counter < self.warmup_steps

        if self.states is None:
            # First step: always full compute and initialize
            attention_outputs = self.fn_ref.original_forward(*args, **kwargs)
            if isinstance(attention_outputs, torch.Tensor):
                attention_outputs = [attention_outputs]
            else:
                attention_outputs = list(attention_outputs)
            module_dtype = attention_outputs[0].dtype
            self.num_outputs = len(attention_outputs)
            self.states = [
                TaylorSeerOutputState(
                    self.module_name, self.taylor_factors_dtype, module_dtype, is_skip=self.is_skip
                )
                for _ in range(self.num_outputs)
            ]
            for i, features in enumerate(attention_outputs):
                self.states[i].update(
                    features, self.step_counter, self.max_order, self.predict_steps, is_first_update=True
                )
            return attention_outputs[0] if self.num_outputs == 1 else tuple(attention_outputs)

        if self.stop_predicts is not None and self.step_counter >= self.stop_predicts:
            # After stop_predicts: always full compute without updating state
            attention_outputs = self.fn_ref.original_forward(*args, **kwargs)
            if isinstance(attention_outputs, torch.Tensor):
                attention_outputs = [attention_outputs]
            else:
                attention_outputs = list(attention_outputs)
            return attention_outputs[0] if self.num_outputs == 1 else tuple(attention_outputs)
        else:
            should_predict = self.states[0].remaining_predictions > 0
            if is_warmup_phase or not should_predict:
                # Full compute during warmup or when refresh needed
                attention_outputs = self.fn_ref.original_forward(*args, **kwargs)
                if isinstance(attention_outputs, torch.Tensor):
                    attention_outputs = [attention_outputs]
                else:
                    attention_outputs = list(attention_outputs)
                is_first_update = self.step_counter == 0  # Only True for the very first step
                for i, features in enumerate(attention_outputs):
                    self.states[i].update(features, self.step_counter, self.max_order, self.predict_steps, is_first_update)
                return attention_outputs[0] if self.num_outputs == 1 else tuple(attention_outputs)
            else:
                # Predict using Taylor series
                predicted_outputs = [state.predict(self.step_counter) for state in self.states]
                return predicted_outputs[0] if self.num_outputs == 1 else tuple(predicted_outputs)

    def reset_state(self, module: torch.nn.Module) -> None:
        self.states = None


def apply_taylorseer_cache(module: torch.nn.Module, config: TaylorSeerCacheConfig):
    """
    Applies the TaylorSeer cache to given pipeline.

    Args:
        module (torch.nn.Module): The model to apply the hook to.
        config (TaylorSeerCacheConfig): Configuration for the cache.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import FluxPipeline, TaylorSeerCacheConfig, apply_taylorseer_cache

    >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> config = TaylorSeerCacheConfig(predict_steps=5, max_order=1, warmup_steps=3, taylor_factors_dtype=torch.float32, architecture="flux")
    >>> apply_taylorseer_cache(pipe.transformer, config)
    ```
    """
    if config.skip_identifiers:
        skip_identifiers = config.skip_identifiers
    else:
        skip_identifiers = _CACHE_TEMPLATES.get(config.architecture, {}).get("skip", [])

    if config.cache_identifiers:
        cache_identifiers = config.cache_identifiers
    else:
        cache_identifiers = _CACHE_TEMPLATES.get(config.architecture, {}).get("cache", [])

    logger.debug(f"Skip identifiers: {skip_identifiers}")
    logger.debug(f"Cache identifiers: {cache_identifiers}")

    for name, submodule in module.named_modules():
        if (skip_identifiers and cache_identifiers) or (cache_identifiers):
            if any(re.fullmatch(identifier, name) for identifier in skip_identifiers) or any(
                re.fullmatch(identifier, name) for identifier in cache_identifiers
            ):
                logger.debug(f"Applying TaylorSeer cache to {name}")
                _apply_taylorseer_cache_hook(name, submodule, config)
        elif isinstance(submodule, (*_ATTENTION_CLASSES, AttentionModuleMixin)):
            logger.debug(f"Applying TaylorSeer cache to {name}")
            _apply_taylorseer_cache_hook(name, submodule, config)


def _apply_taylorseer_cache_hook(name: str, module: Attention, config: TaylorSeerCacheConfig):
    """
    Registers the TaylorSeer hook on the specified attention module.
    Args:
        name (str): Name of the module.
        module (Attention): The attention module.
        config (TaylorSeerCacheConfig): Configuration for the cache.
    """

    is_skip = any(
        re.fullmatch(identifier, name) for identifier in _CACHE_TEMPLATES.get(config.architecture, {}).get("skip", [])
    )

    registry = HookRegistry.check_if_exists_or_initialize(module)

    hook = TaylorSeerAttentionCacheHook(
        name,
        config.predict_steps,
        config.max_order,
        config.warmup_steps,
        config.taylor_factors_dtype,
        stop_predicts=config.stop_predicts,
        is_skip=is_skip,
    )

    registry.register_hook(hook, _TAYLORSEER_ATTENTION_CACHE_HOOK)