import torch
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
from .hooks import ModelHook
import math
from ..models.attention import Attention
from ..models.attention import AttentionModuleMixin
from ._common import (
    _ATTENTION_CLASSES,
)
from ..hooks import HookRegistry
from ..utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
_TAYLORSEER_ATTENTION_CACHE_HOOK = "taylorseer_attention_cache"

@dataclass
class TaylorSeerCacheConfig:
    """
    Configuration for TaylorSeer cache.
    See: https://huggingface.co/papers/2503.06923

    Attributes:
        warmup_steps (int, defaults to 3): Number of warmup steps without caching.
        predict_steps (int, defaults to 5): Number of prediction (cache) steps between non-cached steps.
        max_order (int, defaults to 1): Maximum order of Taylor series expansion to approximate the features.
        taylor_factors_dtype (torch.dtype, defaults to torch.float32): Data type for Taylor series expansion factors.
    """
    warmup_steps: int = 3
    predict_steps: int = 5
    max_order: int = 1
    taylor_factors_dtype: torch.dtype = torch.float32

    def __repr__(self) -> str:
        return f"TaylorSeerCacheConfig(warmup_steps={self.warmup_steps}, predict_steps={self.predict_steps}, max_order={self.max_order}, taylor_factors_dtype={self.taylor_factors_dtype})"

class TaylorSeerOutputState:
    """
    Manages the state for Taylor series-based prediction of a single attention output.
    Tracks Taylor expansion factors, last update step, and remaining prediction steps.
    The Taylor expansion uses the timestep as the independent variable for approximation.
    """

    def __init__(self, module_name: str, taylor_factors_dtype: torch.dtype, module_dtype: torch.dtype):
        self.module_name = module_name
        self.remaining_predictions: int = 0
        self.last_update_step: Optional[int] = None
        self.taylor_factors: Dict[int, torch.Tensor] = {}
        self.taylor_factors_dtype = taylor_factors_dtype
        self.module_dtype = module_dtype

    def reset(self):
        self.remaining_predictions = 0
        self.last_update_step = None
        self.taylor_factors = {}

    def update(self, features: torch.Tensor, current_step: int, max_order: int, predict_steps: int, is_first_update: bool):
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
                    # Finite difference: (current - previous) / delta for forward approximation
                    new_factors[i + 1] = (new_factors[i] - self.taylor_factors[i].to(self.taylor_factors_dtype)) / delta_step

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
        if self.last_update_step is None:
            raise ValueError("Cannot predict without prior update.")
        step_offset = current_step - self.last_update_step
        device = self.taylor_factors[0].device
        output = torch.zeros_like(self.taylor_factors[0], device=device, dtype=self.taylor_factors_dtype)
        for order in range(len(self.taylor_factors)):
            output += self.taylor_factors[order] * (step_offset ** order) / math.factorial(order)
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
        module_dtype: torch.dtype = None,
    ):
        super().__init__()
        self.module_name = module_name
        self.predict_steps = predict_steps
        self.max_order = max_order
        self.warmup_steps = warmup_steps
        self.step_counter = -1
        self.states: Optional[List[TaylorSeerOutputState]] = None
        self.num_outputs: Optional[int] = None
        self.taylor_factors_dtype = taylor_factors_dtype
        self.module_dtype = module_dtype

    def initialize_hook(self, module: torch.nn.Module):
        self.step_counter = -1
        self.states = None
        self.num_outputs = None
        self.module_dtype = None
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
                TaylorSeerOutputState(self.module_name, self.taylor_factors_dtype, module_dtype)
                for _ in range(self.num_outputs)
            ]
            for i, features in enumerate(attention_outputs):
                self.states[i].update(features, self.step_counter, self.max_order, self.predict_steps, is_first_update=True)
            return attention_outputs[0] if self.num_outputs == 1 else tuple(attention_outputs)

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
        if self.states is not None:
            for state in self.states:
                state.reset()

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

    >>> config = TaylorSeerCacheConfig(predict_steps=5, max_order=1, warmup_steps=3, taylor_factors_dtype=torch.float32)
    >>> apply_taylorseer_cache(pipe.transformer, config)
    ```
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, (*_ATTENTION_CLASSES, AttentionModuleMixin)):
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
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TaylorSeerAttentionCacheHook(
        name,
        config.predict_steps,
        config.max_order,
        config.warmup_steps,
        config.taylor_factors_dtype,
    )
    registry.register_hook(hook, _TAYLORSEER_ATTENTION_CACHE_HOOK)