# Experimental hook for TaylorSeer cache
# Supports Flux only for now

import torch
from dataclasses import dataclass
from typing import Callable
from .hooks import ModelHook
import math
from ..models.attention import Attention
from ..models.attention import AttentionModuleMixin
from ._common import (
    _ATTENTION_CLASSES,
)
from ..hooks import HookRegistry

_TAYLORSEER_ATTENTION_CACHE_HOOK = "taylorseer_attention_cache"

@dataclass
class TaylorSeerCacheConfig:
    fresh_threshold: int = 5 # interleave cache and compute: `fresh_threshold` steps are cached, then 1 full compute step is performed
    max_order: int = 1 # order of Taylor series expansion
    current_timestep_callback: Callable[[], int] = None

class TaylorSeerState:
    def __init__(self):
        self.predict_counter: int = 0
        self.last_step: int = 1000
        self.taylor_factors: dict[int, torch.Tensor] = {}

    def reset(self):
        self.predict_counter = 0
        self.last_step = 1000
        self.taylor_factors = {}

    def update(self, features: torch.Tensor, current_step: int, max_order: int, refresh_threshold: int):
        N = math.abs(current_step - self.last_step)
        # initialize the first order taylor factors
        new_taylor_factors = {0: features}
        for i in range(max_order):
            if (self.taylor_factors.get(i) is not None) and current_step > 1:
                new_taylor_factors[i+1] = (self.taylor_factors[i] - new_taylor_factors[i]) / N
            else:
                break
        self.taylor_factors = new_taylor_factors
        self.last_step = current_step
        self.predict_counter = refresh_threshold

    def predict(self, current_step: int):
        k = current_step - self.last_step
        device = self.taylor_factors[0].device
        output = torch.zeros_like(self.taylor_factors[0], device=device)
        for i in range(len(self.taylor_factors)):
            output += self.taylor_factors[i] * (k ** i) / math.factorial(i)
        self.predict_counter -= 1
        return output

class TaylorSeerAttentionCacheHook(ModelHook):
    _is_stateful = True

    def __init__(self, fresh_threshold: int, max_order: int, current_timestep_callback: Callable[[], int]):
        super().__init__()
        self.fresh_threshold = fresh_threshold
        self.max_order = max_order
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.states = None
        self.num_outputs = None
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        current_step = self.current_timestep_callback()
        assert current_step is not None, "timestep is required for TaylorSeerAttentionCacheHook"

        if self.states is None:
            attention_outputs = self.fn_ref.original_forward(*args, **kwargs)
            self.num_outputs = len(attention_outputs)
            self.states = [TaylorSeerState() for _ in range(self.num_outputs)]
            for i, feat in enumerate(attention_outputs):
                self.states[i].update(feat, current_step, self.max_order, self.fresh_threshold)
            return attention_outputs

        should_predict = self.states[0].predict_counter > 0

        if not should_predict:
            attention_outputs = self.fn_ref.original_forward(*args, **kwargs)
            for i, feat in enumerate(attention_outputs):
                self.states[i].update(feat, current_step, self.max_order, self.fresh_threshold)
            return attention_outputs
        else:
            predicted_outputs = [state.predict(current_step) for state in self.states]
            return tuple(predicted_outputs)

    def reset_state(self, module: torch.nn.Module) -> None:
        if self.states is not None:
            for state in self.states:
                state.reset()
        return module

def apply_taylorseer_cache(module: torch.nn.Module, config: TaylorSeerCacheConfig):
    for name, submodule in module.named_modules():
        if not isinstance(submodule, (*_ATTENTION_CLASSES, AttentionModuleMixin)):
            continue
        print(f"Applying TaylorSeer cache to {name}")
        _apply_taylorseer_cache_on_attention_class(name, submodule, config)


def _apply_taylorseer_cache_on_attention_class(name: str, module: Attention, config: TaylorSeerCacheConfig):
    _apply_taylorseer_cache_hook(module, config)


def _apply_taylorseer_cache_hook(module: Attention, config: TaylorSeerCacheConfig):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TaylorSeerAttentionCacheHook(config.fresh_threshold, config.max_order, config.current_timestep_callback)
    registry.register_hook(hook, _TAYLORSEER_ATTENTION_CACHE_HOOK)