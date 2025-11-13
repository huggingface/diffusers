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
        self.predict_counter: int = 1
        self.last_step: int = 1000
        self.taylor_factors: dict[int, torch.Tensor] = {}

    def reset(self):
        self.predict_counter = 1
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
        self.predict_counter = (self.predict_counter + 1) % refresh_threshold

    def predict(self, current_step: int, refresh_threshold: int):
        k = current_step - self.last_step
        device = self.taylor_factors[0].device
        output = torch.zeros_like(self.taylor_factors[0], device=device)
        for i in range(len(self.taylor_factors)):
            output += self.taylor_factors[i] * (k ** i) / math.factorial(i)
        self.predict_counter = (self.predict_counter + 1) % refresh_threshold
        return output

class TaylorSeerAttentionCacheHook(ModelHook):
    _is_stateful = True

    def __init__(self, fresh_threshold: int, max_order: int, current_timestep_callback: Callable[[], int]):
        super().__init__()
        self.fresh_threshold = fresh_threshold
        self.max_order = max_order
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.img_state = TaylorSeerState()
        self.txt_state = TaylorSeerState()
        self.ip_state = TaylorSeerState()
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        current_step = self.current_timestep_callback()
        assert current_step is not None, "timestep is required for TaylorSeerAttentionCacheHook"
        should_predict = self.img_state.predict_counter > 0

        if not should_predict:
            attention_outputs = self.fn_ref.original_forward(*args, **kwargs)
            if len(attention_outputs) == 2:
                attn_output, context_attn_output = attention_outputs
                self.img_state.update(attn_output, current_step, self.max_order, self.fresh_threshold)
                self.txt_state.update(context_attn_output, current_step, self.max_order, self.fresh_threshold)
            elif len(attention_outputs) == 3:
                attn_output, context_attn_output, ip_attn_output = attention_outputs
                self.img_state.update(attn_output, current_step, self.max_order, self.fresh_threshold)
                self.txt_state.update(context_attn_output, current_step, self.max_order, self.fresh_threshold)
                self.ip_state.update(ip_attn_output, current_step, self.max_order, self.fresh_threshold)
        else:
            attn_output = self.img_state.predict(current_step, self.fresh_threshold)
            context_attn_output = self.txt_state.predict(current_step, self.fresh_threshold)
            ip_attn_output = self.ip_state.predict(current_step, self.fresh_threshold)
            attention_outputs = (attn_output, context_attn_output, ip_attn_output)
            return attention_outputs

    def reset_state(self, module: torch.nn.Module) -> None:
        self.img_state.reset()
        self.txt_state.reset()
        self.ip_state.reset()
        return module

def apply_taylorseer_cache(module: torch.nn.Module, config: TaylorSeerCacheConfig):
    for name, submodule in module.named_modules():
        if not isinstance(submodule, (*_ATTENTION_CLASSES, AttentionModuleMixin)):
            # PAB has been implemented specific to Diffusers' Attention classes. However, this does not mean that PAB
            # cannot be applied to this layer. For custom layers, users can extend this functionality and implement
            # their own PAB logic similar to `_apply_pyramid_attention_broadcast_on_attention_class`.
            continue
        _apply_taylorseer_cache_on_attention_class(name, submodule, config)


def _apply_taylorseer_cache_on_attention_class(name: str, module: Attention, config: TaylorSeerCacheConfig):
    _apply_taylorseer_cache_hook(module, config)


def _apply_taylorseer_cache_hook(module: Attention, config: TaylorSeerCacheConfig):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TaylorSeerAttentionCacheHook(config.fresh_threshold, config.max_order, config.current_timestep_callback)
    registry.register_hook(hook, _TAYLORSEER_ATTENTION_CACHE_HOOK)