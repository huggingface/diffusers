import torch
import torch.nn as nn
import torch.distributed as dist

import copy
import os
import random

from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate


class ForwardEMAModel(EMAModel):
    def __init__(
        self,
        model,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        # super().__init__(
        #     parameters = model.parameters(),
        #     decay = decay,
        #     min_decay = min_decay,
        #     update_after_step = update_after_step,
        #     use_ema_warmup = use_ema_warmup,
        #     inv_gamma = inv_gamma,
        #     power = power,
        #     model_cls = model_cls,
        #     model_config= model_config,
        #     **kwargs,
        # )
        
        target_model = copy.deepcopy(model)
        target_model.train()
        
        sync_params(target_model.parameters())
        sync_params(target_model.buffers())
        for dst, src in zip(target_model.parameters(), model.parameters()):
            dst.data.copy_(src.data)
        
        self.model = target_model
        for param in self.model.parameters():
            param.requires_grad = False
        
        parameters = self.model.parameters()
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        #self.shadow_params = [p.clone().detach() for p in parameters]
        self.shadow_params = [p.detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

        self.model_cls = model_cls
        self.model_config = model_config
        
    
    def __call__(self, *args):
        model_output = self.model(*args)
        return model_output

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            #self.shadow_params = shadow_params
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.data.copy_(param.to(s_param.device).data)
            
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)