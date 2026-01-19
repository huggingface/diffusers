import copy
import inspect
from typing import Any, List, Optional, Union

import torch


class BaseAsyncScheduler:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler

    def __getattr__(self, name: str):
        if hasattr(self.scheduler, name):
            return getattr(self.scheduler, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name == "scheduler":
            super().__setattr__(name, value)
        else:
            if hasattr(self, "scheduler") and hasattr(self.scheduler, name):
                setattr(self.scheduler, name, value)
            else:
                super().__setattr__(name, value)

    def clone_for_request(self, num_inference_steps: int, device: Union[str, torch.device, None] = None, **kwargs):
        local = copy.deepcopy(self.scheduler)
        local.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        cloned = self.__class__(local)
        return cloned

    def __repr__(self):
        return f"BaseAsyncScheduler({repr(self.scheduler)})"

    def __str__(self):
        return f"BaseAsyncScheduler wrapping: {str(self.scheduler)}"


def async_retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.
    Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Backwards compatible: by default the function behaves exactly as before and returns
        (timesteps_tensor, num_inference_steps)

    If the caller passes `return_scheduler=True` in kwargs, the function will **not** mutate the passed
    scheduler. Instead it will use a cloned scheduler if available (via `scheduler.clone_for_request`)
    or a deepcopy fallback, call `set_timesteps` on that cloned scheduler, and return:
        (timesteps_tensor, num_inference_steps, scheduler_in_use)

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Optional kwargs:
        return_scheduler (bool, default False): if True, return (timesteps, num_inference_steps, scheduler_in_use)
            where `scheduler_in_use` is a scheduler instance that already has timesteps set.
            This mode will prefer `scheduler.clone_for_request(...)` if available, to avoid mutating the original scheduler.

    Returns:
        `(timesteps_tensor, num_inference_steps)` by default (backwards compatible), or
        `(timesteps_tensor, num_inference_steps, scheduler_in_use)` if `return_scheduler=True`.
    """
    # pop our optional control kwarg (keeps compatibility)
    return_scheduler = bool(kwargs.pop("return_scheduler", False))

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")

    # choose scheduler to call set_timesteps on
    scheduler_in_use = scheduler
    if return_scheduler:
        # Do not mutate the provided scheduler: prefer to clone if possible
        if hasattr(scheduler, "clone_for_request"):
            try:
                # clone_for_request may accept num_inference_steps or other kwargs; be permissive
                scheduler_in_use = scheduler.clone_for_request(
                    num_inference_steps=num_inference_steps or 0, device=device
                )
            except Exception:
                scheduler_in_use = copy.deepcopy(scheduler)
        else:
            # fallback deepcopy (scheduler tends to be smallish - acceptable)
            scheduler_in_use = copy.deepcopy(scheduler)

    # helper to test if set_timesteps supports a particular kwarg
    def _accepts(param_name: str) -> bool:
        try:
            return param_name in set(inspect.signature(scheduler_in_use.set_timesteps).parameters.keys())
        except (ValueError, TypeError):
            # if signature introspection fails, be permissive and attempt the call later
            return False

    # now call set_timesteps on the chosen scheduler_in_use (may be original or clone)
    if timesteps is not None:
        accepts_timesteps = _accepts("timesteps")
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler_in_use.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler_in_use.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps_out = scheduler_in_use.timesteps
        num_inference_steps = len(timesteps_out)
    elif sigmas is not None:
        accept_sigmas = _accepts("sigmas")
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler_in_use.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler_in_use.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps_out = scheduler_in_use.timesteps
        num_inference_steps = len(timesteps_out)
    else:
        # default path
        scheduler_in_use.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps_out = scheduler_in_use.timesteps

    if return_scheduler:
        return timesteps_out, num_inference_steps, scheduler_in_use
    return timesteps_out, num_inference_steps
