import math
import torch
import torch.nn as nn
import numpy as np


class LambdaWarmUpCosineScheduler:
    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0 and n % self.verbosity_interval == 0:
            print(f"step: {n}, lr-multiplier: {self.last_f}, cycle: {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * math.pi))
        self.last_f = f
        return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler):
    """Linear decay instead of cosine decay after warmup."""

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0 and n % self.verbosity_interval == 0:
            print(f"step: {n}, lr-multiplier: {self.last_f}, cycle: {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n
            ) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
        self.last_f = f
        return f

# Copied from cosmos-predict2.5/cosmos_predict2/_src/imaginaire/utils/optim_instantiate.py
def get_base_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a LambdaLR scheduler wrapping LambdaLinearScheduler.

    scheduler_cfg keys (matching LoRA experiment defaults):
        warm_up_steps  (list[int])
        cycle_lengths  (list[int])
        f_start        (list[float])
        f_max          (list[float])
        f_min          (list[float])
    """
    net_scheduler = LambdaLinearScheduler(
        warm_up_steps=scheduler_cfg["warm_up_steps"],
        cycle_lengths=scheduler_cfg["cycle_lengths"],
        f_start=scheduler_cfg.get("f_start", [1e-6]),
        f_max=scheduler_cfg["f_max"],
        f_min=scheduler_cfg["f_min"],
    )

    num_param_groups = len(optimizer.param_groups)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[net_scheduler.schedule] * num_param_groups,
    )


def build_optimizer_and_scheduler(trainable_params):
    optimizer_cfg = dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    )

    scheduler_cfg = dict(
        warm_up_steps=[2000],
        cycle_lengths=[100000],
        f_start=[1e-6],
        f_max=[0.5],
        f_min=[0.2],
    )

    optimizer = torch.optim.AdamW(trainable_params, **optimizer_cfg)
    scheduler = get_base_scheduler(optimizer, scheduler_cfg)
    return optimizer, scheduler
