import math
from typing import List, Optional, Union

import numpy as np
import torch

from ..sigmas.beta_sigmas import BetaSigmas
from ..sigmas.exponential_sigmas import ExponentialSigmas
from ..sigmas.karras_sigmas import KarrasSigmas


class FlowMatchSD3:
    def __call__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int,
        shift: float,
        use_dynamic_shifting: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        This is different to others that directly calculate `sigmas`.
        It needs `sigma_min` and `sigma_max` after shift
        https://github.com/huggingface/diffusers/blob/0ed09a17bbab784a78fb163b557b4827467b0468/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L89-L95
        Then we calculate `sigmas` from that `sigma_min` and `sigma_max`.
        https://github.com/huggingface/diffusers/blob/0ed09a17bbab784a78fb163b557b4827467b0468/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L238-L240
        Shifting happens again after (outside of this).
        https://github.com/huggingface/diffusers/blob/0ed09a17bbab784a78fb163b557b4827467b0468/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L248-L251
        """
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()
        timesteps = np.linspace(sigma_max * num_train_timesteps, sigma_min * num_train_timesteps, num_inference_steps)
        sigmas = timesteps / num_train_timesteps
        return sigmas


class FlowMatchFlux:
    def __call__(self, num_inference_steps: int, **kwargs) -> np.ndarray:
        return np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)


class FlowMatchLinearQuadratic:
    def __call__(
        self, num_inference_steps: int, threshold_noise: float = 0.25, linear_steps: Optional[int] = None, **kwargs
    ) -> np.ndarray:
        if linear_steps is None:
            linear_steps = num_inference_steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * num_inference_steps
        quadratic_steps = num_inference_steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_inference_steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
        sigma_schedule = np.array([1.0 - x for x in sigma_schedule]).astype(np.float32)
        return sigma_schedule


class FlowMatchHunyuanVideo:
    def __call__(self, num_inference_steps: int, **kwargs) -> np.ndarray:
        return np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1].copy()


class FlowMatchSANA:
    def __call__(self, num_inference_steps: int, num_train_timesteps: int, shift: float, **kwargs) -> np.ndarray:
        alphas = np.linspace(1, 1 / num_train_timesteps, num_inference_steps + 1)
        sigmas = 1.0 - alphas
        sigmas = np.flip(shift * sigmas / (1 + (shift - 1) * sigmas))[:-1].copy()
        return sigmas


BASE_SCHEDULE_MAP = {
    "FlowMatchHunyuanVideo": FlowMatchHunyuanVideo,
    "FlowMatchLinearQuadratic": FlowMatchLinearQuadratic,
    "FlowMatchFlux": FlowMatchFlux,
    "FlowMatchSD3": FlowMatchSD3,
    "FlowMatchSANA": FlowMatchSANA,
}


class FlowMatchSchedule:
    scale_model_input = False

    base_schedules = BASE_SCHEDULE_MAP

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        base_schedule: Optional[Union[str]] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        **kwargs,
    ):
        self.set_base_schedule(base_schedule)
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.invert_sigmas = invert_sigmas
        self.shift_terminal = shift_terminal
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def set_base_schedule(self, base_schedule: Union[str]):
        if base_schedule is None:
            raise ValueError("Must set base schedule.")
        if isinstance(base_schedule, str):
            if base_schedule not in self.base_schedules:
                raise ValueError(f"Expected one of {self.base_schedules.keys()}")
            _class = self.base_schedules[base_schedule]
            self.base_schedule = _class()
        else:
            self.base_schedule = base_schedule()

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Stretches and shifts the timestep schedule to ensure it terminates at the configured `shift_terminal` config
        value.

        Reference:
        https://github.com/Lightricks/LTX-Video/blob/a01a171f8fe3d99dce2728d60a73fecf4d4238ae/ltx_video/schedulers/rf.py#L51

        Args:
            t (`torch.Tensor`):
                A tensor of timesteps to be stretched and shifted.

        Returns:
            `torch.Tensor`:
                A tensor of adjusted timesteps such that the final value equals `self.shift_terminal`.
        """
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def __call__(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        sigma_schedule: Optional[Union[KarrasSigmas, ExponentialSigmas, BetaSigmas]] = None,
        mu: Optional[float] = None,
        shift: Optional[float] = None,
    ):
        shift = shift or self.shift
        if self.use_dynamic_shifting and mu is None:
            raise ValueError("You have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            sigmas = self.base_schedule(
                num_inference_steps=num_inference_steps,
                num_train_timesteps=self.num_train_timesteps,
                shift=shift,
                use_dynamic_shifting=self.use_dynamic_shifting,
            )
        else:
            # NOTE: current usage is **without** `sigma_last` - different than BetaSchedule
            sigmas = np.array(sigmas).astype(np.float32)

        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        if sigma_schedule is not None:
            sigmas = sigma_schedule(sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.num_train_timesteps

        if self.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        return sigmas, timesteps
