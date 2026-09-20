# TEMPLATE — the /scaffold command copies this to
# src/diffusers/schedulers/scheduling_<snake>.py and renames the class.
# It is convention-correct (passes the gate); the numerical method is left as a
# TODO because that is the engineer's actual work, not something to fabricate.
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class EulerLiteScheduler(SchedulerMixin, ConfigMixin):
    """Contract-only Euler-style scheduler scaffold (no sampler math yet)."""

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
        self.num_inference_steps: Optional[int] = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device, None] = None):
        """Set the discrete timesteps used for the denoising loop.

        Args:
            num_inference_steps: Number of diffusion steps used at inference.
            device: Device the timesteps should be moved to.
        """
        self.num_inference_steps = num_inference_steps
        step = self.config.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step).round()[::-1].clone()
        self.timesteps = timesteps.to(device) if device is not None else timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample at the previous timestep.

        Args:
            model_output: Direct output from the learned diffusion model.
            timestep: The current discrete timestep in the diffusion chain.
            sample: A current instance of a sample created by the diffusion process.
            generator: A torch.Generator for reproducible sampling.
            return_dict: Whether to return a SchedulerOutput or a plain tuple.

        Returns:
            SchedulerOutput or tuple with the predicted previous sample.
        """
        noise = torch.randn(
            sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
        )
        # TODO(engineer): replace this placeholder with the real update rule.
        prev_sample = sample - model_output + 0.0 * noise

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
