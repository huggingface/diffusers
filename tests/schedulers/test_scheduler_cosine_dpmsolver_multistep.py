import unittest
import warnings

import torch

from diffusers import CosineDPMSolverMultistepScheduler

from ..testing_utils import require_torchsde


@require_torchsde
class CosineDPMSolverMultistepSchedulerTest(unittest.TestCase):
    """Regression tests for `CosineDPMSolverMultistepScheduler` (used by Stable Audio Open)."""

    def _run_loop(self, **scheduler_kwargs):
        scheduler = CosineDPMSolverMultistepScheduler(**scheduler_kwargs)
        scheduler.set_timesteps(num_inference_steps=10, device="cpu")
        sample = torch.randn(1, 4, 8)
        generator = torch.Generator().manual_seed(0)
        for t in scheduler.timesteps:
            model_output = torch.randn_like(sample)
            sample = scheduler.step(model_output, t, sample, generator=generator).prev_sample
        return sample

    def test_step_does_not_recurse_with_zero_final_sigma(self):
        # See https://github.com/huggingface/diffusers/issues/13274. With the defaults
        # used by Stable Audio Open (sigma_min=0.3, sigma_max=500, final_sigmas_type="zero")
        # querying the Brownian sampler at sigma_next=0 used to fall below the configured
        # `sigma_min` interval and recurse until Python's recursion limit was hit.
        for sigma_schedule in ("exponential", "karras"):
            with self.subTest(sigma_schedule=sigma_schedule):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample = self._run_loop(
                        sigma_schedule=sigma_schedule,
                        final_sigmas_type="zero",
                        sigma_min=0.3,
                        sigma_max=500.0,
                    )
                self.assertFalse(torch.isnan(sample).any().item())
