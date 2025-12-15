import tempfile
import unittest

import numpy as np
import torch

from diffusers import FlowUniPCMultistepScheduler


class FlowUniPCMultistepSchedulerKarrasTest(unittest.TestCase):
    def test_set_timesteps_with_karras_sigmas(self):
        num_inference_steps = 4
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            solver_order=2,
        )

        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # TODO: use constants for sigmas and timesteps
        sigma_max, sigma_min, rho = (
            scheduler.config.sigma_max,
            scheduler.config.sigma_min,
            scheduler.config.rho,
        )
        ramp = np.arange(num_inference_steps + 1, dtype=np.float32) / num_inference_steps
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        expected_sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        expected_sigmas = expected_sigmas / (1 + expected_sigmas)
        expected_sigmas = torch.from_numpy(expected_sigmas.astype(np.float32))

        # FlowUniPCMultistepScheduler appends a terminal sigma of zero after conversion to torch.Tensor.
        self.assertEqual(scheduler.sigmas.shape[0], expected_sigmas.shape[0] + 1)
        self.assertTrue(torch.allclose(scheduler.sigmas[:-1], expected_sigmas, atol=1e-6))

        expected_timesteps = torch.from_numpy(
            (expected_sigmas.numpy() * scheduler.config.num_train_timesteps).astype(np.int64)
        )
        self.assertTrue(torch.equal(scheduler.timesteps, expected_timesteps))
        self.assertEqual(scheduler.sigmas[-1].item(), 0.0)

    def test_set_timesteps_with_custom_karras_sigmas(self):
        num_inference_steps = 3
        sigma_max, sigma_min, rho = 50.0, 0.005, 5.0
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            solver_order=2,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
        )

        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # TODO: use constants for sigmas and timesteps
        ramp = np.arange(num_inference_steps + 1, dtype=np.float32) / num_inference_steps
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        expected_sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        expected_sigmas = expected_sigmas / (1 + expected_sigmas)
        expected_sigmas = torch.from_numpy(expected_sigmas.astype(np.float32))

        self.assertEqual(scheduler.sigmas.shape[0], expected_sigmas.shape[0] + 1)
        self.assertTrue(torch.allclose(scheduler.sigmas[:-1], expected_sigmas, atol=1e-6))
        expected_timesteps = torch.from_numpy(
            (expected_sigmas.numpy() * scheduler.config.num_train_timesteps).astype(np.int64)
        )
        self.assertTrue(torch.equal(scheduler.timesteps, expected_timesteps))
        self.assertEqual(scheduler.sigmas[-1].item(), 0.0)

    # TODO: add test
    # def test_timesteps_respected_when_steps_match_train(self):
    #     scheduler = FlowUniPCMultistepScheduler(
    #         num_train_timesteps=8,
    #         solver_order=2,
    #     )
    #     before_sigmas = scheduler.sigmas.clone()
    #     scheduler.set_timesteps(num_inference_steps=scheduler.config.num_train_timesteps)
    #     self.assertTrue(torch.allclose(scheduler.sigmas[:-1], before_sigmas))
    #     self.assertEqual(scheduler.sigmas[-1].item(), 0.0)

    def test_step_preserves_dtype_and_device(self):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=10,
            solver_order=2,
        )
        scheduler.set_timesteps(num_inference_steps=4, device="cpu")

        sample = torch.randn(2, 3, 4, dtype=torch.float16)
        residual = torch.randn_like(sample)
        timestep = scheduler.timesteps[0]

        output = scheduler.step(residual, timestep, sample).prev_sample
        self.assertEqual(output.dtype, sample.dtype)
        self.assertEqual(output.device, sample.device)

    def test_save_and_load_round_trip(self):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=12,
            solver_order=2,
        )
        scheduler.set_timesteps(num_inference_steps=6)

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded = FlowUniPCMultistepScheduler.from_pretrained(tmpdir)

        loaded.set_timesteps(num_inference_steps=6)
        self.assertTrue(torch.equal(scheduler.timesteps, loaded.timesteps))
        self.assertTrue(torch.allclose(scheduler.sigmas, loaded.sigmas))

    def test_full_loop_no_nan(self):
        torch.manual_seed(0)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=16,
            solver_order=2,
            sigma_max=1.0,
            sigma_min=0.01,
        )
        scheduler.set_timesteps(num_inference_steps=6)

        def model(sample, t):
            return 0.05 * torch.tanh(sample)

        sample = torch.ones(2, 3, 4)
        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        self.assertFalse(torch.isnan(sample).any())
        self.assertEqual(sample.shape, (2, 3, 4))