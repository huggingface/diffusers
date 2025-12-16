import tempfile
import unittest

import torch

from diffusers import FlowUniPCMultistepScheduler


class FlowUniPCMultistepSchedulerKarrasTest(unittest.TestCase):
    def test_set_timesteps(self):
        num_inference_steps = 4
        num_train_timesteps = 1000
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            solver_order=2,
        )
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # 0 appended to end for sigmas
        expected_sigmas = [0.9950248599052429, 0.9787454605102539, 0.8774884343147278, 0.3604971766471863, 0.009900986216962337, 0.0]
        expected_sigmas = torch.tensor(expected_sigmas)
        expected_timesteps = (expected_sigmas * num_train_timesteps).to(torch.int64)
        expected_timesteps = expected_timesteps[0:-1]
        self.assertTrue(torch.allclose(scheduler.sigmas, expected_sigmas))
        self.assertTrue(torch.all(expected_timesteps == scheduler.timesteps))


    def test_inference_train_same_schedule(self):
        num_inference_steps = 4
        num_train_timesteps = num_inference_steps
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            solver_order=2,
        )
        before_sigmas = scheduler.sigmas.clone()
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        after_sigmas = scheduler.sigmas

        self.assertTrue(torch.allclose(before_sigmas, after_sigmas))

    def test_set_timesteps_with_nondefault_args(self):
        num_inference_steps = 4
        scheduler = FlowUniPCMultistepScheduler(
            sigma_max=50.0,
            sigma_min=0.005,
            rho=5.0,
            final_sigmas_type="sigma_min",
        )

        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        expected_sigmas = torch.tensor([0.9803921580314636,
                                        0.9388325214385986,
                                        0.7652841210365295,
                                        0.2545345723628998,
                                        0.004975131247192621,
                                        0.004975131247192621])
        self.assertTrue(torch.allclose(scheduler.sigmas, expected_sigmas))

    def test_step(self):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=10,
            solver_order=2,
        )
        scheduler.set_timesteps(num_inference_steps=4, device="cpu")

        sample = torch.randn(2, 3, 4, dtype=torch.float16)
        residual = torch.randn_like(sample)
        timestep = scheduler.timesteps[0]

        output = scheduler.step(residual, timestep, sample).prev_sample
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual(output.dtype, sample.dtype)
        self.assertEqual(output.device, sample.device)

    def test_save_and_load_round_trip(self):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=12,
            solver_order=2,
            sigma_max=50.0,
            sigma_min=0.005,
            rho=5.0,
            final_sigmas_type="sigma_min",
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
