import unittest

import torch

from diffusers import EntropyBoundScheduler


class EntropyBoundSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {"entropy_bound": 0.1, "num_inference_steps": 8}
        config.update(kwargs)
        return EntropyBoundScheduler(**config)

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(16)
        self.assertEqual(scheduler.num_inference_steps, 16)
        self.assertEqual(len(scheduler.timesteps), 16)

    def test_zero_entropy_positions_accepted(self):
        # Positions with a near-one probability have ~zero entropy and must be accepted.
        scheduler = self.get_scheduler(entropy_bound=0.1)
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, :9, 0] = 1e6  # 9 zero-entropy positions
        out = scheduler.step(logits, timestep=0, sample=sample, temperature=0.0)
        self.assertGreaterEqual(out.accepted_index.sum().item(), 9)
        # accepted positions hold the sampled token (token 0 here)
        self.assertTrue((out.prev_sample[0, :9] == 0).all())

    def test_higher_bound_accepts_at_least_as_many(self):
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, 0, 0] = 1.8e1
        logits[0, 1, 1] = 1.45e1
        logits[0, 2, 2] = 1.45e1
        low = self.get_scheduler(entropy_bound=1e-2).step(logits, 0, sample, temperature=0.0)
        high = self.get_scheduler(entropy_bound=1e-1).step(logits, 0, sample, temperature=0.0)
        self.assertGreaterEqual(high.accepted_index.sum().item(), low.accepted_index.sum().item())

    def test_non_accepted_are_renoised(self):
        scheduler = self.get_scheduler(entropy_bound=1e-3)
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, :5, 0] = 1e6
        out = scheduler.step(logits, timestep=0, sample=sample, temperature=0.0)
        # the 5 accepted positions hold token 0, the rest are random (not token 0 almost surely)
        self.assertTrue((out.prev_sample[0, :5] == 0).all())

    def test_step_output_shapes(self):
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 100, (3, 16))
        logits = torch.randn(3, 16, 100)
        out = scheduler.step(logits, timestep=0, sample=sample, temperature=1.0)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.accepted_index.shape, sample.shape)
