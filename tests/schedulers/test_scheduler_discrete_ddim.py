import unittest

import torch

from diffusers import DiscreteDDIMScheduler


class DiscreteDDIMSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {"num_inference_steps": 8}
        config.update(kwargs)
        return DiscreteDDIMScheduler(**config)

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(16)
        self.assertEqual(scheduler.num_inference_steps, 16)
        self.assertEqual(len(scheduler.timesteps), 16)
        self.assertEqual(scheduler.timesteps[0].item(), 0)
        self.assertEqual(scheduler.timesteps[-1].item(), 15)

    def test_set_timesteps_invalid(self):
        scheduler = self.get_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(0)

    def test_last_step_commits_predicted_tokens(self):
        # On the final step alpha_s = 1, so the posterior deterministically commits the sampled clean tokens.
        n = 8
        scheduler = self.get_scheduler(num_inference_steps=n)
        scheduler.set_timesteps(n)
        sample = torch.randint(0, 100, (2, 16))
        logits = torch.zeros(2, 16, 100)
        out = scheduler.step(logits, timestep=n - 1, sample=sample, temperature=0.0)
        self.assertTrue(torch.equal(out.prev_sample, out.sampled_tokens))

    def test_intermediate_step_keeps_agreeing_positions(self):
        # Where the prediction agrees with the current token, almost all posterior mass is on the clean route.
        n = 8
        scheduler = self.get_scheduler(num_inference_steps=n)
        scheduler.set_timesteps(n)
        sample = torch.randint(0, 100, (1, 256))
        logits = torch.zeros(1, 256, 100)
        # argmax of zero logits is token 0; make the sample already equal token 0 everywhere
        sample = torch.zeros_like(sample)
        out = scheduler.step(logits, timestep=n // 2, sample=sample, temperature=0.0)
        kept = (out.prev_sample == sample).sum().item()
        self.assertGreaterEqual(kept, 250)

    def test_step_output_shapes(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (3, 16))
        logits = torch.randn(3, 16, 100)
        out = scheduler.step(logits, timestep=2, sample=sample, temperature=1.0)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.sampled_tokens.shape, sample.shape)
        self.assertEqual(out.sampled_probs.shape, sample.shape)

    def test_return_tuple(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (1, 16))
        logits = torch.randn(1, 16, 100)
        out = scheduler.step(logits, timestep=2, sample=sample, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)

    def test_to_loo_only_shifts_observed_token(self):
        # The denoiser->LOO conversion moves only the observed token's logit at each position (eq. 13).
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 100, (2, 16))
        logits = torch.randn(2, 16, 100)
        loo = scheduler._to_loo_logits(logits, sample, alpha=0.4)
        diff = loo - logits
        moved = diff.abs() > 0
        self.assertTrue(torch.equal(moved.sum(dim=-1), torch.ones(2, 16, dtype=torch.long)))

    def test_step_correct_output_shapes(self):
        scheduler = self.get_scheduler(corrector_steps=1, corrector_k=4)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (3, 16))
        logits = torch.randn(3, 16, 100)
        out = scheduler.step_correct(logits, timestep=2, sample=sample)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.prev_sample.dtype, sample.dtype)

    def test_step_correct_resamples_at_most_k(self):
        # A corrector sweep holds all but `corrector_k` positions per row fixed.
        k = 3
        scheduler = self.get_scheduler(corrector_steps=1, corrector_k=k)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (4, 16))
        logits = torch.randn(4, 16, 100)
        out = scheduler.step_correct(logits, timestep=2, sample=sample)
        changed = (out.prev_sample != sample).sum(dim=-1)
        self.assertTrue(torch.all(changed <= k))

    def test_step_correct_return_tuple(self):
        scheduler = self.get_scheduler(corrector_steps=1)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (1, 16))
        logits = torch.randn(1, 16, 100)
        out = scheduler.step_correct(logits, timestep=2, sample=sample, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
