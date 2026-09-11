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
        # Continuous corruption level: 1 is fully noised, 0 is clean, and the grid decreases.
        self.assertEqual(scheduler.timesteps.dtype, torch.float32)
        self.assertEqual(scheduler.timesteps[0].item(), 1.0)
        self.assertEqual(scheduler.timesteps[-1].item(), 1.0 / 16)
        self.assertTrue(bool((scheduler.timesteps[1:] < scheduler.timesteps[:-1]).all()))

    def test_step_index_round_trips_every_timestep(self):
        # A pipeline iterating `scheduler.timesteps` must recover the matching index for each entry.
        for num_inference_steps in (1, 3, 8, 31):
            scheduler = self.get_scheduler()
            scheduler.set_timesteps(num_inference_steps)
            self.assertIsNone(scheduler.step_index)
            for index, timestep in enumerate(scheduler.timesteps):
                scheduler._step_index = None
                scheduler._init_step_index(timestep)
                self.assertEqual(scheduler.step_index, index)

    def test_set_begin_index(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        scheduler.set_begin_index(3)
        scheduler._init_step_index(scheduler.timesteps[0])
        self.assertEqual(scheduler.step_index, 3)

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
        out = scheduler.step(logits, timestep=scheduler.timesteps[n - 1], sample=sample)
        self.assertTrue(torch.equal(out.prev_sample, out.pred_original_sample))
        # alpha_s = 1 leaves the clean route as the only one with mass, so every position commits.
        self.assertTrue(bool(out.committed_mask.all()))

    def test_intermediate_step_keeps_agreeing_positions(self):
        # Where the prediction agrees with the current token, almost all posterior mass is on the clean route.
        n = 8
        scheduler = self.get_scheduler(num_inference_steps=n)
        scheduler.set_timesteps(n)
        sample = torch.randint(0, 100, (1, 256))
        logits = torch.zeros(1, 256, 100)
        # argmax of zero logits is token 0; make the sample already equal token 0 everywhere
        sample = torch.zeros_like(sample)
        out = scheduler.step(logits, timestep=scheduler.timesteps[n // 2], sample=sample)
        kept = (out.prev_sample == sample).sum().item()
        self.assertGreaterEqual(kept, 250)

    def test_step_output_shapes(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (3, 16))
        logits = torch.randn(3, 16, 100)
        out = scheduler.step(logits, timestep=scheduler.timesteps[2], sample=sample)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.pred_original_sample.shape, sample.shape)
        self.assertEqual(out.sampled_probs.shape, sample.shape)
        self.assertEqual(out.committed_mask.shape, sample.shape)
        self.assertIsNone(out.edited_mask)

    def test_return_tuple(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (1, 16))
        logits = torch.randn(1, 16, 100)
        out = scheduler.step(logits, timestep=scheduler.timesteps[2], sample=sample, return_dict=False)
        self.assertIsInstance(out, tuple)
        # Fixed arity, always including the trailing `edited_mask` even when it is None (matches AmusedScheduler).
        self.assertEqual(len(out), 6)
        self.assertIsNone(out[-1])

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
        out = scheduler.step_correct(logits, timestep=scheduler.timesteps[2], sample=sample)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.prev_sample.dtype, sample.dtype)
        # A sweep reports at full sequence length, not `(batch, corrector_k)`.
        self.assertEqual(out.pred_original_sample.shape, sample.shape)
        self.assertEqual(out.sampled_probs.shape, sample.shape)
        self.assertEqual(out.committed_mask.shape, sample.shape)

    def test_step_correct_resamples_at_most_k(self):
        # A corrector sweep holds all but `corrector_k` positions per row fixed.
        k = 3
        scheduler = self.get_scheduler(corrector_steps=1, corrector_k=k)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (4, 16))
        logits = torch.randn(4, 16, 100)
        out = scheduler.step_correct(logits, timestep=scheduler.timesteps[2], sample=sample)
        changed = (out.prev_sample != sample).sum(dim=-1)
        self.assertTrue(torch.all(changed <= k))
        self.assertTrue(torch.all(out.committed_mask.sum(dim=-1) == k))

    def test_step_correct_return_tuple(self):
        scheduler = self.get_scheduler(corrector_steps=1)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (1, 16))
        logits = torch.randn(1, 16, 100)
        out = scheduler.step_correct(logits, timestep=scheduler.timesteps[2], sample=sample, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 6)
        self.assertIsNone(out[-1])

    def test_step_advances_step_index(self):
        n = 4
        scheduler = self.get_scheduler(num_inference_steps=n)
        scheduler.set_timesteps(n)
        sample = torch.randint(0, 100, (1, 8))
        logits = torch.randn(1, 8, 100)
        for index, timestep in enumerate(scheduler.timesteps):
            out = scheduler.step(logits, timestep=timestep, sample=sample)
            sample = out.prev_sample
            self.assertEqual(scheduler.step_index, index + 1)

    def test_step_correct_is_independent_of_step_index(self):
        # The sweep resolves its grid point from `timestep`, so running it before or after predictor steps
        # have advanced `step_index` must give the same result.
        scheduler = self.get_scheduler(corrector_steps=1, corrector_k=2)
        scheduler.set_timesteps(8)
        sample = torch.randint(0, 100, (2, 16))
        logits = torch.randn(2, 16, 100)
        timestep = scheduler.timesteps[2]

        fresh = scheduler.step_correct(
            logits, timestep=timestep, sample=sample, generator=torch.Generator().manual_seed(0)
        )
        scheduler._step_index = 5
        advanced = scheduler.step_correct(
            logits, timestep=timestep, sample=sample, generator=torch.Generator().manual_seed(0)
        )
        self.assertTrue(torch.equal(fresh.prev_sample, advanced.prev_sample))
