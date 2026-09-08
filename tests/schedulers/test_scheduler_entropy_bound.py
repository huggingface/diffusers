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
        # Same decreasing corruption-level grid as DiscreteDDIMScheduler, so the two are interchangeable.
        self.assertEqual(scheduler.timesteps.dtype, torch.float32)
        self.assertEqual(scheduler.timesteps[0].item(), 1.0)
        self.assertEqual(scheduler.timesteps[-1].item(), 1.0 / 16)
        self.assertTrue(bool((scheduler.timesteps[1:] < scheduler.timesteps[:-1]).all()))

    def test_step_index_round_trips_every_timestep(self):
        for num_inference_steps in (1, 3, 8, 31):
            scheduler = self.get_scheduler()
            scheduler.set_timesteps(num_inference_steps)
            for index, timestep in enumerate(scheduler.timesteps):
                scheduler._step_index = None
                scheduler._init_step_index(timestep)
                self.assertEqual(scheduler.step_index, index)

    def test_a_temperature_that_could_reach_zero_raises(self):
        # The annealed temperature divides the logits, so it must stay positive across the whole schedule.
        # It moves linearly between the two ends, so both have to be bounded: `t_max=0` pins it at zero, and a
        # negative `t_min` crosses zero mid-schedule (t_max=1, t_min=-1, n=4 hits exactly 0.0 on step 2).
        for kwargs in ({"t_max": 0.0, "t_min": 0.0}, {"t_max": -1.0}, {"t_max": 1.0, "t_min": -1.0}):
            with self.assertRaises(ValueError, msg=kwargs):
                EntropyBoundScheduler(**kwargs)

    def test_t_min_zero_stays_positive_across_the_schedule(self):
        # `t_min` is only approached in the limit, so 0 is legal and the logits stay finite.
        scheduler = EntropyBoundScheduler(entropy_bound=0.1, t_max=0.8, t_min=0.0, num_inference_steps=4)
        scheduler.set_timesteps(4)
        sample = torch.arange(6).view(1, 6)
        logits = torch.randn(1, 6, 12, generator=torch.Generator().manual_seed(0))
        for timestep in scheduler.timesteps:
            out = scheduler.step(logits, timestep, sample)
            self.assertTrue(bool(torch.isfinite(out.pred_logits).all()))
            sample = out.prev_sample

    def test_zero_entropy_positions_accepted(self):
        # Positions with a near-one probability have ~zero entropy and must be accepted.
        scheduler = self.get_scheduler(entropy_bound=0.1)
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, :9, 0] = 1e6  # 9 zero-entropy positions
        out = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)
        self.assertGreaterEqual(out.committed_mask.sum().item(), 9)
        # accepted positions hold the sampled token (token 0 here)
        self.assertTrue((out.prev_sample[0, :9] == 0).all())

    def test_higher_bound_accepts_at_least_as_many(self):
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, 0, 0] = 1.8e1
        logits[0, 1, 1] = 1.45e1
        logits[0, 2, 2] = 1.45e1
        low_scheduler = self.get_scheduler(entropy_bound=1e-2)
        high_scheduler = self.get_scheduler(entropy_bound=1e-1)
        low = low_scheduler.step(logits, low_scheduler.timesteps[0], sample)
        high = high_scheduler.step(logits, high_scheduler.timesteps[0], sample)
        self.assertGreaterEqual(high.committed_mask.sum().item(), low.committed_mask.sum().item())

    def test_non_accepted_are_renoised(self):
        scheduler = self.get_scheduler(entropy_bound=1e-3)
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, :5, 0] = 1e6
        out = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)
        # the 5 accepted positions hold token 0, the rest are random (not token 0 almost surely)
        self.assertTrue((out.prev_sample[0, :5] == 0).all())

    def test_step_output_shapes(self):
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 100, (3, 16))
        logits = torch.randn(3, 16, 100)
        out = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.committed_mask.shape, sample.shape)
        self.assertEqual(out.pred_original_sample.shape, sample.shape)
        self.assertEqual(out.sampled_probs.shape, sample.shape)
        self.assertIsNone(out.edited_mask)

    def test_return_tuple_has_fixed_arity(self):
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 100, (1, 16))
        logits = torch.randn(1, 16, 100)
        out = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 6)
        self.assertIsNone(out[-1])

    def test_sampled_probs_measured_on_unshaped_distribution(self):
        # `DiscreteSchedulerOutput.sampled_probs` is the confidence under the raw denoiser, not under this
        # scheduler's annealed sampling distribution, so a threshold on it does not move with `t_min`/`t_max`.
        scheduler = self.get_scheduler(t_max=0.5, t_min=0.5)
        sample = torch.randint(0, 64, (2, 16))
        logits = torch.randn(2, 16, 64) * 3.0
        out = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)

        raw_probs = torch.softmax(logits.float(), dim=-1)
        expected = torch.gather(raw_probs, -1, out.pred_original_sample.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.equal(out.sampled_probs, expected))

        annealed_probs = torch.softmax(logits.float() / 0.5, dim=-1)
        annealed = torch.gather(annealed_probs, -1, out.pred_original_sample.unsqueeze(-1)).squeeze(-1)
        self.assertFalse(torch.allclose(out.sampled_probs, annealed))

    def test_temperature_annealing_uses_step_index(self):
        # t_max on the first step, t_min on the last: a flat anneal must leave the schedule temperature-invariant.
        n = 4
        scheduler = self.get_scheduler(num_inference_steps=n, t_max=1.0, t_min=1.0)
        scheduler.set_timesteps(n)
        sample = torch.randint(0, 32, (1, 8))
        logits = torch.randn(1, 8, 32)
        for index, timestep in enumerate(scheduler.timesteps):
            out = scheduler.step(logits, timestep=timestep, sample=sample)
            self.assertEqual(scheduler.step_index, index + 1)
            sample = out.prev_sample
