import unittest

import torch

from diffusers import UniformRefinementScheduler


class UniformRefinementSchedulerTest(unittest.TestCase):
    """The uniform corruption process (no mask token), as used by DiffusionGemma's block refinement."""

    def get_scheduler(self, num_inference_steps=48, begin_index=None, **kwargs):
        config = {"num_inference_steps": num_inference_steps, "threshold": 1.0, "editing_threshold": None}
        config.update(kwargs)
        scheduler = UniformRefinementScheduler(**config)
        scheduler.set_timesteps(num_inference_steps)
        if begin_index is not None:
            scheduler.set_begin_index(begin_index)
        return scheduler

    def test_set_timesteps(self):
        scheduler = self.get_scheduler(num_inference_steps=16)
        self.assertEqual(len(scheduler.timesteps), 16)
        # Same decreasing corruption-level grid as the other discrete schedulers.
        self.assertEqual(scheduler.timesteps.dtype, torch.float32)
        self.assertEqual(scheduler.timesteps[0].item(), 1.0)
        self.assertEqual(scheduler.timesteps[-1].item(), 1.0 / 16)
        self.assertTrue(bool((scheduler.timesteps[1:] < scheduler.timesteps[:-1]).all()))

    def test_cumulative_quota_progression(self):
        # threshold=1.0 disables threshold commits, so only the even per-step quota applies: ceil(256/48)=6, then 11.
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        out0 = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)
        self.assertEqual(scheduler._committed.sum().item(), 6)
        self.assertEqual(out0.committed_mask.sum().item(), 6)
        out1 = scheduler.step(logits, timestep=scheduler.timesteps[1], sample=out0.prev_sample)
        self.assertEqual(scheduler._committed.sum().item(), 11)
        self.assertEqual(out1.committed_mask.sum().item(), 5)

    def test_last_step_commits_all(self):
        # `step` reads `timestep` only on the first call, so jumping to the final step needs `set_begin_index`.
        scheduler = self.get_scheduler(begin_index=47)
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        scheduler.step(logits, timestep=scheduler.timesteps[47], sample=sample)
        self.assertTrue(scheduler._committed.all())

    def test_threshold_commits_beyond_quota(self):
        scheduler = self.get_scheduler(threshold=0.5)
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, torch.arange(20), 0] = 1e6  # 20 high-confidence positions (token 0)
        scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)
        # 20 positions exceed the threshold and get committed regardless of the quota
        self.assertEqual(scheduler._committed.sum().item(), 20)

    def test_editing_replaces_committed_token(self):
        scheduler = self.get_scheduler(threshold=1.0, editing_threshold=0.5, begin_index=24)
        sample = torch.zeros(1, 256, dtype=torch.long)
        scheduler._committed = torch.ones_like(sample, dtype=torch.bool)  # pretend all committed
        logits = torch.zeros(1, 256, 10000)
        logits[0, 0, 1] = 1e6  # confidently predicts token 1 at position 0 (differs from current token 0)
        out = scheduler.step(logits, timestep=scheduler.timesteps[24], sample=sample)
        self.assertEqual(out.prev_sample[0, 0].item(), 1)
        self.assertTrue((out.prev_sample[0, 1:] == 0).all())
        self.assertEqual(out.edited_mask.sum().item(), 1)
        self.assertEqual(out.committed_mask.sum().item(), 0)

    def test_set_timesteps_clears_committed_state(self):
        # The committed set is per-block state that cannot be read off the sequence, so starting a new block
        # requires `set_timesteps`. See §5.3 of the refactor design.
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        for index in range(3):
            sample = scheduler.step(logits, timestep=scheduler.timesteps[index], sample=sample).prev_sample
        self.assertEqual(scheduler._committed.sum().item(), 16)

        scheduler.set_timesteps(48)
        self.assertIsNone(scheduler._committed)
        self.assertIsNone(scheduler.step_index)
        scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)
        self.assertEqual(scheduler._committed.sum().item(), 6)

    def test_stepping_past_the_schedule_raises(self):
        # Without this the committed quota would silently over-commit on a second block.
        scheduler = self.get_scheduler(num_inference_steps=2)
        sample = torch.randint(0, 100, (1, 8))
        logits = torch.zeros(1, 8, 100)
        for index in range(2):
            sample = scheduler.step(logits, timestep=scheduler.timesteps[index], sample=sample).prev_sample
        with self.assertRaisesRegex(ValueError, "set_timesteps"):
            scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)

    def test_sampled_probs_measured_on_unshaped_distribution(self):
        # `confidence` drives both the commit quota and the `threshold` comparison here, so `sampled_probs` must be
        # the raw denoiser confidence: otherwise `threshold=0.95` would mean something different at every
        # temperature. Pinned directly rather than via a trajectory, since a temperature change need not move the
        # committed set (the quota only depends on the *ordering* of confidences).
        scheduler = self.get_scheduler(temperature=0.7, threshold=0.95)
        sample = torch.randint(0, 64, (2, 16))
        logits = torch.randn(2, 16, 64) * 3.0
        out = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample)

        raw_probs = torch.softmax(logits.float(), dim=-1)
        expected = torch.gather(raw_probs, -1, out.pred_original_sample.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.equal(out.sampled_probs, expected))

        shaped_probs = torch.softmax(logits.float() / 0.7, dim=-1)
        shaped = torch.gather(shaped_probs, -1, out.pred_original_sample.unsqueeze(-1)).squeeze(-1)
        self.assertFalse(torch.allclose(out.sampled_probs, shaped))

    def test_invalid_threshold_raises(self):
        # The knob moved here from `LLaDA2Pipeline.__call__`; its validation moved with it.
        with self.assertRaises(ValueError):
            self.get_scheduler(threshold=-0.5)

    def test_non_positive_editing_threshold_disables_editing(self):
        # `0.0` and negatives disable editing, exactly as in `BlockRefinementScheduler`.
        for editing_threshold in (0.0, -1.0):
            scheduler = self.get_scheduler(threshold=1.0, editing_threshold=editing_threshold, begin_index=24)
            sample = torch.arange(8).view(1, 8)
            logits = torch.zeros(1, 8, 32)
            logits[0, :, 5] = 20.0
            out = scheduler.step(logits, scheduler.timesteps[24], sample)
            self.assertFalse(bool(out.edited_mask.any()), editing_threshold)

    def test_changing_the_block_width_mid_schedule_raises(self):
        # Silently re-initializing the committed state would hide a caller error.
        scheduler = self.get_scheduler()
        logits = torch.zeros(1, 8, 32)
        scheduler.step(logits, scheduler.timesteps[0], torch.arange(8).view(1, 8))
        with self.assertRaisesRegex(ValueError, "changed shape mid-schedule"):
            scheduler.step(torch.zeros(1, 4, 32), scheduler.timesteps[1], torch.arange(4).view(1, 4))

    def test_return_tuple_has_fixed_arity(self):
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 100, (1, 16))
        logits = torch.randn(1, 16, 100)
        out = scheduler.step(logits, timestep=scheduler.timesteps[0], sample=sample, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 6)
        self.assertIsNotNone(out[-1])


if __name__ == "__main__":
    unittest.main()
