import math
import tempfile
import unittest

import torch

from diffusers import BD3LMTokenDiffusionScheduler


class BD3LMTokenDiffusionSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {
            "block_size": 8,
            "num_inference_steps": 8,
            "noise_type": "loglinear",
            "nucleus_p": 1.0,
            "mask_token_id": 31,
        }
        config.update(kwargs)
        return BD3LMTokenDiffusionScheduler(**config)

    # ------------------------------------------------------------------
    # Timestep management
    # ------------------------------------------------------------------

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        self.assertEqual(scheduler.num_inference_steps, 8)
        self.assertEqual(len(scheduler.timesteps), 8)
        # Should go from 1.0 down to near 0.0
        self.assertAlmostEqual(scheduler.timesteps[0].item(), 1.0, places=4)
        self.assertAlmostEqual(scheduler.timesteps[-1].item(), 0.0, places=4)

    def test_set_timesteps_invalid(self):
        scheduler = self.get_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(0)

    # ------------------------------------------------------------------
    # Noise schedule: _compute_move_chance
    # ------------------------------------------------------------------

    def test_compute_move_chance_loglinear(self):
        scheduler = self.get_scheduler(noise_type="loglinear")
        t = torch.tensor([0.0, 0.5, 1.0])
        mc = scheduler._compute_move_chance(t)
        self.assertTrue(torch.allclose(mc, t))

    def test_compute_move_chance_cosine(self):
        scheduler = self.get_scheduler(noise_type="cosine")
        t = torch.tensor([0.0, 0.5, 1.0])
        mc = scheduler._compute_move_chance(t)
        eps = 1e-3
        expected_0 = 1.0 - (1.0 - eps) * math.cos(0.0)
        expected_half = 1.0 - (1.0 - eps) * math.cos(0.5 * math.pi / 2.0)
        expected_1 = 1.0 - (1.0 - eps) * math.cos(math.pi / 2.0)
        self.assertAlmostEqual(mc[0].item(), expected_0, places=5)
        self.assertAlmostEqual(mc[1].item(), expected_half, places=5)
        self.assertAlmostEqual(mc[2].item(), expected_1, places=5)

    def test_compute_move_chance_square(self):
        scheduler = self.get_scheduler(noise_type="square")
        t = torch.tensor([0.0, 0.5, 1.0])
        mc = scheduler._compute_move_chance(t)
        eps = 1e-3
        self.assertAlmostEqual(mc[0].item(), eps, places=5)
        self.assertAlmostEqual(mc[1].item(), 0.25, places=5)
        self.assertAlmostEqual(mc[2].item(), 1.0, places=5)

    def test_compute_move_chance_square_root(self):
        scheduler = self.get_scheduler(noise_type="square_root")
        t = torch.tensor([0.0, 0.25, 1.0])
        mc = scheduler._compute_move_chance(t)
        eps = 1e-3
        self.assertAlmostEqual(mc[0].item(), eps, places=5)
        self.assertAlmostEqual(mc[1].item(), 0.5, places=5)
        self.assertAlmostEqual(mc[2].item(), 1.0, places=5)

    def test_compute_move_chance_log(self):
        scheduler = self.get_scheduler(noise_type="log")
        t = torch.tensor([0.0, 1.0])
        mc = scheduler._compute_move_chance(t)
        self.assertAlmostEqual(mc[0].item(), 0.0, places=5)
        self.assertAlmostEqual(mc[1].item(), 1.0, places=5)

    # ------------------------------------------------------------------
    # Sigma computation
    # ------------------------------------------------------------------

    def test_compute_sigma(self):
        scheduler = self.get_scheduler(noise_type="loglinear")
        sigma = scheduler.compute_sigma(0.5, batch_size=2)
        self.assertEqual(sigma.shape, (2,))
        # sigma = -log(1 - move_chance) = -log(1 - 0.5) = log(2)
        expected = math.log(2.0)
        self.assertAlmostEqual(sigma[0].item(), expected, places=4)
        self.assertAlmostEqual(sigma[1].item(), expected, places=4)

    def test_compute_sigma_clamps_at_max(self):
        scheduler = self.get_scheduler(noise_type="loglinear")
        # At t=1.0, move_chance=1.0, so -log(0) -> inf, should be clamped.
        sigma = scheduler.compute_sigma(1.0, batch_size=1)
        eps = 1e-3
        sigma_max = -math.log(eps)
        self.assertAlmostEqual(sigma[0].item(), sigma_max, places=3)

    # ------------------------------------------------------------------
    # Config save/load
    # ------------------------------------------------------------------

    def test_save_load_config_round_trip(self):
        scheduler = self.get_scheduler(block_size=16, noise_type="cosine", nucleus_p=0.9, mask_token_id=99)
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded = BD3LMTokenDiffusionScheduler.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.block_size, 16)
        self.assertEqual(loaded.config.noise_type, "cosine")
        self.assertAlmostEqual(loaded.config.nucleus_p, 0.9)
        self.assertEqual(loaded.config.mask_token_id, 99)

    def test_from_config(self):
        scheduler = self.get_scheduler(block_size=16, noise_type="square")
        new_scheduler = BD3LMTokenDiffusionScheduler.from_config(scheduler.config)
        self.assertEqual(new_scheduler.config.block_size, 16)
        self.assertEqual(new_scheduler.config.noise_type, "square")

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def test_step_commits_tokens(self):
        """Running enough steps should commit masked tokens to non-mask values."""
        scheduler = self.get_scheduler(block_size=4)
        scheduler.set_timesteps(8)

        batch_size, block_size, vocab_size = 1, 4, 32
        mask_id = 31

        sample = torch.full((batch_size, block_size), mask_id, dtype=torch.long)

        # Create logits with strong preference for non-mask tokens.
        logits = torch.zeros(batch_size, block_size, vocab_size)
        for i in range(block_size):
            logits[0, i, i] = 10.0

        # Run all denoising steps.
        for step_idx in range(scheduler.num_inference_steps):
            t = scheduler.timesteps[step_idx]
            out = scheduler.step(
                model_output=logits,
                timestep=t,
                sample=sample,
                mask_token_id=mask_id,
                return_dict=True,
            )
            sample = out.prev_sample

        # After all steps, no mask tokens should remain.
        self.assertFalse((sample == mask_id).any().item())

    def test_step_preserves_unmasked_tokens(self):
        """Already-unmasked positions must be preserved (copy flag)."""
        scheduler = self.get_scheduler(block_size=4)
        scheduler.set_timesteps(4)

        batch_size, block_size, vocab_size = 1, 4, 32
        mask_id = 31

        # Positions 0,1 are already unmasked; positions 2,3 are masked.
        sample = torch.tensor([[5, 10, mask_id, mask_id]], dtype=torch.long)
        logits = torch.zeros(batch_size, block_size, vocab_size)
        for i in range(block_size):
            logits[0, i, i % (vocab_size - 2)] = 10.0

        out = scheduler.step(
            model_output=logits,
            timestep=scheduler.timesteps[0],
            sample=sample,
            mask_token_id=mask_id,
            return_dict=True,
        )

        # Unmasked positions must be unchanged.
        self.assertEqual(out.prev_sample[0, 0].item(), 5)
        self.assertEqual(out.prev_sample[0, 1].item(), 10)

    def test_step_return_tuple(self):
        """return_dict=False should return a plain tuple."""
        scheduler = self.get_scheduler(block_size=4)
        scheduler.set_timesteps(4)

        vocab_size = 32
        sample = torch.full((1, 4), 31, dtype=torch.long)
        logits = torch.randn(1, 4, vocab_size)

        result = scheduler.step(
            model_output=logits,
            timestep=scheduler.timesteps[0],
            sample=sample,
            mask_token_id=31,
            return_dict=False,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_step_batched(self):
        """step works with batch_size > 1."""
        scheduler = self.get_scheduler(block_size=4)
        scheduler.set_timesteps(4)

        batch_size, vocab_size = 3, 32
        mask_id = 31
        sample = torch.full((batch_size, 4), mask_id, dtype=torch.long)
        logits = torch.randn(batch_size, 4, vocab_size)

        out = scheduler.step(
            model_output=logits,
            timestep=scheduler.timesteps[0],
            sample=sample,
            mask_token_id=mask_id,
            return_dict=True,
        )

        self.assertEqual(out.prev_sample.shape, (batch_size, 4))

    # ------------------------------------------------------------------
    # Nucleus filtering
    # ------------------------------------------------------------------

    def test_nucleus_filtering_passthrough(self):
        """nucleus_p=1.0 should not alter the distribution."""
        probs = torch.softmax(torch.randn(1, 4, 32), dim=-1)
        filtered = BD3LMTokenDiffusionScheduler._nucleus_filtering(probs, nucleus_p=1.0)
        self.assertTrue(torch.allclose(probs, filtered))

    def test_nucleus_filtering_truncates(self):
        """nucleus_p < 1.0 should zero out low-probability tokens."""
        probs = torch.zeros(1, 1, 4)
        probs[0, 0] = torch.tensor([0.5, 0.3, 0.15, 0.05])
        filtered = BD3LMTokenDiffusionScheduler._nucleus_filtering(probs, nucleus_p=0.8)
        # Token with prob 0.05 should be zeroed out.
        self.assertAlmostEqual(filtered[0, 0, 3].item(), 0.0, places=5)
        # Filtered probs should still sum to ~1.
        self.assertAlmostEqual(filtered.sum().item(), 1.0, places=4)

    def test_nucleus_filtering_keeps_top1(self):
        """Nucleus filtering always keeps at least the top-1 token."""
        probs = torch.zeros(1, 1, 4)
        probs[0, 0] = torch.tensor([0.1, 0.1, 0.1, 0.7])
        filtered = BD3LMTokenDiffusionScheduler._nucleus_filtering(probs, nucleus_p=0.01)
        # Top-1 (index 3) must be kept.
        self.assertGreater(filtered[0, 0, 3].item(), 0.0)

    # ------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------

    def test_check_should_stop_all_unmasked(self):
        mask_id = 31
        sequences = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        self.assertTrue(BD3LMTokenDiffusionScheduler.check_should_stop(sequences, mask_id))

    def test_check_should_stop_has_masks(self):
        mask_id = 31
        sequences = torch.tensor([[1, 31, 3, 4]], dtype=torch.long)
        self.assertFalse(BD3LMTokenDiffusionScheduler.check_should_stop(sequences, mask_id))

    def test_check_eos_finished(self):
        eos_id = 2
        prompt_length = 2
        sequences = torch.tensor([[10, 11, 5, eos_id, 7, 8]], dtype=torch.long)
        finished = torch.tensor([False])

        finished = BD3LMTokenDiffusionScheduler.check_eos_finished(sequences, prompt_length, eos_id, finished)
        self.assertTrue(finished[0].item())

    def test_check_eos_finished_no_eos(self):
        eos_id = 2
        prompt_length = 2
        sequences = torch.tensor([[10, 11, 5, 6, 7, 8]], dtype=torch.long)
        finished = torch.tensor([False])

        finished = BD3LMTokenDiffusionScheduler.check_eos_finished(sequences, prompt_length, eos_id, finished)
        self.assertFalse(finished[0].item())

    def test_check_eos_finished_already_finished(self):
        eos_id = 2
        sequences = torch.tensor([[10, 11, 5, 6]], dtype=torch.long)
        finished = torch.tensor([True])

        finished = BD3LMTokenDiffusionScheduler.check_eos_finished(sequences, 2, eos_id, finished)
        self.assertTrue(finished[0].item())

    # ------------------------------------------------------------------
    # add_noise
    # ------------------------------------------------------------------

    def test_add_noise(self):
        scheduler = self.get_scheduler()
        original = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        mask_id = 31

        gen = torch.Generator().manual_seed(42)
        # At t=1.0 (loglinear), all tokens should be masked.
        t_full = torch.tensor([1.0])
        noisy = scheduler.add_noise(original, t_full, mask_token_id=mask_id, generator=gen)
        self.assertTrue((noisy == mask_id).all().item())

    def test_add_noise_zero(self):
        scheduler = self.get_scheduler()
        original = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        mask_id = 31

        # At t=0.0 (loglinear), no tokens should be masked.
        t_zero = torch.tensor([0.0])
        noisy = scheduler.add_noise(original, t_zero, mask_token_id=mask_id)
        self.assertTrue(torch.equal(noisy, original))

    def test_add_noise_partial(self):
        scheduler = self.get_scheduler()
        original = torch.arange(100).unsqueeze(0).long()
        mask_id = 999

        gen = torch.Generator().manual_seed(0)
        t_half = torch.tensor([0.5])
        noisy = scheduler.add_noise(original, t_half, mask_token_id=mask_id, generator=gen)

        num_masked = (noisy == mask_id).sum().item()
        # With 100 tokens and move_chance=0.5, we expect roughly 50 masked.
        self.assertGreater(num_masked, 20)
        self.assertLess(num_masked, 80)


if __name__ == "__main__":
    unittest.main()
