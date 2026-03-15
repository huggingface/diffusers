import unittest

import torch

from diffusers import TokenDiffusionScheduler


class TokenDiffusionSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {
            "vocab_size": 128,
            "mask_token_id": 127,
            "num_train_timesteps": 100,
            "eps": 1e-3,
        }
        config.update(kwargs)
        return TokenDiffusionScheduler(**config)

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(10)
        self.assertEqual(len(scheduler.timesteps), 10)
        self.assertTrue((scheduler.timesteps[:-1] >= scheduler.timesteps[1:]).all().item())

    def test_alpha_schedule_monotone_and_bounded(self):
        # alpha(t) should be in (0, 1] and non-increasing in t for supported schedules.
        schedules = ["log_linear", "linear", "cosine", "geometric"]
        t = torch.linspace(0, 1, 33, dtype=torch.float32)

        for name in schedules:
            scheduler = self.get_scheduler(alpha_schedule=name)
            alpha = scheduler._alpha_t(t)
            self.assertTrue(((alpha > 0) & (alpha <= 1)).all().item())
            # monotone non-increasing: alpha[i] >= alpha[i+1]
            self.assertTrue((alpha[:-1] >= alpha[1:]).all().item())

    def test_mdlm_weights_match_log_linear_1_over_t(self):
        scheduler = self.get_scheduler(alpha_schedule="log_linear", eps=1e-3, num_train_timesteps=1000)
        timesteps = torch.tensor([1, 10, 100, 999], dtype=torch.long)
        w = scheduler.get_mdlm_loss_weights(timesteps).squeeze(-1)
        t_cont = timesteps.to(dtype=torch.float32) / float(scheduler.num_train_timesteps - 1)
        expected = 1.0 / t_cont
        self.assertTrue(torch.allclose(w, expected, rtol=5e-5, atol=1e-5))

    def test_add_noise_absorbing_keeps_shape_dtype(self):
        scheduler = self.get_scheduler()
        batch_size, seq_len = 4, 16
        x0 = torch.randint(0, scheduler.vocab_size, (batch_size, seq_len), dtype=torch.long)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), dtype=torch.long)

        xt = scheduler.add_noise(x0, noise=None, timesteps=timesteps)
        self.assertEqual(xt.shape, x0.shape)
        self.assertEqual(xt.dtype, torch.long)

        # xt values must be valid token ids
        self.assertTrue(((xt >= 0) & (xt < scheduler.vocab_size)).all().item())

    def test_step_preserves_unmasked_tokens(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(5)

        batch_size, seq_len = 2, 12
        x_t = torch.randint(0, scheduler.vocab_size, (batch_size, seq_len), dtype=torch.long)
        x_t[:, :3] = scheduler.mask_token_id  # ensure some masked positions

        # Model predicts uniform logits; step should never change already unmasked positions
        logits = torch.zeros((batch_size, seq_len, scheduler.vocab_size), dtype=torch.float32)
        out = scheduler.step(logits, scheduler.timesteps[0], x_t, return_dict=True)
        x_prev = out.prev_sample

        self.assertTrue((x_prev[:, 3:] == x_t[:, 3:]).all().item())

    def test_step_never_samples_mask_token(self):
        scheduler = self.get_scheduler()
        # Use a single inference step so the scheduler denoises to t=0 in one go (p_denoise = 1).
        scheduler.set_timesteps(1)

        batch_size, seq_len = 2, 12
        x_t = torch.full((batch_size, seq_len), scheduler.mask_token_id, dtype=torch.long)
        logits = torch.zeros((batch_size, seq_len, scheduler.vocab_size), dtype=torch.float32)

        gen = torch.Generator().manual_seed(0)
        x_prev = scheduler.step(logits, scheduler.timesteps[0], x_t, generator=gen, return_dict=True).prev_sample

        # Mask token is forbidden as an x0 prediction, and the scheduler performs a final noise-removal step.
        self.assertTrue((x_prev != scheduler.mask_token_id).all().item())

    def test_uniform_add_noise_excludes_mask_if_configured(self):
        scheduler = self.get_scheduler(forward_process="uniform", exclude_mask_from_uniform=True)
        batch_size, seq_len = 8, 64
        x0 = torch.randint(0, scheduler.vocab_size, (batch_size, seq_len), dtype=torch.long)
        # Make sure some originals are mask token too (uniform should still sample non-mask replacements).
        x0[:, :5] = scheduler.mask_token_id

        # Use the noisiest time (highest replace probability).
        timesteps = torch.full((batch_size,), scheduler.num_train_timesteps - 1, dtype=torch.long)
        xt = scheduler.add_noise(x0, noise=None, timesteps=timesteps)

        # Mask token should be rare-to-absent under uniform corruption when excluded.
        self.assertFalse((xt == scheduler.mask_token_id).any().item())

    def test_uniform_step_runs_and_returns_valid_ids(self):
        scheduler = self.get_scheduler(forward_process="uniform", exclude_mask_from_uniform=True)
        scheduler.set_timesteps(2)

        batch_size, seq_len = 2, 16
        x_t = torch.randint(0, scheduler.vocab_size, (batch_size, seq_len), dtype=torch.long)
        logits = torch.zeros((batch_size, seq_len, scheduler.vocab_size), dtype=torch.float32)

        gen = torch.Generator().manual_seed(0)
        x_prev = scheduler.step(logits, scheduler.timesteps[0], x_t, generator=gen, return_dict=True).prev_sample

        self.assertEqual(x_prev.shape, x_t.shape)
        self.assertTrue(((x_prev >= 0) & (x_prev < scheduler.vocab_size)).all().item())
        # With exclusion, mask token should not appear.
        self.assertFalse((x_prev == scheduler.mask_token_id).any().item())

    def test_alpha_helpers_shapes(self):
        scheduler = self.get_scheduler(num_train_timesteps=10)
        timesteps = torch.tensor([0, 1, 9], dtype=torch.long)

        alpha = scheduler.get_alpha(timesteps)
        dalpha = scheduler.get_alpha_prime(timesteps)

        self.assertEqual(alpha.shape, (3, 1))
        self.assertEqual(dalpha.shape, (3, 1))

    def test_set_timesteps_precomputes_alphas(self):
        for schedule in ["log_linear", "linear", "cosine", "geometric"]:
            scheduler = self.get_scheduler(alpha_schedule=schedule)
            scheduler.set_timesteps(10)

            self.assertIsNotNone(scheduler.alphas)
            self.assertEqual(len(scheduler.alphas), 10)

            # Verify pre-computed alphas match on-the-fly computation.
            for i, ts in enumerate(scheduler.timesteps):
                t = scheduler._t_from_timestep(int(ts.item()), device=torch.device("cpu"))
                expected = scheduler._alpha_t(t).to(dtype=torch.float32)
                self.assertTrue(
                    torch.allclose(scheduler.alphas[i].cpu(), expected, atol=1e-6),
                    f"Alpha mismatch at step {i} for schedule {schedule}",
                )

    def test_set_timesteps_builds_step_index_map(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(5)

        self.assertIsNotNone(scheduler._step_index_map)
        self.assertEqual(len(scheduler._step_index_map), 5)

        for i, ts in enumerate(scheduler.timesteps):
            self.assertEqual(scheduler._step_index_map[int(ts.item())], i)

    def test_step_respects_block_mask(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(1)

        batch_size, seq_len = 2, 8
        x = torch.full((batch_size, seq_len), scheduler.mask_token_id, dtype=torch.long)
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, :4] = True

        logits = torch.zeros((batch_size, seq_len, scheduler.config.vocab_size), dtype=torch.float32)
        gen = torch.Generator().manual_seed(0)
        out = scheduler.step(logits, scheduler.timesteps[0], x, generator=gen, return_dict=True, block_mask=block_mask)

        # Block positions should be denoised (non-mask) after the final noise-removal step.
        self.assertTrue((out.prev_sample[:, :4] != scheduler.mask_token_id).all().item())
        # Outside the block, tokens should remain unchanged (still mask).
        self.assertTrue((out.prev_sample[:, 4:] == scheduler.mask_token_id).all().item())

    def test_add_noise_respects_block_mask(self):
        scheduler = self.get_scheduler()
        batch_size, seq_len = 2, 16
        x0 = torch.randint(0, scheduler.config.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
        block_mask = torch.zeros_like(x0, dtype=torch.bool)
        block_mask[:, :4] = True

        # Use high noise timestep so almost all block positions get noised.
        timesteps = torch.full((batch_size,), scheduler.num_train_timesteps - 1, dtype=torch.long)
        xt = scheduler.add_noise(x0, noise=None, timesteps=timesteps, block_mask=block_mask)

        # Outside the block, tokens must be unchanged.
        self.assertTrue(torch.equal(xt[:, 4:], x0[:, 4:]))

    def test_step_without_block_mask_unchanged(self):
        """Passing block_mask=None should produce the same result as before."""
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(3)

        batch_size, seq_len = 2, 8
        x_t = torch.full((batch_size, seq_len), scheduler.mask_token_id, dtype=torch.long)
        logits = torch.randn((batch_size, seq_len, scheduler.config.vocab_size), dtype=torch.float32)

        gen1 = torch.Generator().manual_seed(42)
        out1 = scheduler.step(logits, scheduler.timesteps[0], x_t, generator=gen1, return_dict=True)

        gen2 = torch.Generator().manual_seed(42)
        out2 = scheduler.step(logits, scheduler.timesteps[0], x_t, generator=gen2, return_dict=True, block_mask=None)

        self.assertTrue(torch.equal(out1.prev_sample, out2.prev_sample))

    def test_step_uses_precomputed_alphas_consistent_with_recompute(self):
        """Verify step() produces identical results whether using pre-computed or recomputed alphas."""
        for process in ["absorbing", "uniform"]:
            scheduler = self.get_scheduler(forward_process=process)
            scheduler.set_timesteps(3)

            batch_size, seq_len = 2, 8
            if process == "absorbing":
                x_t = torch.full((batch_size, seq_len), scheduler.mask_token_id, dtype=torch.long)
            else:
                x_t = torch.randint(0, scheduler.vocab_size, (batch_size, seq_len), dtype=torch.long)
            logits = torch.randn((batch_size, seq_len, scheduler.vocab_size), dtype=torch.float32)

            gen1 = torch.Generator().manual_seed(42)
            out1 = scheduler.step(logits, scheduler.timesteps[0], x_t, generator=gen1, return_dict=True)

            # Temporarily clear pre-computed alphas to force recomputation.
            saved_alphas = scheduler.alphas
            saved_map = scheduler._step_index_map
            scheduler.alphas = None
            scheduler._step_index_map = None

            gen2 = torch.Generator().manual_seed(42)
            out2 = scheduler.step(logits, scheduler.timesteps[0], x_t, generator=gen2, return_dict=True)

            scheduler.alphas = saved_alphas
            scheduler._step_index_map = saved_map

            self.assertTrue(
                torch.equal(out1.prev_sample, out2.prev_sample),
                f"Mismatch for forward_process={process}",
            )


if __name__ == "__main__":
    unittest.main()
