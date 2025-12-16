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


if __name__ == "__main__":
    unittest.main()
