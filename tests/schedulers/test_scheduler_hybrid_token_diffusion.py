import unittest

import torch

from diffusers import HybridTokenDiffusionScheduler


class HybridTokenDiffusionSchedulerTest(unittest.TestCase):
    def test_add_noise_and_step_shapes(self):
        vocab_size = 32
        scheduler = HybridTokenDiffusionScheduler(vocab_size=vocab_size, mask_token_id=vocab_size - 1)
        scheduler.set_timesteps(4, device="cpu")

        batch_size, seq_len = 2, 8
        x0 = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), dtype=torch.long)
        x_t = scheduler.add_noise(x0, noise=None, timesteps=timesteps)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertEqual(x_t.dtype, torch.long)

        logits = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32)
        gen = torch.Generator().manual_seed(0)
        out = scheduler.step(logits, scheduler.timesteps[0], x_t, generator=gen, return_dict=True)
        self.assertEqual(out.prev_sample.shape, x0.shape)
        self.assertTrue(((out.prev_sample >= 0) & (out.prev_sample < vocab_size)).all().item())


if __name__ == "__main__":
    unittest.main()
