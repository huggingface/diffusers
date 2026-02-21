import unittest

import torch

from diffusers import BlockTokenDiffusionScheduler


class BlockTokenDiffusionSchedulerTest(unittest.TestCase):
    def test_step_respects_block_mask(self):
        vocab_size = 32
        scheduler = BlockTokenDiffusionScheduler(vocab_size=vocab_size, mask_token_id=vocab_size - 1)
        scheduler.set_timesteps(1)

        batch_size, seq_len = 2, 8
        x = torch.full((batch_size, seq_len), scheduler.mask_token_id, dtype=torch.long)
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, :4] = True

        logits = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32)
        gen = torch.Generator().manual_seed(0)
        out = scheduler.step(logits, scheduler.timesteps[0], x, generator=gen, return_dict=True, block_mask=block_mask)

        # Block positions should be denoised (non-mask) after the final noise-removal step.
        self.assertTrue((out.prev_sample[:, :4] != scheduler.mask_token_id).all().item())
        # Outside the block, tokens should remain unchanged (still mask).
        self.assertTrue((out.prev_sample[:, 4:] == scheduler.mask_token_id).all().item())


if __name__ == "__main__":
    unittest.main()
