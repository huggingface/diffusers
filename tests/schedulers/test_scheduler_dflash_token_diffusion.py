import tempfile
import unittest

import torch

from diffusers import DFlashTokenDiffusionScheduler


class DFlashTokenDiffusionSchedulerTest(unittest.TestCase):
    def get_scheduler(self):
        return DFlashTokenDiffusionScheduler()

    # ------------------------------------------------------------------
    # set_timesteps
    # ------------------------------------------------------------------
    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(4)
        self.assertEqual(scheduler.num_inference_steps, 4)
        self.assertEqual(len(scheduler.timesteps), 4)
        self.assertEqual(scheduler.timesteps[0].item(), 3)
        self.assertEqual(scheduler.timesteps[-1].item(), 0)

    def test_set_timesteps_single(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(1)
        self.assertEqual(scheduler.num_inference_steps, 1)
        self.assertEqual(len(scheduler.timesteps), 1)
        self.assertEqual(scheduler.timesteps[0].item(), 0)

    def test_set_timesteps_invalid(self):
        scheduler = self.get_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(0)
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(-1)

    # ------------------------------------------------------------------
    # Config round-trip
    # ------------------------------------------------------------------
    def test_save_load_config_round_trip(self):
        scheduler = self.get_scheduler()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded = DFlashTokenDiffusionScheduler.from_pretrained(tmpdir)
        # The scheduler has no user-configurable params, but it should survive the round-trip.
        self.assertIsInstance(loaded, DFlashTokenDiffusionScheduler)
        self.assertEqual(loaded.order, 1)

    def test_from_config(self):
        scheduler = self.get_scheduler()
        new_scheduler = DFlashTokenDiffusionScheduler.from_config(scheduler.config)
        self.assertIsInstance(new_scheduler, DFlashTokenDiffusionScheduler)
        self.assertEqual(new_scheduler.order, 1)

    # ------------------------------------------------------------------
    # sample() – greedy
    # ------------------------------------------------------------------
    def test_sample_greedy(self):
        scheduler = self.get_scheduler()
        logits = torch.tensor([[[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]]])  # (1, 2, 3)
        tokens = scheduler.sample(logits, temperature=0.0)
        self.assertEqual(tokens.shape, (1, 2))
        self.assertEqual(tokens[0, 0].item(), 1)  # argmax of [1,5,2]
        self.assertEqual(tokens[0, 1].item(), 2)  # argmax of [3,1,4]

    def test_sample_greedy_batched(self):
        scheduler = self.get_scheduler()
        logits = torch.tensor(
            [
                [[10.0, 0.0], [0.0, 10.0]],
                [[0.0, 10.0], [10.0, 0.0]],
            ]
        )  # (2, 2, 2)
        tokens = scheduler.sample(logits, temperature=0.0)
        self.assertEqual(tokens.shape, (2, 2))
        self.assertEqual(tokens[0, 0].item(), 0)
        self.assertEqual(tokens[0, 1].item(), 1)
        self.assertEqual(tokens[1, 0].item(), 1)
        self.assertEqual(tokens[1, 1].item(), 0)

    # ------------------------------------------------------------------
    # sample() – multinomial
    # ------------------------------------------------------------------
    def test_sample_multinomial(self):
        scheduler = self.get_scheduler()
        # One token has overwhelming probability; multinomial should pick it.
        logits = torch.tensor([[[0.0, 100.0, -100.0]]])  # (1, 1, 3)
        tokens = scheduler.sample(logits, temperature=1.0)
        self.assertEqual(tokens.shape, (1, 1))
        self.assertEqual(tokens[0, 0].item(), 1)

    # ------------------------------------------------------------------
    # step() – return dict
    # ------------------------------------------------------------------
    def test_step_all_accepted(self):
        """All draft tokens match the posterior => accepted_length == block_size - 1."""
        scheduler = self.get_scheduler()
        batch_size, block_size, vocab_size = 1, 4, 8

        # Draft tokens: [0, 3, 3, 3]
        draft_tokens = torch.tensor([[0, 3, 3, 3]], dtype=torch.long)
        # Target logits: make argmax = [3, 3, 3, X] so posterior[:, :-1] matches draft[:, 1:]
        logits = torch.zeros(batch_size, block_size, vocab_size)
        logits[:, 0, 3] = 10.0
        logits[:, 1, 3] = 10.0
        logits[:, 2, 3] = 10.0
        logits[:, 3, 5] = 10.0  # last posterior token (next_token candidate)

        out = scheduler.step(logits, 0, draft_tokens, temperature=0.0, return_dict=True)

        self.assertEqual(out.prev_sample.shape, (1, 4))
        self.assertEqual(out.accepted_length.shape, (1,))
        self.assertEqual(out.accepted_length[0].item(), 3)  # all 3 comparisons match
        self.assertEqual(out.next_token.shape, (1,))
        self.assertEqual(out.next_token[0].item(), 5)
        self.assertEqual(out.posterior.shape, (1, 4))

    def test_step_none_accepted(self):
        """First draft token already mismatches => accepted_length == 0."""
        scheduler = self.get_scheduler()
        batch_size, block_size, vocab_size = 1, 4, 8

        draft_tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        logits = torch.zeros(batch_size, block_size, vocab_size)
        logits[:, 0, 5] = 10.0  # posterior[0] = 5, but draft[1] = 1 => mismatch
        logits[:, 1, 2] = 10.0
        logits[:, 2, 3] = 10.0
        logits[:, 3, 4] = 10.0

        out = scheduler.step(logits, 0, draft_tokens, temperature=0.0, return_dict=True)

        self.assertEqual(out.accepted_length[0].item(), 0)
        self.assertEqual(out.next_token[0].item(), 5)  # posterior at index 0

    def test_step_partial_accepted(self):
        """First two match, third does not => accepted_length == 2."""
        scheduler = self.get_scheduler()
        batch_size, block_size, vocab_size = 1, 5, 8

        # draft: [0, 3, 4, 7, 2]
        draft_tokens = torch.tensor([[0, 3, 4, 7, 2]], dtype=torch.long)
        logits = torch.zeros(batch_size, block_size, vocab_size)
        logits[:, 0, 3] = 10.0  # match draft[1]=3
        logits[:, 1, 4] = 10.0  # match draft[2]=4
        logits[:, 2, 0] = 10.0  # mismatch draft[3]=7
        logits[:, 3, 2] = 10.0
        logits[:, 4, 6] = 10.0

        out = scheduler.step(logits, 0, draft_tokens, temperature=0.0, return_dict=True)

        self.assertEqual(out.accepted_length[0].item(), 2)
        self.assertEqual(out.next_token[0].item(), 0)  # posterior at index 2

    def test_step_single_token_block(self):
        """Block with a single token => accepted_length == 0."""
        scheduler = self.get_scheduler()
        draft_tokens = torch.tensor([[5]], dtype=torch.long)
        logits = torch.zeros(1, 1, 8)
        logits[:, 0, 3] = 10.0

        out = scheduler.step(logits, 0, draft_tokens, temperature=0.0, return_dict=True)
        self.assertEqual(out.accepted_length[0].item(), 0)
        self.assertEqual(out.next_token[0].item(), 3)

    # ------------------------------------------------------------------
    # step() – return tuple
    # ------------------------------------------------------------------
    def test_step_return_tuple(self):
        scheduler = self.get_scheduler()
        draft_tokens = torch.tensor([[0, 1, 2]], dtype=torch.long)
        logits = torch.randn(1, 3, 8)

        result = scheduler.step(logits, 0, draft_tokens, temperature=0.0, return_dict=False)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        prev_sample, accepted_length, next_token, posterior = result
        self.assertEqual(prev_sample.shape, (1, 3))
        self.assertEqual(accepted_length.shape, (1,))
        self.assertEqual(next_token.shape, (1,))
        self.assertEqual(posterior.shape, (1, 3))

    # ------------------------------------------------------------------
    # step() – batched
    # ------------------------------------------------------------------
    def test_step_batched(self):
        scheduler = self.get_scheduler()
        batch_size, block_size, vocab_size = 3, 4, 16
        draft_tokens = torch.randint(0, vocab_size, (batch_size, block_size))
        logits = torch.randn(batch_size, block_size, vocab_size)

        out = scheduler.step(logits, 0, draft_tokens, temperature=0.0, return_dict=True)

        self.assertEqual(out.prev_sample.shape, (batch_size, block_size))
        self.assertEqual(out.accepted_length.shape, (batch_size,))
        self.assertEqual(out.next_token.shape, (batch_size,))
        self.assertEqual(out.posterior.shape, (batch_size, block_size))

    # ------------------------------------------------------------------
    # check_should_stop()
    # ------------------------------------------------------------------
    def test_check_should_stop_no_stop_tokens(self):
        output_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        self.assertFalse(DFlashTokenDiffusionScheduler.check_should_stop(output_ids, None, 2))

    def test_check_should_stop_found(self):
        # Stop token 99 is in the generated portion (after num_input_tokens=2).
        output_ids = torch.tensor([[1, 2, 3, 99, 5]], dtype=torch.long)
        self.assertTrue(DFlashTokenDiffusionScheduler.check_should_stop(output_ids, [99], 2))

    def test_check_should_stop_only_in_prompt(self):
        # Stop token 1 is only in the prompt portion => should NOT stop.
        output_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        self.assertFalse(DFlashTokenDiffusionScheduler.check_should_stop(output_ids, [1], 2))

    def test_check_should_stop_multiple_stop_tokens(self):
        output_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        self.assertTrue(DFlashTokenDiffusionScheduler.check_should_stop(output_ids, [40, 99], 2))
        self.assertFalse(DFlashTokenDiffusionScheduler.check_should_stop(output_ids, [99, 100], 2))

    # ------------------------------------------------------------------
    # add_noise()
    # ------------------------------------------------------------------
    def test_add_noise_prompt_preserved(self):
        scheduler = self.get_scheduler()
        original = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.long)
        attention_mask = torch.ones_like(original)
        mask_token_id = 99
        prompt_length = 3

        gen = torch.Generator().manual_seed(42)
        noisy, masked = scheduler.add_noise(
            original,
            attention_mask,
            prompt_length=prompt_length,
            block_size=4,
            mask_token_id=mask_token_id,
            generator=gen,
        )

        # Prompt positions should never be masked.
        self.assertFalse(masked[0, :prompt_length].any().item())
        # Prompt tokens should be unchanged.
        self.assertTrue(torch.equal(noisy[0, :prompt_length], original[0, :prompt_length]))

    def test_add_noise_masked_positions(self):
        scheduler = self.get_scheduler()
        original = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        attention_mask = torch.ones_like(original)
        mask_token_id = 99

        gen = torch.Generator().manual_seed(0)
        noisy, masked = scheduler.add_noise(
            original,
            attention_mask,
            prompt_length=2,
            block_size=3,
            mask_token_id=mask_token_id,
            generator=gen,
        )

        # Where masked is True, noisy should equal mask_token_id.
        self.assertTrue((noisy[masked] == mask_token_id).all().item())
        # Where masked is False, noisy should equal original.
        self.assertTrue(torch.equal(noisy[~masked], original[~masked]))

    def test_add_noise_respects_attention_mask(self):
        scheduler = self.get_scheduler()
        original = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
        mask_token_id = 99

        gen = torch.Generator().manual_seed(42)
        noisy, masked = scheduler.add_noise(
            original,
            attention_mask,
            prompt_length=1,
            block_size=3,
            mask_token_id=mask_token_id,
            generator=gen,
        )

        # Padding positions (attention_mask=0) should never be masked.
        self.assertFalse(masked[0, 4].item())
        self.assertFalse(masked[0, 5].item())

    def test_add_noise_output_shapes(self):
        scheduler = self.get_scheduler()
        batch_size, seq_len = 2, 10
        original = torch.randint(0, 50, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        mask_token_id = 99

        noisy, masked = scheduler.add_noise(
            original,
            attention_mask,
            prompt_length=2,
            block_size=4,
            mask_token_id=mask_token_id,
        )

        self.assertEqual(noisy.shape, (batch_size, seq_len))
        self.assertEqual(masked.shape, (batch_size, seq_len))
        self.assertEqual(noisy.dtype, torch.long)
        self.assertEqual(masked.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
