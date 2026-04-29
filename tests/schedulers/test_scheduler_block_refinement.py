import tempfile
import unittest

import torch

from diffusers import BlockRefinementScheduler


class BlockRefinementSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {
            "block_length": 32,
            "num_inference_steps": 8,
            "threshold": 0.95,
            "editing_threshold": None,
            "minimal_topk": 1,
        }
        config.update(kwargs)
        return BlockRefinementScheduler(**config)

    def _make_logits_from_probs(self, target_probs: torch.Tensor, vocab_size: int = 100) -> torch.Tensor:
        """Create logits where softmax of the target token has approximately the given probability."""
        batch_size, block_length = target_probs.shape
        logits = torch.zeros(batch_size, block_length, vocab_size)
        # Set token 0 as the "predicted" token with a logit proportional to desired probability
        for b in range(batch_size):
            for t in range(block_length):
                p = target_probs[b, t].item()
                if p > 0:
                    logits[b, t, t % (vocab_size - 1)] = 10.0 * p
        return logits

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        self.assertEqual(scheduler.num_inference_steps, 8)
        self.assertEqual(len(scheduler.timesteps), 8)
        self.assertEqual(scheduler.timesteps[0].item(), 7)
        self.assertEqual(scheduler.timesteps[-1].item(), 0)

    def test_set_timesteps_invalid(self):
        scheduler = self.get_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(0)

    def test_get_num_transfer_tokens_even(self):
        scheduler = self.get_scheduler()
        schedule = scheduler.get_num_transfer_tokens(block_length=32, num_inference_steps=8)
        self.assertEqual(schedule.sum().item(), 32)
        self.assertEqual(len(schedule), 8)
        self.assertTrue((schedule == 4).all().item())

    def test_get_num_transfer_tokens_remainder(self):
        scheduler = self.get_scheduler()
        schedule = scheduler.get_num_transfer_tokens(block_length=10, num_inference_steps=3)
        self.assertEqual(schedule.sum().item(), 10)
        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule[0].item(), 4)
        self.assertEqual(schedule[1].item(), 3)
        self.assertEqual(schedule[2].item(), 3)

    def test_transfer_schedule_created_on_set_timesteps(self):
        scheduler = self.get_scheduler(block_length=16)
        scheduler.set_timesteps(4)
        self.assertIsNotNone(scheduler._transfer_schedule)
        self.assertEqual(scheduler._transfer_schedule.sum().item(), 16)

    def test_save_load_config_round_trip(self):
        scheduler = self.get_scheduler(block_length=64, threshold=0.8, editing_threshold=0.5, minimal_topk=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded = BlockRefinementScheduler.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.block_length, 64)
        self.assertEqual(loaded.config.threshold, 0.8)
        self.assertEqual(loaded.config.editing_threshold, 0.5)
        self.assertEqual(loaded.config.minimal_topk, 2)

    def test_from_config(self):
        scheduler = self.get_scheduler(block_length=16, threshold=0.7)
        new_scheduler = BlockRefinementScheduler.from_config(scheduler.config)
        self.assertEqual(new_scheduler.config.block_length, 16)
        self.assertEqual(new_scheduler.config.threshold, 0.7)

    def test_step_commits_tokens(self):
        """Verify that step() commits mask tokens based on confidence."""
        scheduler = self.get_scheduler(block_length=8)
        scheduler.set_timesteps(2)

        batch_size, block_length, vocab_size = 1, 8, 32
        mask_id = 31

        sample = torch.full((batch_size, block_length), mask_id, dtype=torch.long)
        # Create logits where confidence decreases with position
        logits = torch.zeros(batch_size, block_length, vocab_size)
        for i in range(block_length):
            logits[0, i, i] = 10.0 - i  # decreasing confidence

        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            temperature=0.0,
            threshold=0.95,
            return_dict=True,
        )

        # With 8 tokens and 2 steps, first step should commit 4 tokens
        committed = out.transfer_index[0].sum().item()
        self.assertEqual(committed, 4)

    def test_step_no_editing_by_default(self):
        """Without editing_threshold, no non-mask tokens should be changed."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        vocab_size = 32
        sample = torch.tensor([[10, 20, 31, 31]], dtype=torch.long)
        logits = torch.zeros(1, 4, vocab_size)
        logits[0, :, 15] = 10.0  # predict token 15 for all positions

        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=31,
            temperature=0.0,
            editing_threshold=None,
            return_dict=True,
        )

        self.assertFalse(out.editing_transfer_index.any().item())
        self.assertFalse(out.transfer_index[0, 0].item())
        self.assertFalse(out.transfer_index[0, 1].item())

    def test_step_editing_replaces_tokens(self):
        """With editing_threshold, non-mask tokens with high confidence and different prediction get replaced."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        vocab_size = 32
        sample = torch.tensor([[10, 20, 31, 31]], dtype=torch.long)
        logits = torch.zeros(1, 4, vocab_size)
        # Token 0: predict 50 (different from 10) with very high logit
        logits[0, 0, 15] = 20.0
        # Token 1: predict 20 (same as current)
        logits[0, 1, 20] = 20.0
        # Mask tokens
        logits[0, 2, 5] = 5.0
        logits[0, 3, 6] = 5.0

        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=31,
            temperature=0.0,
            editing_threshold=0.5,
            return_dict=True,
        )

        # Token 0 should be edited (different prediction, high confidence)
        self.assertTrue(out.editing_transfer_index[0, 0].item())
        # Token 1 should NOT be edited (same prediction)
        self.assertFalse(out.editing_transfer_index[0, 1].item())

    def test_step_prompt_mask_prevents_editing(self):
        """Prompt positions should never be edited even with editing enabled."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        vocab_size = 32
        sample = torch.tensor([[10, 20, 31, 31]], dtype=torch.long)
        logits = torch.zeros(1, 4, vocab_size)
        logits[0, :, 15] = 20.0
        prompt_mask = torch.tensor([True, True, False, False])

        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=31,
            temperature=0.0,
            editing_threshold=0.5,
            prompt_mask=prompt_mask,
            return_dict=True,
        )

        self.assertFalse(out.editing_transfer_index[0, 0].item())
        self.assertFalse(out.editing_transfer_index[0, 1].item())

    def test_step_return_tuple(self):
        """Verify tuple output when return_dict=False."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        vocab_size = 32
        sample = torch.full((1, 4), 31, dtype=torch.long)
        logits = torch.randn(1, 4, vocab_size)

        result = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=31,
            temperature=0.0,
            return_dict=False,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

    def test_step_batched(self):
        """Verify step works with batch_size > 1."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        batch_size, vocab_size = 3, 32
        mask_id = 31
        sample = torch.full((batch_size, 4), mask_id, dtype=torch.long)
        logits = torch.randn(batch_size, 4, vocab_size)

        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            temperature=0.0,
            return_dict=True,
        )

        self.assertEqual(out.prev_sample.shape, (batch_size, 4))
        self.assertEqual(out.transfer_index.shape, (batch_size, 4))

    def test_check_block_should_continue_finished(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        finished = torch.tensor([True, True])
        result = scheduler.check_block_should_continue(
            step_idx=0,
            masks_remaining=True,
            editing_enabled=False,
            editing_transfer_index=torch.zeros(2, 32, dtype=torch.bool),
            post_steps=0,
            max_post_steps=16,
            finished=finished,
        )
        self.assertFalse(result)

    def test_check_block_should_continue_no_masks_no_edits(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        finished = torch.tensor([False])
        result = scheduler.check_block_should_continue(
            step_idx=5,
            masks_remaining=False,
            editing_enabled=True,
            editing_transfer_index=torch.zeros(1, 32, dtype=torch.bool),
            post_steps=1,
            max_post_steps=16,
            finished=finished,
        )
        self.assertFalse(result)

    def test_check_block_should_continue_steps_exhausted(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        finished = torch.tensor([False])
        result = scheduler.check_block_should_continue(
            step_idx=8,
            masks_remaining=True,
            editing_enabled=False,
            editing_transfer_index=torch.zeros(1, 32, dtype=torch.bool),
            post_steps=0,
            max_post_steps=16,
            finished=finished,
        )
        self.assertFalse(result)

    def test_check_eos_finished_marks_batch(self):
        """When EOS is committed and all tokens before it are unmasked, mark batch as finished."""
        mask_id, eos_id, prompt_length = 99, 2, 2
        # cur_x: [prompt, prompt, token, eos, mask, mask]
        cur_x = torch.tensor([[10, 11, 5, eos_id, mask_id, mask_id]], dtype=torch.long)
        sampled_tokens = torch.tensor([[0, 0, 0, eos_id]], dtype=torch.long)
        final_transfer = torch.tensor([[False, False, False, True]])
        finished = torch.tensor([False])

        finished = BlockRefinementScheduler.check_eos_finished(
            cur_x=cur_x,
            sampled_tokens=sampled_tokens,
            final_transfer=final_transfer,
            finished=finished,
            eos_token_id=eos_id,
            mask_token_id=mask_id,
            prompt_length=prompt_length,
        )
        self.assertTrue(finished[0].item())

    def test_check_eos_finished_ignores_when_masks_before_eos(self):
        """If there are still mask tokens between prompt and EOS, don't mark as finished."""
        mask_id, eos_id, prompt_length = 99, 2, 2
        # cur_x: [prompt, prompt, mask, eos] — mask before EOS
        cur_x = torch.tensor([[10, 11, mask_id, eos_id]], dtype=torch.long)
        sampled_tokens = torch.tensor([[0, 0]], dtype=torch.long)
        final_transfer = torch.tensor([[False, True]])
        finished = torch.tensor([False])

        finished = BlockRefinementScheduler.check_eos_finished(
            cur_x=cur_x,
            sampled_tokens=sampled_tokens,
            final_transfer=final_transfer,
            finished=finished,
            eos_token_id=eos_id,
            mask_token_id=mask_id,
            prompt_length=prompt_length,
        )
        self.assertFalse(finished[0].item())

    def test_check_eos_finished_already_finished(self):
        """Already-finished batches should stay finished."""
        mask_id, eos_id = 99, 2
        cur_x = torch.tensor([[10, 11, 5, 6]], dtype=torch.long)
        sampled_tokens = torch.tensor([[0, 0]], dtype=torch.long)
        final_transfer = torch.tensor([[False, False]])
        finished = torch.tensor([True])

        finished = BlockRefinementScheduler.check_eos_finished(
            cur_x=cur_x,
            sampled_tokens=sampled_tokens,
            final_transfer=final_transfer,
            finished=finished,
            eos_token_id=eos_id,
            mask_token_id=mask_id,
            prompt_length=2,
        )
        self.assertTrue(finished[0].item())

    def test_add_noise(self):
        scheduler = self.get_scheduler(block_length=4)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        mask_token_id = 99

        gen = torch.Generator().manual_seed(42)
        noisy, noisy_rev, masked, masked_rev = scheduler.add_noise(
            input_ids,
            attention_mask,
            prompt_length=2,
            block_length=4,
            mask_token_id=mask_token_id,
            generator=gen,
        )

        # Prompt positions should never be masked
        self.assertFalse(masked[0, 0].item())
        self.assertFalse(masked[0, 1].item())
        self.assertFalse(masked_rev[0, 0].item())
        self.assertFalse(masked_rev[0, 1].item())

        # Noisy should have mask_token_id where masked is True
        self.assertTrue((noisy[masked] == mask_token_id).all().item())
        self.assertTrue((noisy_rev[masked_rev] == mask_token_id).all().item())

        # masked and masked_rev should be complementary within valid non-prompt positions
        non_prompt = torch.zeros_like(masked)
        non_prompt[0, 2:] = True
        combined = masked | masked_rev
        self.assertTrue((combined[0, 2:] == non_prompt[0, 2:]).all().item())


class TestTopPFiltering(unittest.TestCase):
    def test_top_p_filtering(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        filtered = BlockRefinementScheduler._top_p_filtering(logits, top_p=0.5)
        self.assertTrue((filtered > torch.finfo(filtered.dtype).min).any())
        self.assertTrue((filtered == torch.finfo(filtered.dtype).min).any())

    def test_top_p_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_p_filtering(logits, top_p=None)
        self.assertTrue(torch.equal(result, logits))

    def test_top_p_filtering_one(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_p_filtering(logits, top_p=1.0)
        self.assertTrue(torch.equal(result, logits))


class TestTopKFiltering(unittest.TestCase):
    def test_top_k_filtering(self):
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        filtered = BlockRefinementScheduler._top_k_filtering(logits, top_k=2)
        self.assertAlmostEqual(filtered[0, 1].item(), 4.0)
        self.assertAlmostEqual(filtered[0, 3].item(), 3.0)
        self.assertEqual(filtered[0, 0].item(), torch.finfo(filtered.dtype).min)
        self.assertEqual(filtered[0, 2].item(), torch.finfo(filtered.dtype).min)

    def test_top_k_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_k_filtering(logits, top_k=None)
        self.assertTrue(torch.equal(result, logits))

    def test_top_k_filtering_zero(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_k_filtering(logits, top_k=0)
        self.assertTrue(torch.equal(result, logits))

    def test_top_k_filtering_large_k(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_k_filtering(logits, top_k=100)
        self.assertTrue(torch.equal(result, logits))


class TestSampleFromLogits(unittest.TestCase):
    def test_greedy_sampling(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        tokens, probs = BlockRefinementScheduler._sample_from_logits(
            logits,
            temperature=0.0,
            top_k=None,
            top_p=None,
            generator=None,
            use_multinomial=False,
        )
        self.assertEqual(tokens.item(), 1)
        self.assertEqual(tokens.shape, (1,))
        self.assertEqual(probs.shape, (1,))

    def test_multinomial_sampling(self):
        logits = torch.tensor([[0.0, 100.0, -100.0]])
        gen = torch.Generator().manual_seed(42)
        tokens, probs = BlockRefinementScheduler._sample_from_logits(
            logits,
            temperature=1.0,
            top_k=None,
            top_p=None,
            generator=gen,
            use_multinomial=True,
        )
        self.assertEqual(tokens.item(), 1)

    def test_temperature_scaling(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        tokens, _ = BlockRefinementScheduler._sample_from_logits(
            logits,
            temperature=0.01,
            top_k=None,
            top_p=None,
            generator=None,
            use_multinomial=False,
        )
        self.assertEqual(tokens.item(), 2)

    def test_negative_temperature_raises(self):
        logits = torch.tensor([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            BlockRefinementScheduler._sample_from_logits(
                logits,
                temperature=-1.0,
                top_k=None,
                top_p=None,
                generator=None,
                use_multinomial=False,
            )


if __name__ == "__main__":
    unittest.main()
