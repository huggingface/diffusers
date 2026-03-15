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

    def test_set_timesteps(self):
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)
        self.assertEqual(scheduler.num_inference_steps, 8)
        self.assertEqual(len(scheduler.timesteps), 8)
        # Timesteps should count down
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
        # 32 / 8 = 4 each, no remainder
        self.assertTrue((schedule == 4).all().item())

    def test_get_num_transfer_tokens_remainder(self):
        scheduler = self.get_scheduler()
        schedule = scheduler.get_num_transfer_tokens(block_length=10, num_inference_steps=3)
        self.assertEqual(schedule.sum().item(), 10)
        self.assertEqual(len(schedule), 3)
        # 10 / 3 = 3 base, 1 remainder -> [4, 3, 3]
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

        batch_size, block_length = 1, 8
        mask_id = 99

        # All positions are masked
        sample = torch.full((batch_size, block_length), mask_id, dtype=torch.long)
        sampled_tokens = torch.arange(block_length, dtype=torch.long).unsqueeze(0)
        # Confidence decreasing: first tokens are most confident
        sampled_probs = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]])

        out = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            threshold=0.95,
            return_dict=True,
        )

        # With 8 tokens and 2 steps, first step should commit 4 tokens
        committed = out.transfer_index[0].sum().item()
        self.assertEqual(committed, 4)
        # The 4 most confident (highest prob) should be committed
        self.assertTrue(out.transfer_index[0, 0].item())
        self.assertTrue(out.transfer_index[0, 1].item())
        self.assertTrue(out.transfer_index[0, 2].item())
        self.assertTrue(out.transfer_index[0, 3].item())

    def test_step_threshold_commits_all_above(self):
        """When enough tokens exceed threshold, commit all of them (not just num_to_transfer)."""
        scheduler = self.get_scheduler(block_length=8)
        scheduler.set_timesteps(4)  # 2 tokens per step

        batch_size, block_length = 1, 8
        mask_id = 99

        sample = torch.full((batch_size, block_length), mask_id, dtype=torch.long)
        sampled_tokens = torch.arange(block_length, dtype=torch.long).unsqueeze(0)
        # 5 tokens above threshold of 0.5
        sampled_probs = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.55, 0.1, 0.1, 0.1]])

        out = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            threshold=0.5,
            return_dict=True,
        )

        # All 5 above threshold should be committed (more than num_to_transfer=2)
        committed = out.transfer_index[0].sum().item()
        self.assertEqual(committed, 5)

    def test_step_no_editing_by_default(self):
        """Without editing_threshold, no non-mask tokens should be changed."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        sample = torch.tensor([[10, 20, 99, 99]], dtype=torch.long)
        sampled_tokens = torch.tensor([[50, 60, 70, 80]], dtype=torch.long)
        sampled_probs = torch.tensor([[0.99, 0.99, 0.99, 0.99]])

        out = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=99,
            editing_threshold=None,
            return_dict=True,
        )

        # Non-mask positions should not be edited
        self.assertFalse(out.editing_transfer_index.any().item())
        # Only mask positions should be committed
        self.assertFalse(out.transfer_index[0, 0].item())
        self.assertFalse(out.transfer_index[0, 1].item())

    def test_step_editing_replaces_tokens(self):
        """With editing_threshold, non-mask tokens with high confidence and different prediction get replaced."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        sample = torch.tensor([[10, 20, 99, 99]], dtype=torch.long)
        # Token 0: model predicts 50 (different from 10) with high confidence
        # Token 1: model predicts 20 (same as current) — should NOT edit
        sampled_tokens = torch.tensor([[50, 20, 70, 80]], dtype=torch.long)
        sampled_probs = torch.tensor([[0.99, 0.99, 0.5, 0.5]])

        out = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=99,
            editing_threshold=0.8,
            return_dict=True,
        )

        # Token 0 should be edited (different prediction, high confidence)
        self.assertTrue(out.editing_transfer_index[0, 0].item())
        # Token 1 should NOT be edited (same prediction)
        self.assertFalse(out.editing_transfer_index[0, 1].item())
        # prev_sample should reflect the edit
        self.assertEqual(out.prev_sample[0, 0].item(), 50)

    def test_step_prompt_mask_prevents_editing(self):
        """Prompt positions should never be edited even with editing enabled."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        sample = torch.tensor([[10, 20, 99, 99]], dtype=torch.long)
        sampled_tokens = torch.tensor([[50, 60, 70, 80]], dtype=torch.long)
        sampled_probs = torch.tensor([[0.99, 0.99, 0.99, 0.99]])
        prompt_mask = torch.tensor([True, True, False, False])

        out = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=99,
            editing_threshold=0.5,
            prompt_mask=prompt_mask,
            return_dict=True,
        )

        # Prompt positions should not be edited
        self.assertFalse(out.editing_transfer_index[0, 0].item())
        self.assertFalse(out.editing_transfer_index[0, 1].item())

    def test_step_return_tuple(self):
        """Verify tuple output when return_dict=False."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        sample = torch.full((1, 4), 99, dtype=torch.long)
        sampled_tokens = torch.arange(4, dtype=torch.long).unsqueeze(0)
        sampled_probs = torch.ones(1, 4)

        result = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=99,
            return_dict=False,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

    def test_step_batched(self):
        """Verify step works with batch_size > 1."""
        scheduler = self.get_scheduler(block_length=4)
        scheduler.set_timesteps(2)

        batch_size = 3
        mask_id = 99
        sample = torch.full((batch_size, 4), mask_id, dtype=torch.long)
        sampled_tokens = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        sampled_probs = torch.rand(batch_size, 4)

        out = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            return_dict=True,
        )

        self.assertEqual(out.prev_sample.shape, (batch_size, 4))
        self.assertEqual(out.transfer_index.shape, (batch_size, 4))

    def test_step_output_shape_matches_input(self):
        """All output tensors should match the input sample shape."""
        scheduler = self.get_scheduler(block_length=8)
        scheduler.set_timesteps(4)

        sample = torch.full((2, 8), 99, dtype=torch.long)
        sampled_tokens = torch.zeros_like(sample)
        sampled_probs = torch.rand(2, 8)

        out = scheduler.step(
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
            timestep=0,
            sample=sample,
            mask_token_id=99,
            return_dict=True,
        )

        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.transfer_index.shape, sample.shape)
        self.assertEqual(out.editing_transfer_index.shape, sample.shape)
        self.assertEqual(out.sampled_tokens.shape, sample.shape)
        self.assertEqual(out.sampled_probs.shape, sample.shape)


if __name__ == "__main__":
    unittest.main()
