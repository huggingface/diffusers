import tempfile
import unittest

import torch

from diffusers import BlockRefinementScheduler


class BlockRefinementSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {
            "block_length": 32,
            "num_inference_steps": 8,
            "mask_token_id": 31,
            "threshold": 0.95,
            "editing_threshold": None,
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
        # Same decreasing corruption-level grid as the other discrete schedulers.
        self.assertEqual(scheduler.timesteps.dtype, torch.float32)
        self.assertEqual(scheduler.timesteps[0].item(), 1.0)
        self.assertEqual(scheduler.timesteps[-1].item(), 1.0 / 8)
        self.assertTrue(bool((scheduler.timesteps[1:] < scheduler.timesteps[:-1]).all()))

    def test_set_timesteps_invalid(self):
        scheduler = self.get_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(0)

    def test_deprecated_get_num_transfer_tokens_still_works(self):
        scheduler = self.get_scheduler()
        with self.assertWarns(FutureWarning):
            schedule = scheduler.get_num_transfer_tokens(block_length=10, num_inference_steps=3)
        self.assertEqual(schedule.tolist(), [4, 3, 3])

    def test_deprecated_set_timesteps_block_length_is_ignored(self):
        scheduler = self.get_scheduler()
        with self.assertWarns(FutureWarning):
            scheduler.set_timesteps(4, block_length=16)
        self.assertEqual(len(scheduler.timesteps), 4)

    def test_invalid_config_raises(self):
        # These knobs moved here from `LLaDA2Pipeline.__call__`; their validation moved with them.
        with self.assertRaises(ValueError):
            self.get_scheduler(sampling_method="multinomail")
        with self.assertRaises(ValueError):
            self.get_scheduler(threshold=-0.5)

    def test_save_load_config_round_trip(self):
        scheduler = self.get_scheduler(block_length=64, threshold=0.8, editing_threshold=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded = BlockRefinementScheduler.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.block_length, 64)
        self.assertEqual(loaded.config.threshold, 0.8)
        self.assertEqual(loaded.config.editing_threshold, 0.5)

    def test_from_config(self):
        scheduler = self.get_scheduler(block_length=16, threshold=0.7)
        new_scheduler = BlockRefinementScheduler.from_config(scheduler.config)
        self.assertEqual(new_scheduler.config.block_length, 16)
        self.assertEqual(new_scheduler.config.threshold, 0.7)

    def test_step_commits_tokens(self):
        """Verify that step() commits mask tokens based on confidence."""
        scheduler = self.get_scheduler(block_length=8, num_inference_steps=2)
        scheduler.set_timesteps(2)

        batch_size, block_length, vocab_size = 1, 8, 32
        mask_id = 31

        sample = torch.full((batch_size, block_length), mask_id, dtype=torch.long)
        # Create logits where confidence decreases with position
        logits = torch.zeros(batch_size, block_length, vocab_size)
        for i in range(block_length):
            logits[0, i, i] = 10.0 - i  # decreasing confidence

        out = scheduler.step(model_output=logits, timestep=scheduler.timesteps[0], sample=sample, return_dict=True)

        # With 8 tokens and 2 steps, first step should commit 4 tokens
        committed = out.committed_mask[0].sum().item()
        self.assertEqual(committed, 4)

    def test_step_no_editing_by_default(self):
        """Without editing_threshold, no non-mask tokens should be changed."""
        scheduler = self.get_scheduler(block_length=4, num_inference_steps=2)
        scheduler.set_timesteps(2)

        vocab_size = 32
        sample = torch.tensor([[10, 20, 31, 31]], dtype=torch.long)
        logits = torch.zeros(1, 4, vocab_size)
        logits[0, :, 15] = 10.0  # predict token 15 for all positions

        out = scheduler.step(model_output=logits, timestep=scheduler.timesteps[0], sample=sample, return_dict=True)

        self.assertFalse(out.edited_mask.any().item())
        self.assertFalse(out.committed_mask[0, 0].item())
        self.assertFalse(out.committed_mask[0, 1].item())

    def test_step_editing_replaces_tokens(self):
        """With editing_threshold, non-mask tokens with high confidence and different prediction get replaced."""
        scheduler = self.get_scheduler(block_length=4, num_inference_steps=2, editing_threshold=0.5)
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

        out = scheduler.step(model_output=logits, timestep=scheduler.timesteps[0], sample=sample, return_dict=True)

        # Token 0 should be edited (different prediction, high confidence)
        self.assertTrue(out.edited_mask[0, 0].item())
        # Token 1 should NOT be edited (same prediction)
        self.assertFalse(out.edited_mask[0, 1].item())

    def test_deprecated_prompt_mask_still_honoured(self):
        """`prompt_mask` moved to the pipeline, but the deprecated kwarg keeps working for one release."""
        scheduler = self.get_scheduler(block_length=4, num_inference_steps=2, editing_threshold=0.5)
        scheduler.set_timesteps(2)

        sample = torch.tensor([[10, 20, 31, 31]], dtype=torch.long)
        logits = torch.zeros(1, 4, 32)
        logits[0, :, 15] = 20.0
        prompt_mask = torch.tensor([True, True, False, False])

        with self.assertWarns(FutureWarning):
            out = scheduler.step(
                model_output=logits,
                timestep=scheduler.timesteps[0],
                sample=sample,
                prompt_mask=prompt_mask,
                return_dict=True,
            )
        self.assertFalse(out.edited_mask[0, 0].item())
        self.assertFalse(out.edited_mask[0, 1].item())

    def test_step_edit_edits_without_a_timestep(self):
        """The post-mask phase is a confidence-thresholded overwrite, not a diffusion step."""
        scheduler = self.get_scheduler(block_length=4, num_inference_steps=2, editing_threshold=0.5)
        scheduler.set_timesteps(2)

        sample = torch.tensor([[10, 20, 5, 6]], dtype=torch.long)  # fully resolved, no mask tokens
        logits = torch.zeros(1, 4, 32)
        logits[0, 0, 15] = 20.0  # differs from 10 -> edited
        logits[0, 1, 20] = 20.0  # equals current -> not edited
        logits[0, 2, 5] = 20.0
        logits[0, 3, 6] = 20.0

        out = scheduler.step_edit(model_output=logits, sample=sample)
        self.assertTrue(out.edited_mask[0, 0].item())
        self.assertFalse(out.edited_mask[0, 1].item())
        self.assertEqual(out.prev_sample[0, 0].item(), 15)
        # Nothing is being unmasked, so nothing commits.
        self.assertFalse(out.committed_mask.any().item())

    def test_step_edit_does_not_advance_step_index(self):
        # A pipeline runs the editing phase after exhausting the schedule, so extra sweeps must not overrun it.
        scheduler = self.get_scheduler(block_length=4, num_inference_steps=2, editing_threshold=0.5)
        scheduler.set_timesteps(2)
        sample = torch.tensor([[10, 20, 5, 6]], dtype=torch.long)
        logits = torch.randn(1, 4, 32)

        scheduler.step(model_output=logits, timestep=scheduler.timesteps[0], sample=sample)
        self.assertEqual(scheduler.step_index, 1)
        for _ in range(5):
            scheduler.step_edit(model_output=logits, sample=sample)
        self.assertEqual(scheduler.step_index, 1)

    def test_mask_token_id_required(self):
        scheduler = self.get_scheduler(mask_token_id=None, num_inference_steps=2)
        scheduler.set_timesteps(2)
        sample = torch.full((1, 4), 31, dtype=torch.long)
        logits = torch.randn(1, 4, 32)
        with self.assertRaisesRegex(ValueError, "UniformRefinementScheduler"):
            scheduler.step(model_output=logits, timestep=scheduler.timesteps[0], sample=sample)

    def test_step_return_tuple(self):
        """Verify tuple output when return_dict=False."""
        scheduler = self.get_scheduler(block_length=4, num_inference_steps=2)
        scheduler.set_timesteps(2)

        vocab_size = 32
        sample = torch.full((1, 4), 31, dtype=torch.long)
        logits = torch.randn(1, 4, vocab_size)

        result = scheduler.step(model_output=logits, timestep=scheduler.timesteps[0], sample=sample, return_dict=False)

        self.assertIsInstance(result, tuple)
        # Fixed arity, always including the trailing `edited_mask` (matches AmusedScheduler).
        self.assertEqual(len(result), 6)

    def test_step_batched(self):
        """Verify step works with batch_size > 1."""
        scheduler = self.get_scheduler(block_length=4, num_inference_steps=2)
        scheduler.set_timesteps(2)

        batch_size, vocab_size = 3, 32
        mask_id = 31
        sample = torch.full((batch_size, 4), mask_id, dtype=torch.long)
        logits = torch.randn(batch_size, 4, vocab_size)

        out = scheduler.step(model_output=logits, timestep=scheduler.timesteps[0], sample=sample, return_dict=True)

        self.assertEqual(out.prev_sample.shape, (batch_size, 4))
        self.assertEqual(out.committed_mask.shape, (batch_size, 4))

    def test_deprecated_loop_control_helpers_still_work(self):
        """Both moved to the pipeline (they are loop control, not scheduling) but warn rather than vanish."""
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(8)

        with self.assertWarns(FutureWarning):
            should_continue = scheduler.check_block_should_continue(
                step_idx=0,
                masks_remaining=True,
                editing_enabled=False,
                editing_transfer_index=torch.zeros(2, 32, dtype=torch.bool),
                post_steps=0,
                max_post_steps=16,
                finished=torch.tensor([True, True]),
            )
        self.assertFalse(should_continue)

        mask_id, eos_id = 99, 2
        with self.assertWarns(FutureWarning):
            finished = BlockRefinementScheduler.check_eos_finished(
                cur_x=torch.tensor([[10, 11, 5, eos_id, mask_id, mask_id]], dtype=torch.long),
                sampled_tokens=torch.tensor([[0, 0, 0, eos_id]], dtype=torch.long),
                final_transfer=torch.tensor([[False, False, False, True]]),
                finished=torch.tensor([False]),
                eos_token_id=eos_id,
                mask_token_id=mask_id,
                prompt_length=2,
            )
        self.assertTrue(finished[0].item())

    def test_add_noise_masks_at_the_given_rate(self):
        scheduler = self.get_scheduler(mask_token_id=99)
        original = torch.arange(1, 9, dtype=torch.long).repeat(4, 1)

        gen = torch.Generator().manual_seed(42)
        noisy, mask = scheduler.add_noise(original, 1.0, generator=gen)
        self.assertTrue(bool(mask.all()))
        self.assertTrue(bool((noisy == 99).all()))

        noisy, mask = scheduler.add_noise(original, 0.0, generator=gen)
        self.assertFalse(bool(mask.any()))
        self.assertTrue(torch.equal(noisy, original))

    def test_add_noise_accepts_per_example_rates(self):
        scheduler = self.get_scheduler(mask_token_id=99)
        original = torch.arange(1, 9, dtype=torch.long).repeat(2, 1)
        gen = torch.Generator().manual_seed(0)
        noisy, mask = scheduler.add_noise(original, torch.tensor([1.0, 0.0]), generator=gen)
        self.assertTrue(bool(mask[0].all()))
        self.assertFalse(bool(mask[1].any()))
        self.assertTrue(bool((noisy[mask] == 99).all()))

    def test_add_noise_rejects_the_old_signature(self):
        # The second positional argument changed meaning, so a silent shim would corrupt training data.
        scheduler = self.get_scheduler(mask_token_id=99)
        original = torch.arange(1, 9, dtype=torch.long).repeat(1, 1)
        with self.assertRaisesRegex(ValueError, "hard break"):
            scheduler.add_noise(original, torch.ones_like(original), prompt_length=2, block_length=4, mask_token_id=99)


if __name__ == "__main__":
    unittest.main()
