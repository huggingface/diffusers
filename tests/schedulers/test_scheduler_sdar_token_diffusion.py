import tempfile
import unittest

import torch

from diffusers import SDARTokenDiffusionScheduler


class SDARTokenDiffusionSchedulerTest(unittest.TestCase):
    def get_scheduler(self, **kwargs):
        config = {
            "block_length": 32,
            "num_inference_steps": 8,
            "remasking_strategy": "low_confidence_dynamic",
            "confidence_threshold": 0.9,
            "entropy_threshold": 0.35,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
        }
        config.update(kwargs)
        return SDARTokenDiffusionScheduler(**config)

    # ------------------------------------------------------------------
    # set_timesteps
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # get_num_transfer_tokens
    # ------------------------------------------------------------------
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
        # First `remainder` entries get +1
        self.assertEqual(schedule[0].item(), 4)
        self.assertEqual(schedule[1].item(), 3)
        self.assertEqual(schedule[2].item(), 3)

    # ------------------------------------------------------------------
    # save / load config round trip
    # ------------------------------------------------------------------
    def test_save_load_config_round_trip(self):
        scheduler = self.get_scheduler(
            block_length=64,
            remasking_strategy="sequential",
            confidence_threshold=0.8,
            entropy_threshold=0.5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded = SDARTokenDiffusionScheduler.from_pretrained(tmpdir)

        self.assertEqual(loaded.config.block_length, 64)
        self.assertEqual(loaded.config.remasking_strategy, "sequential")
        self.assertEqual(loaded.config.confidence_threshold, 0.8)
        self.assertEqual(loaded.config.entropy_threshold, 0.5)

    # ------------------------------------------------------------------
    # from_config
    # ------------------------------------------------------------------
    def test_from_config(self):
        scheduler = self.get_scheduler(block_length=16, remasking_strategy="entropy_bounded")
        new_scheduler = SDARTokenDiffusionScheduler.from_config(scheduler.config)
        self.assertEqual(new_scheduler.config.block_length, 16)
        self.assertEqual(new_scheduler.config.remasking_strategy, "entropy_bounded")

    # ------------------------------------------------------------------
    # step – remasking strategies
    # ------------------------------------------------------------------
    def _make_step_inputs(self, batch_size=1, block_length=8, vocab_size=32, mask_id=31, num_steps=2):
        sample = torch.full((batch_size, block_length), mask_id, dtype=torch.long)
        logits = torch.zeros(batch_size, block_length, vocab_size)
        for i in range(block_length):
            logits[:, i, i % (vocab_size - 1)] = 10.0 - i  # decreasing confidence
        scheduler = self.get_scheduler(block_length=block_length, num_inference_steps=num_steps)
        scheduler.set_timesteps(num_steps)
        num_transfer_tokens = scheduler.get_num_transfer_tokens(block_length, num_steps)
        return scheduler, logits, sample, num_transfer_tokens, mask_id

    def test_step_sequential(self):
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs()
        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            num_transfer_tokens=ntt,
            remasking_strategy="sequential",
            temperature=0.0,
            return_dict=True,
        )
        # With 8 tokens and 2 steps, first step commits 4 tokens sequentially from the first mask
        committed = out.transfer_index[0].sum().item()
        self.assertEqual(committed, 4)
        # Sequential: first 4 positions should be committed
        self.assertTrue(out.transfer_index[0, :4].all().item())
        self.assertFalse(out.transfer_index[0, 4:].any().item())

    def test_step_low_confidence_static(self):
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs()
        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            num_transfer_tokens=ntt,
            remasking_strategy="low_confidence_static",
            temperature=0.0,
            return_dict=True,
        )
        committed = out.transfer_index[0].sum().item()
        self.assertEqual(committed, 4)

    def test_step_low_confidence_dynamic(self):
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs()
        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            num_transfer_tokens=ntt,
            remasking_strategy="low_confidence_dynamic",
            confidence_threshold=0.9,
            temperature=0.0,
            return_dict=True,
        )
        # Should commit at least step_transfer tokens
        committed = out.transfer_index[0].sum().item()
        self.assertGreaterEqual(committed, 4)

    def test_step_entropy_bounded(self):
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs()
        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            num_transfer_tokens=ntt,
            remasking_strategy="entropy_bounded",
            entropy_threshold=0.35,
            temperature=0.0,
            return_dict=True,
        )
        committed = out.transfer_index[0].sum().item()
        self.assertGreater(committed, 0)

    def test_step_unknown_strategy_raises(self):
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs()
        with self.assertRaises(ValueError):
            scheduler.step(
                model_output=logits,
                timestep=0,
                sample=sample,
                mask_token_id=mask_id,
                num_transfer_tokens=ntt,
                remasking_strategy="nonexistent",
                temperature=0.0,
            )

    # ------------------------------------------------------------------
    # step – output shapes
    # ------------------------------------------------------------------
    def test_step_output_shapes(self):
        batch_size, block_length, vocab_size = 2, 8, 32
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs(
            batch_size=batch_size, block_length=block_length, vocab_size=vocab_size
        )
        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            num_transfer_tokens=ntt,
            temperature=0.0,
            return_dict=True,
        )
        self.assertEqual(out.prev_sample.shape, (batch_size, block_length))
        self.assertEqual(out.transfer_index.shape, (batch_size, block_length))
        self.assertEqual(out.sampled_tokens.shape, (batch_size, block_length))
        self.assertEqual(out.sampled_probs.shape, (batch_size, block_length))

    # ------------------------------------------------------------------
    # step – return_dict=False
    # ------------------------------------------------------------------
    def test_step_return_tuple(self):
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs()
        result = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            num_transfer_tokens=ntt,
            temperature=0.0,
            return_dict=False,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    # ------------------------------------------------------------------
    # step – batched
    # ------------------------------------------------------------------
    def test_step_batched(self):
        batch_size = 3
        scheduler, logits, sample, ntt, mask_id = self._make_step_inputs(batch_size=batch_size)
        out = scheduler.step(
            model_output=logits,
            timestep=0,
            sample=sample,
            mask_token_id=mask_id,
            num_transfer_tokens=ntt,
            temperature=0.0,
            return_dict=True,
        )
        self.assertEqual(out.prev_sample.shape, (batch_size, 8))
        self.assertEqual(out.transfer_index.shape, (batch_size, 8))

    # ------------------------------------------------------------------
    # sample – greedy and multinomial
    # ------------------------------------------------------------------
    def test_sample_greedy(self):
        scheduler = self.get_scheduler()
        logits = torch.tensor([[[1.0, 5.0, 2.0]]])  # (1, 1, 3)
        tokens, probs = scheduler.sample(logits, temperature=0.0)
        self.assertEqual(tokens.item(), 1)
        self.assertEqual(tokens.shape, (1, 1))
        self.assertEqual(probs.shape, (1, 1))

    def test_sample_multinomial(self):
        scheduler = self.get_scheduler()
        logits = torch.tensor([[[0.0, 100.0, -100.0]]])
        gen = torch.Generator().manual_seed(42)
        tokens, probs = scheduler.sample(logits, temperature=1.0, generator=gen)
        self.assertEqual(tokens.item(), 1)

    # ------------------------------------------------------------------
    # check_should_stop
    # ------------------------------------------------------------------
    def test_check_should_stop_with_stop_tokens(self):
        scheduler = self.get_scheduler()
        sequences = torch.tensor([[1, 2, 3, 99, 5]], dtype=torch.long)
        self.assertTrue(scheduler.check_should_stop(sequences, prompt_length=2, stop_token_ids=[99]))

    def test_check_should_stop_without_stop_tokens(self):
        scheduler = self.get_scheduler()
        sequences = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        self.assertFalse(scheduler.check_should_stop(sequences, prompt_length=2, stop_token_ids=None))

    def test_check_should_stop_no_match(self):
        scheduler = self.get_scheduler()
        sequences = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        self.assertFalse(scheduler.check_should_stop(sequences, prompt_length=2, stop_token_ids=[99]))

    def test_check_should_stop_in_prompt_only(self):
        scheduler = self.get_scheduler()
        # Stop token present only in the prompt region — should NOT trigger stop
        sequences = torch.tensor([[99, 2, 3, 4, 5]], dtype=torch.long)
        self.assertFalse(scheduler.check_should_stop(sequences, prompt_length=2, stop_token_ids=[99]))

    # ------------------------------------------------------------------
    # add_noise
    # ------------------------------------------------------------------
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


if __name__ == "__main__":
    unittest.main()
