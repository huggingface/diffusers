import pytest
import torch

from diffusers import BlockRefinementScheduler


class TestBlockRefinementScheduler:
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
        assert scheduler.num_inference_steps == 8
        assert len(scheduler.timesteps) == 8
        assert scheduler.timesteps[0].item() == 7
        assert scheduler.timesteps[-1].item() == 0

    def test_set_timesteps_invalid(self):
        scheduler = self.get_scheduler()
        with pytest.raises(ValueError):
            scheduler.set_timesteps(0)

    def test_get_num_transfer_tokens_even(self):
        scheduler = self.get_scheduler()
        schedule = scheduler.get_num_transfer_tokens(block_length=32, num_inference_steps=8)
        assert schedule.sum().item() == 32
        assert len(schedule) == 8
        assert (schedule == 4).all().item()

    def test_get_num_transfer_tokens_remainder(self):
        scheduler = self.get_scheduler()
        schedule = scheduler.get_num_transfer_tokens(block_length=10, num_inference_steps=3)
        assert schedule.sum().item() == 10
        assert len(schedule) == 3
        assert schedule[0].item() == 4
        assert schedule[1].item() == 3
        assert schedule[2].item() == 3

    def test_transfer_schedule_created_on_set_timesteps(self):
        scheduler = self.get_scheduler(block_length=16)
        scheduler.set_timesteps(4)
        assert scheduler._transfer_schedule is not None
        assert scheduler._transfer_schedule.sum().item() == 16

    def test_save_load_config_round_trip(self, tmp_path):
        scheduler = self.get_scheduler(block_length=64, threshold=0.8, editing_threshold=0.5, minimal_topk=2)
        scheduler.save_config(tmp_path)
        loaded = BlockRefinementScheduler.from_pretrained(tmp_path)

        assert loaded.config.block_length == 64
        assert loaded.config.threshold == 0.8
        assert loaded.config.editing_threshold == 0.5
        assert loaded.config.minimal_topk == 2

    def test_from_config(self):
        scheduler = self.get_scheduler(block_length=16, threshold=0.7)
        new_scheduler = BlockRefinementScheduler.from_config(scheduler.config)
        assert new_scheduler.config.block_length == 16
        assert new_scheduler.config.threshold == 0.7

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
        assert committed == 4

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

        assert not out.editing_transfer_index.any().item()
        assert not out.transfer_index[0, 0].item()
        assert not out.transfer_index[0, 1].item()

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
        assert out.editing_transfer_index[0, 0].item()
        # Token 1 should NOT be edited (same prediction)
        assert not out.editing_transfer_index[0, 1].item()

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

        assert not out.editing_transfer_index[0, 0].item()
        assert not out.editing_transfer_index[0, 1].item()

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

        assert isinstance(result, tuple)
        assert len(result) == 6

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

        assert out.prev_sample.shape == (batch_size, 4)
        assert out.transfer_index.shape == (batch_size, 4)

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
        assert not result

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
        assert not result

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
        assert not result

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
        assert finished[0].item()

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
        assert not finished[0].item()

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
        assert finished[0].item()

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
        assert not masked[0, 0].item()
        assert not masked[0, 1].item()
        assert not masked_rev[0, 0].item()
        assert not masked_rev[0, 1].item()

        # Noisy should have mask_token_id where masked is True
        assert (noisy[masked] == mask_token_id).all().item()
        assert (noisy_rev[masked_rev] == mask_token_id).all().item()

        # masked and masked_rev should be complementary within valid non-prompt positions
        non_prompt = torch.zeros_like(masked)
        non_prompt[0, 2:] = True
        combined = masked | masked_rev
        assert (combined[0, 2:] == non_prompt[0, 2:]).all().item()


class TestTopPFiltering:
    def test_top_p_filtering(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        filtered = BlockRefinementScheduler._top_p_filtering(logits, top_p=0.5)
        assert (filtered > torch.finfo(filtered.dtype).min).any()
        assert (filtered == torch.finfo(filtered.dtype).min).any()

    def test_top_p_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_p_filtering(logits, top_p=None)
        assert torch.equal(result, logits)

    def test_top_p_filtering_one(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_p_filtering(logits, top_p=1.0)
        assert torch.equal(result, logits)


class TestTopKFiltering:
    def test_top_k_filtering(self):
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        filtered = BlockRefinementScheduler._top_k_filtering(logits, top_k=2)
        assert filtered[0, 1].item() == pytest.approx(4.0, abs=1e-7)
        assert filtered[0, 3].item() == pytest.approx(3.0, abs=1e-7)
        assert filtered[0, 0].item() == torch.finfo(filtered.dtype).min
        assert filtered[0, 2].item() == torch.finfo(filtered.dtype).min

    def test_top_k_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_k_filtering(logits, top_k=None)
        assert torch.equal(result, logits)

    def test_top_k_filtering_zero(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_k_filtering(logits, top_k=0)
        assert torch.equal(result, logits)

    def test_top_k_filtering_large_k(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = BlockRefinementScheduler._top_k_filtering(logits, top_k=100)
        assert torch.equal(result, logits)


class TestSampleFromLogits:
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
        assert tokens.item() == 1
        assert tokens.shape == (1,)
        assert probs.shape == (1,)

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
        assert tokens.item() == 1

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
        assert tokens.item() == 2

    def test_negative_temperature_raises(self):
        logits = torch.tensor([[1.0, 2.0]])
        with pytest.raises(ValueError):
            BlockRefinementScheduler._sample_from_logits(
                logits,
                temperature=-1.0,
                top_k=None,
                top_p=None,
                generator=None,
                use_multinomial=False,
            )


class TestBlockRefinementSchedulerUniform:
    """Tests for the uniform corruption mode (`mask_token_id=None`), matching DiffusionGemma's block refinement."""

    def get_scheduler(self, **kwargs):
        config = {"block_length": 256, "num_inference_steps": 48, "threshold": 1.0, "editing_threshold": None}
        config.update(kwargs)
        scheduler = BlockRefinementScheduler(**config)
        scheduler.set_timesteps(config["num_inference_steps"], block_length=config["block_length"])
        return scheduler

    def test_cumulative_quota_progression(self):
        # threshold=1.0 disables threshold commits, so only the even per-step quota applies: ceil(256/48)=6, then 11.
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        out0 = scheduler.step(logits, timestep=0, sample=sample, mask_token_id=None)
        assert scheduler._committed.sum().item() == 6
        scheduler.step(logits, timestep=1, sample=out0.prev_sample, mask_token_id=None)
        assert scheduler._committed.sum().item() == 11

    def test_last_step_commits_all(self):
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        scheduler.step(logits, timestep=0, sample=sample, mask_token_id=None)
        scheduler.step(logits, timestep=47, sample=sample, mask_token_id=None)
        assert scheduler._committed.all()

    def test_threshold_commits_beyond_quota(self):
        scheduler = self.get_scheduler(threshold=0.5)
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        logits[0, torch.arange(20), 0] = 1e6  # 20 high-confidence positions (token 0)
        scheduler.step(logits, timestep=0, sample=sample, mask_token_id=None, temperature=0.0)
        # 20 positions exceed the threshold and get committed regardless of the quota
        assert scheduler._committed.sum().item() == 20

    def test_editing_replaces_committed_token(self):
        scheduler = self.get_scheduler(threshold=1.0, editing_threshold=0.5)
        sample = torch.zeros(1, 256, dtype=torch.long)
        scheduler._committed = torch.ones_like(sample, dtype=torch.bool)  # pretend all committed
        logits = torch.zeros(1, 256, 10000)
        logits[0, 0, 1] = 1e6  # confidently predicts token 1 at position 0 (differs from current token 0)
        out = scheduler.step(logits, timestep=24, sample=sample, mask_token_id=None, temperature=0.0)
        assert out.prev_sample[0, 0].item() == 1
        assert (out.prev_sample[0, 1:] == 0).all()

    def test_reset_on_new_block(self):
        scheduler = self.get_scheduler()
        sample = torch.randint(0, 10000, (1, 256))
        logits = torch.zeros(1, 256, 10000)
        scheduler.step(logits, timestep=5, sample=sample, mask_token_id=None)
        scheduler.step(logits, timestep=0, sample=sample, mask_token_id=None)  # new block resets committed
        assert scheduler._committed.sum().item() == 6
