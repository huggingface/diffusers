"""
The contract shared by every discrete (token-space) scheduler.

Each scheduler keeps its own test file for its own sampler and edge cases; this file asserts only what a
*pipeline* is allowed to rely on, so that one denoising loop can drive any of them:

    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        sample = scheduler.step(logits, t, sample).prev_sample

A new discrete scheduler is expected to be added to `SCHEDULERS` below and pass unchanged.
"""

import inspect

import pytest
import torch

from diffusers import (
    BlockRefinementScheduler,
    DiscreteDDIMScheduler,
    EntropyBoundScheduler,
    UniformRefinementScheduler,
)
from diffusers.schedulers.scheduling_utils import DiscreteSchedulerOutput


BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_STEPS = 2, 8, 16, 4
MASK_TOKEN_ID = VOCAB_SIZE - 1


def _uniform_sample() -> torch.LongTensor:
    """A canvas for the uniform corruption process: every position holds a real token."""
    return torch.arange(BATCH_SIZE * SEQ_LEN).remainder(VOCAB_SIZE - 1).view(BATCH_SIZE, SEQ_LEN)


def _absorbing_sample() -> torch.LongTensor:
    """A canvas for the absorbing (masked) process: every position is still undecided."""
    return torch.full((BATCH_SIZE, SEQ_LEN), MASK_TOKEN_ID, dtype=torch.long)


# (scheduler class, config, canvas factory) — the config is only what the scheduler cannot default.
SCHEDULERS = [
    pytest.param(DiscreteDDIMScheduler, {}, _uniform_sample, id="discrete_ddim"),
    pytest.param(UniformRefinementScheduler, {}, _uniform_sample, id="uniform_refinement"),
    pytest.param(EntropyBoundScheduler, {}, _uniform_sample, id="entropy_bound"),
    pytest.param(BlockRefinementScheduler, {"mask_token_id": MASK_TOKEN_ID}, _absorbing_sample, id="block_refinement"),
]

parametrize_schedulers = pytest.mark.parametrize("scheduler_cls, config, make_sample", SCHEDULERS)


def _logits(seed: int = 0) -> torch.Tensor:
    """Denoiser logits, fixed per `seed` so a divergence cannot come from the inputs."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, generator=generator)


@parametrize_schedulers
class TestDiscreteSchedulerContract:
    def _scheduler(self, scheduler_cls, config, num_inference_steps: int = NUM_STEPS):
        scheduler = scheduler_cls(**config)
        scheduler.set_timesteps(num_inference_steps)
        return scheduler

    # --- The timestep grid ---

    def test_timesteps_are_a_decreasing_corruption_level_in_the_unit_interval(
        self, scheduler_cls, config, make_sample
    ):
        scheduler = self._scheduler(scheduler_cls, config)

        timesteps = scheduler.timesteps
        assert timesteps.dtype == torch.float32
        assert len(timesteps) == NUM_STEPS
        assert scheduler.num_inference_steps == NUM_STEPS
        # `1.0` is fully corrupted and `0.0` is clean, so the loop runs from 1 down towards (never onto) 0.
        assert timesteps[0].item() == 1.0
        assert timesteps[-1].item() == 1.0 / NUM_STEPS
        assert bool((timesteps[1:] < timesteps[:-1]).all())
        assert bool(((timesteps > 0.0) & (timesteps <= 1.0)).all())

    def test_set_timesteps_rejects_a_non_positive_step_count(self, scheduler_cls, config, make_sample):
        scheduler = scheduler_cls(**config)
        with pytest.raises(ValueError):
            scheduler.set_timesteps(0)

    def test_set_timesteps_clears_the_step_index(self, scheduler_cls, config, make_sample):
        """A pipeline calls `set_timesteps` per block, and that has to be enough to start over."""
        scheduler = self._scheduler(scheduler_cls, config)
        scheduler.step(_logits(), scheduler.timesteps[0], make_sample())
        assert scheduler.step_index is not None

        scheduler.set_timesteps(NUM_STEPS)
        assert scheduler.step_index is None
        assert scheduler.begin_index is None

    # --- Index recovery: the loop passes `t`, the scheduler recovers `i` ---

    def test_every_timestep_recovers_its_index(self, scheduler_cls, config, make_sample):
        scheduler = self._scheduler(scheduler_cls, config)
        for index, timestep in enumerate(scheduler.timesteps):
            assert scheduler.index_for_timestep(timestep) == index

    def test_step_index_starts_from_the_timestep_it_is_given(self, scheduler_cls, config, make_sample):
        """Entering mid-schedule (an image-to-image style start) is recovered from `timestep` alone."""
        scheduler = self._scheduler(scheduler_cls, config)
        scheduler.step(_logits(), scheduler.timesteps[1], make_sample())
        assert scheduler.step_index == 2

    def test_step_advances_the_step_index(self, scheduler_cls, config, make_sample):
        scheduler = self._scheduler(scheduler_cls, config)
        sample = make_sample()
        for index, timestep in enumerate(scheduler.timesteps):
            sample = scheduler.step(_logits(index), timestep, sample).prev_sample
            assert scheduler.step_index == index + 1

    # --- The output contract ---

    def test_step_returns_the_shared_output(self, scheduler_cls, config, make_sample):
        scheduler = self._scheduler(scheduler_cls, config)
        sample = make_sample()
        logits = _logits()

        output = scheduler.step(logits, scheduler.timesteps[0], sample)

        assert isinstance(output, DiscreteSchedulerOutput)
        for field in ("prev_sample", "pred_original_sample"):
            tokens = getattr(output, field)
            assert tokens.shape == sample.shape, field
            assert tokens.dtype == torch.long, field
        assert output.sampled_probs.shape == sample.shape
        assert output.sampled_probs.is_floating_point()
        assert output.pred_logits.shape == logits.shape
        assert output.committed_mask.shape == sample.shape
        assert output.committed_mask.dtype == torch.bool
        # `edited_mask` is the one optional field: only editing schedulers populate it.
        assert output.edited_mask is None or (
            output.edited_mask.shape == sample.shape and output.edited_mask.dtype == torch.bool
        )

    def test_step_returns_a_fixed_arity_tuple(self, scheduler_cls, config, make_sample):
        """`return_dict=False` always yields all six fields, `edited_mask` included, so indices never shift."""
        scheduler = self._scheduler(scheduler_cls, config)
        sample = make_sample()
        logits = _logits()

        # Seeded per call: these schedulers draw, so only a fixed generator makes the two comparable.
        as_tuple = scheduler.step(
            logits, scheduler.timesteps[0], sample, generator=torch.Generator().manual_seed(0), return_dict=False
        )
        scheduler.set_timesteps(NUM_STEPS)
        as_output = scheduler.step(logits, scheduler.timesteps[0], sample, generator=torch.Generator().manual_seed(0))

        assert isinstance(as_tuple, tuple)
        assert len(as_tuple) == 6
        assert torch.equal(as_tuple[0], as_output.prev_sample)
        assert torch.equal(as_tuple[4], as_output.committed_mask)

    def test_sampled_probs_come_from_the_unmodified_distribution(self, scheduler_cls, config, make_sample):
        """
        Confidence is the probability of the drawn token under the raw denoiser distribution.

        This is what makes a confidence threshold portable between schedulers and independent of whatever
        temperature / top-k / top-p the scheduler drew with.
        """
        scheduler = self._scheduler(scheduler_cls, config)
        logits = _logits()

        output = scheduler.step(logits, scheduler.timesteps[0], make_sample())

        expected = torch.softmax(logits.float(), dim=-1).gather(-1, output.pred_original_sample[..., None])
        torch.testing.assert_close(output.sampled_probs, expected.squeeze(-1))

    # --- Interchangeability ---

    def test_the_canonical_signatures_are_uniform(self, scheduler_cls, config, make_sample):
        """A pipeline calls every discrete scheduler the same way; only deprecation shims may add `**kwargs`."""
        step_params = list(inspect.signature(scheduler_cls.step).parameters.values())[1:]
        positional = [p.name for p in step_params if p.kind is p.POSITIONAL_OR_KEYWORD]
        keyword_only = {p.name for p in step_params if p.kind is p.KEYWORD_ONLY}
        assert positional == ["model_output", "timestep", "sample"]
        assert keyword_only == {"generator", "return_dict"}

        set_timesteps_params = list(inspect.signature(scheduler_cls.set_timesteps).parameters.values())[1:]
        assert [p.name for p in set_timesteps_params if p.kind is p.POSITIONAL_OR_KEYWORD] == [
            "num_inference_steps",
            "device",
        ]

    def test_one_loop_denoises_a_whole_block(self, scheduler_cls, config, make_sample):
        """The loop in the module docstring, run end to end — the only thing a pipeline needs to work."""
        scheduler = self._scheduler(scheduler_cls, config)
        sample = make_sample()
        generator = torch.Generator().manual_seed(0)

        for index, timestep in enumerate(scheduler.timesteps):
            sample = scheduler.step(_logits(index), timestep, sample, generator=generator).prev_sample

        assert sample.shape == (BATCH_SIZE, SEQ_LEN)
        assert sample.dtype == torch.long
        assert bool(((sample >= 0) & (sample < VOCAB_SIZE)).all())
        # Whatever the corruption process, a finished block holds no undecided positions.
        assert not bool((sample == MASK_TOKEN_ID).all(dim=-1).any())
