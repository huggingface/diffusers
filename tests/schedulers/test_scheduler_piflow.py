# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch

from diffusers import PiflowScheduler
from diffusers.schedulers.scheduling_piflow import DXPolicy, PiflowSchedulerOutput, policy_rollout_fm


class TestPiflowScheduler:
    """
    PiflowScheduler's model output is "widened" (``n_grid`` predictions packed into the channel
    dimension) rather than matching the sample shape, so it cannot use ``SchedulerTesterMixin`` (see
    ``TestFlowMapEulerDiscreteScheduler`` for the same situation with a different non-standard
    contract). These tests exercise the contract `Kandinsky6TI2VAPipeline` actually relies on.
    """

    scheduler_class = PiflowScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 5.0,
            "n_grid": 4,
            "eps": 1e-6,
            "final_step_size_scale": 0.5,
            "num_policy_substeps": 32,
        }
        config.update(**kwargs)
        return config

    def make_widened_output(self, velocity: torch.Tensor, n_grid: int) -> torch.Tensor:
        """Pack one velocity prediction into `n_grid` identical grid slots (channel-minor layout)."""
        return velocity.repeat(1, n_grid)

    # ---- config validation ----

    def test_instantiation_with_defaults(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        assert scheduler.config.num_train_timesteps == 1000
        assert scheduler.config.n_grid == 4

    def test_invalid_n_grid_raises(self):
        with pytest.raises(ValueError):
            self.scheduler_class(**self.get_default_config(n_grid=1))

    def test_invalid_eps_raises(self):
        with pytest.raises(ValueError):
            self.scheduler_class(**self.get_default_config(eps=0.0))

    def test_invalid_final_step_size_scale_raises(self):
        with pytest.raises(ValueError):
            self.scheduler_class(**self.get_default_config(final_step_size_scale=0.0))
        with pytest.raises(ValueError):
            self.scheduler_class(**self.get_default_config(final_step_size_scale=1.5))

    def test_invalid_num_policy_substeps_raises(self):
        with pytest.raises(ValueError):
            self.scheduler_class(**self.get_default_config(num_policy_substeps=0))

    # ---- set_timesteps ----

    def test_set_timesteps_shapes_and_monotonic(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        for nfe in [1, 2, 4, 8, 16]:
            scheduler.set_timesteps(nfe)
            assert scheduler.timesteps.shape == (nfe,)
            assert scheduler.sigmas.shape == (nfe + 1,)
            assert scheduler.sigmas[-1].item() == 0.0
            # strictly decreasing: this schedule should never hit the duplicate-timestep branch
            # in `_step_index_for` that other flow schedulers need for interpolated sigmas.
            diffs = scheduler.timesteps[1:] - scheduler.timesteps[:-1]
            assert torch.all(diffs < 0)

    def test_set_timesteps_rejects_unsupported_args(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        with pytest.raises(ValueError):
            scheduler.set_timesteps(4, sigmas=[1.0, 0.5, 0.0])
        with pytest.raises(ValueError):
            scheduler.set_timesteps(4, mu=1.0)
        with pytest.raises(ValueError):
            scheduler.set_timesteps(4, timesteps=[900, 500, 100])
        with pytest.raises(ValueError):
            scheduler.set_timesteps(0)

    # ---- step() input validation ----

    def test_step_rejects_integer_timestep(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(4)
        sample = torch.randn(1, 3)
        model_output = self.make_widened_output(torch.randn(1, 3), scheduler.n_grid)
        with pytest.raises(ValueError):
            scheduler.step(model_output, 0, sample)

    def test_step_after_exhausted_schedule_raises(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(2)
        sample = torch.randn(1, 3)
        for t in scheduler.timesteps:
            model_output = self.make_widened_output(torch.randn(1, 3), scheduler.n_grid)
            sample = scheduler.step(model_output, t, sample).prev_sample
        with pytest.raises(RuntimeError):
            scheduler.step(
                self.make_widened_output(torch.randn(1, 3), scheduler.n_grid), scheduler.timesteps[-1], sample
            )

    def test_step_return_dict_false_returns_tuple(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(2)
        sample = torch.randn(1, 3)
        model_output = self.make_widened_output(torch.randn(1, 3), scheduler.n_grid)
        output = scheduler.step(model_output, scheduler.timesteps[0], sample, return_dict=False)
        assert isinstance(output, tuple)
        output_dict = scheduler.step(model_output, scheduler.timesteps[0], sample, return_dict=True)
        assert isinstance(output_dict, PiflowSchedulerOutput)

    def test_to_grid_rejects_mismatched_shapes(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(2)
        sample = torch.randn(1, 3)
        with pytest.raises(ValueError):
            # channel count not divisible by n_grid
            scheduler.step(torch.randn(1, 5), scheduler.timesteps[0], sample)
        with pytest.raises(ValueError):
            # divisible by n_grid, but the per-grid width doesn't match the sample's channel count
            scheduler.step(torch.randn(1, 4 * 2), scheduler.timesteps[0], sample)

    # ---- correctness ----

    def test_step_is_deterministic(self):
        # Like every diffusers scheduler with an auto-incrementing `_step_index` (e.g.
        # FlowMatchEulerDiscreteScheduler), `step()` only resolves `timestep` into an index on the
        # first call after `set_timesteps`; a second call reuses the advanced internal counter
        # regardless of the `timestep` passed in. So determinism must be checked across two freshly
        # reset schedulers, not two `step()` calls on the same instance.
        torch.manual_seed(0)
        sample = torch.randn(2, 5)
        model_output = self.make_widened_output(torch.randn(2, 5), self.get_default_config()["n_grid"])

        scheduler1 = self.scheduler_class(**self.get_default_config())
        scheduler1.set_timesteps(4)
        out1 = scheduler1.step(model_output, scheduler1.timesteps[0], sample.clone()).prev_sample

        scheduler2 = self.scheduler_class(**self.get_default_config())
        scheduler2.set_timesteps(4)
        out2 = scheduler2.step(model_output, scheduler2.timesteps[0], sample.clone()).prev_sample

        torch.testing.assert_close(out1, out2)

    def test_no_nan_across_configs(self):
        for n_grid in (2, 4, 8):
            for nfe in (1, 2, 8):
                scheduler = self.scheduler_class(**self.get_default_config(n_grid=n_grid))
                scheduler.set_timesteps(nfe)
                sample = torch.randn(2, 6)
                for t in scheduler.timesteps:
                    model_output = self.make_widened_output(torch.randn(2, 6), n_grid)
                    sample = scheduler.step(model_output, t, sample).prev_sample
                assert not torch.isnan(sample).any(), f"NaN with n_grid={n_grid}, nfe={nfe}"
                assert not torch.isinf(sample).any(), f"Inf with n_grid={n_grid}, nfe={nfe}"

    def test_step_recovers_known_clean_target(self):
        """A "perfect" model whose implied x0 prediction is always exactly `target` (i.e. every grid
        slot predicts the velocity of the straight line from the current sample to `target`) should
        drive the sample to `target` after the full schedule: that straight-line ODE has a velocity
        that is exactly constant along the true path, so the scheduler's Euler-style substep
        integration has zero discretization error and the only residual comes from stopping at
        `eps` instead of sigma=0. This catches sign/broadcasting/indexing bugs that a shape-only or
        no-NaN check would miss.
        """
        torch.manual_seed(0)
        scheduler = self.scheduler_class(**self.get_default_config(n_grid=4, num_policy_substeps=64))
        scheduler.set_timesteps(8)
        batch, dim = 2, 3
        target = torch.randn(batch, dim)
        sample = torch.randn(batch, dim) * 3.0 + 5.0  # arbitrary, far from target

        for i, t in enumerate(scheduler.timesteps):
            sigma = scheduler.sigmas[i]
            velocity = (sample - target) / sigma.clamp(min=scheduler.eps)
            model_output = self.make_widened_output(velocity, scheduler.n_grid)
            sample = scheduler.step(model_output, t, sample).prev_sample

        torch.testing.assert_close(sample, target, atol=1e-3, rtol=1e-3)

    def test_final_step_policy_spans_the_nominal_segment(self):
        """On the last step the policy's grid spans the nominal segment down to 0, while the rollout stops at
        `eps`, as in the native samplers the distilled checkpoints were trained with."""
        torch.manual_seed(0)
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(4)
        sample = torch.randn(2, 3)
        model_output = torch.randn(2, 3 * scheduler.n_grid)

        for t in scheduler.timesteps[:-1]:
            sample = scheduler.step(model_output, t, sample).prev_sample
        out = scheduler.step(model_output, scheduler.timesteps[-1], sample).prev_sample

        raw_src = scheduler._piflow_raw_timesteps[-1]
        segment = 1.0 / (4 - 0.5) * 0.5
        sigma = scheduler.sigmas[-2].expand(2).reshape(2, 1)
        policy = DXPolicy(
            scheduler._to_grid(model_output, sample),
            sample,
            sigma,
            torch.full((2,), segment),
            shift=scheduler.config.shift,
            eps=scheduler.eps,
        )
        assert torch.all(policy.raw_t_dst == 0)
        expected, _, _ = policy_rollout_fm(
            sample,
            sigma,
            raw_src.expand(2),
            (raw_src - segment).clamp(min=scheduler.eps).expand(2),
            scheduler.num_policy_substeps,
            policy,
        )
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
