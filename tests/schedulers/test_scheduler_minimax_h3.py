# Copyright 2026 The HuggingFace Team.
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

from diffusers import MiniMaxH3Scheduler


SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


class TestMiniMaxH3SchedulerEulerUpdateForm:
    def get_scheduler(self, **kwargs):
        scheduler = MiniMaxH3Scheduler(shift_provided_sigmas=False, **kwargs)
        scheduler.set_timesteps(sigmas=SIGMAS)
        return scheduler

    def test_velocity_form_is_the_velocity_update(self):
        r"""Every step is `x + (sigma - sigma_next) * v`, bit for bit."""
        scheduler = self.get_scheduler(euler_update_form="velocity")
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(4, 96, generator=generator)
        sigmas = torch.tensor(SIGMAS, dtype=torch.float32)

        for index, timestep in enumerate(scheduler.timesteps):
            velocity = torch.randn(4, 96, generator=generator)
            expected = sample + (sigmas[index] - sigmas[index + 1]) * velocity
            sample = scheduler.step(velocity, timestep, sample).prev_sample
            assert torch.equal(sample, expected)

    def test_forms_agree_up_to_rounding(self):
        r"""The velocity form and the default blend are the same step; they differ in float32 rounding only."""
        velocity_form = self.get_scheduler(euler_update_form="velocity")
        blend_form = self.get_scheduler(euler_update_form="blend")
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(4, 96, generator=generator)
        velocity = torch.randn(4, 96, generator=generator)

        velocity_sample = velocity_form.step(velocity, velocity_form.timesteps[0], sample).prev_sample
        blend_sample = blend_form.step(velocity, blend_form.timesteps[0], sample).prev_sample

        torch.testing.assert_close(velocity_sample, blend_sample, rtol=0, atol=1e-5)

    def test_half_precision_sample_is_stepped_in_float32(self):
        scheduler = self.get_scheduler(euler_update_form="velocity")
        sample = torch.randn(4, 96, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
        velocity = torch.randn(4, 96, generator=torch.Generator().manual_seed(1))

        prev_sample = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample

        expected = sample.float() + (torch.tensor(SIGMAS[0]) - torch.tensor(SIGMAS[1])) * velocity
        assert prev_sample.dtype == torch.bfloat16
        assert torch.equal(prev_sample, expected.to(torch.bfloat16))

    def test_blend_is_the_default(self):
        assert MiniMaxH3Scheduler().config.euler_update_form == "blend"

    def test_unknown_form(self):
        with pytest.raises(ValueError, match="must be 'blend' or 'velocity'"):
            MiniMaxH3Scheduler(euler_update_form="midpoint")

    def test_has_no_effect_with_stochastic_sampling(self):
        with pytest.raises(ValueError, match="has no effect"):
            MiniMaxH3Scheduler(stochastic_sampling=True, euler_update_form="velocity")
