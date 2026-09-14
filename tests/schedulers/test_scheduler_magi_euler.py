# Copyright 2025 SandAI and The HuggingFace Team. All rights reserved.
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

from diffusers import MagiEulerScheduler


class TestMagiEulerScheduler:
    @pytest.mark.parametrize("steps", [1, 4, 12, 16, 64])
    @pytest.mark.parametrize("time_schedule", ["sd3", "square", "piecewise", "linear"])
    def test_time_grid(self, steps, time_schedule):
        scheduler = MagiEulerScheduler(time_schedule=time_schedule)
        scheduler.set_timesteps(steps)
        grid = scheduler.timestep_schedule
        assert grid.dtype == torch.float32
        assert scheduler.timesteps.shape == (steps,)
        assert grid.shape == (steps + 1,)
        assert grid[0] == 0
        torch.testing.assert_close(grid[-1], torch.tensor(1.0), atol=2e-7, rtol=0)
        assert (grid.diff() > 0).all()
        torch.testing.assert_close(scheduler.timesteps, grid[:-1], atol=0, rtol=0)

    @pytest.mark.parametrize(
        "mode,first", [("8,16,16", [0, 0.125, 0.1875, 0.25]), ("16,16,8", [0, 0.0625, 0.125, 0.25])]
    )
    def test_twelve_step_shortcut(self, mode, first):
        scheduler = MagiEulerScheduler(time_schedule="linear", shortcut_mode=mode)
        scheduler.set_timesteps(12)
        torch.testing.assert_close(scheduler.timesteps[:4], torch.tensor(first), atol=0, rtol=0)

    def test_sd3_operation_order_and_endpoint(self):
        scheduler = MagiEulerScheduler()
        scheduler.set_timesteps(64)
        squared = torch.linspace(0, 1, 65) ** 2
        expected = (1 / 3) * squared / (1 + (1 / 3 - 1) * squared)
        torch.testing.assert_close(scheduler.timestep_schedule, expected, atol=0, rtol=0)
        assert scheduler.timestep_schedule[-1] > 1

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_fp32_update(self, dtype):
        scheduler = MagiEulerScheduler(time_schedule="linear")
        scheduler.set_timesteps(4)
        sample = torch.randn((2, 4, 4, 4, 6), generator=torch.Generator().manual_seed(0)).to(dtype)
        velocity = torch.randn(sample.shape, generator=torch.Generator().manual_seed(1)).to(dtype)
        original = sample.clone()
        output = scheduler.step(velocity, 0.25, sample, next_timestep=0.5)
        assert output.prev_sample.dtype == torch.float32
        torch.testing.assert_close(output.prev_sample, sample.float() + velocity.float() * 0.25, atol=0, rtol=0)
        torch.testing.assert_close(sample, original, atol=0, rtol=0)
        assert scheduler.step_index is None
        torch.testing.assert_close(
            scheduler.step(velocity, 0.25, sample, next_timestep=0.5, return_dict=False)[0],
            output.prev_sample,
            atol=0,
            rtol=0,
        )

    def test_per_batch_chunk_update(self):
        scheduler = MagiEulerScheduler()
        scheduler.set_timesteps(64)
        sample = torch.randn((2, 4, 6, 4, 6), generator=torch.Generator().manual_seed(0))
        velocity = sample.sin()
        before = torch.tensor([[0.4, 0.2, 0.0], [0.6, 0.4, 0.2]])
        after = torch.tensor([[0.5, 0.3, 0.1], [0.7, 0.5, 0.3]])
        expected = torch.cat(
            [
                sample[:, :, index * 2 : (index + 1) * 2]
                + velocity[:, :, index * 2 : (index + 1) * 2]
                * (after[:, index] - before[:, index])[:, None, None, None, None]
                for index in range(3)
            ],
            dim=2,
        )
        actual = scheduler.step(velocity, before, sample, next_timestep=after).prev_sample
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert scheduler.step_index is None

    def test_zero_velocity_and_zero_interval(self):
        scheduler = MagiEulerScheduler()
        scheduler.set_timesteps(4)
        sample = torch.randn(1, 4, 4, 4, 4)
        torch.testing.assert_close(scheduler.step(torch.zeros_like(sample), 0.2, sample, 0.9).prev_sample, sample)
        torch.testing.assert_close(scheduler.step(sample, 0.2, sample, 0.2).prev_sample, sample)

    def test_full_trajectory_and_reset(self):
        scheduler = MagiEulerScheduler()
        scheduler.set_timesteps(64)
        initial = torch.randn((1, 4, 4, 4, 6), generator=torch.Generator().manual_seed(0))
        sample = initial.clone()
        expected = initial.clone()
        for index, timestep in enumerate(scheduler.timesteps):
            velocity = sample.sin() * 0.2 + timestep
            expected_velocity = expected.sin() * 0.2 + timestep
            expected = expected + expected_velocity * (scheduler.timestep_schedule[index + 1] - timestep)
            sample = scheduler.step(velocity, timestep, sample).prev_sample
            torch.testing.assert_close(sample, expected, atol=0, rtol=0)
            assert scheduler.step_index == index + 1
        with pytest.raises(ValueError, match="complete"):
            scheduler.step(velocity, scheduler.timesteps[-1], sample)
        scheduler.set_timesteps(64)
        assert scheduler.step_index is None
        for timestep in scheduler.timesteps:
            initial = scheduler.step(initial.sin() * 0.2 + timestep, timestep, initial).prev_sample
        torch.testing.assert_close(initial, sample, atol=0, rtol=0)

    def test_out_of_order_step(self):
        scheduler = MagiEulerScheduler()
        scheduler.set_timesteps(4)
        sample = torch.zeros(1, 1, 1, 1, 1)
        with pytest.raises(ValueError, match="scalar from"):
            scheduler.step(sample, 0.123, sample)
        scheduler.step(sample, scheduler.timesteps[0], sample)
        with pytest.raises(ValueError, match="next sequential"):
            scheduler.step(sample, scheduler.timesteps[0], sample)
        assert scheduler.step_index == 1

    @pytest.mark.parametrize(
        "config", [{"shift": 0}, {"shift": float("nan")}, {"time_schedule": "unknown"}, {"shortcut_mode": "unknown"}]
    )
    def test_invalid_config(self, config):
        with pytest.raises(ValueError):
            MagiEulerScheduler(**config)

    @pytest.mark.parametrize("steps", [0, -1, 1.5, True])
    def test_invalid_step_count(self, steps):
        with pytest.raises(ValueError, match="positive integer"):
            MagiEulerScheduler().set_timesteps(steps)

    @pytest.mark.parametrize(
        "before,after", [(0.5, 0.4), (-0.1, 0.1), (0.9, 1.1), (float("nan"), 1), (0, float("inf"))]
    )
    def test_invalid_endpoints(self, before, after):
        scheduler = MagiEulerScheduler()
        scheduler.set_timesteps(4)
        with pytest.raises(ValueError, match="Timesteps"):
            scheduler.step(torch.ones(1), before, torch.ones(1), after)

    def test_invalid_shapes_and_uninitialized(self):
        scheduler = MagiEulerScheduler()
        sample = torch.zeros(2, 4, 4, 4, 4)
        with pytest.raises(ValueError, match="set_timesteps"):
            scheduler.step(sample, 0.0, sample)
        scheduler.set_timesteps(4)
        with pytest.raises(ValueError, match="same shape"):
            scheduler.step(sample[:1], 0.0, sample)
        with pytest.raises(ValueError, match="require next_timestep"):
            scheduler.step(sample, torch.zeros(2), sample)
        for shape in [(3,), (3, 2), (1, 1, 1), (0,)]:
            with pytest.raises(ValueError, match="Chunk times"):
                scheduler.step(sample, torch.zeros(shape), sample, torch.ones(shape))

    @pytest.mark.parametrize("time_schedule", ["sd3", "square", "piecewise", "linear"])
    def test_save_load(self, tmp_path, time_schedule):
        scheduler = MagiEulerScheduler(shift=5.0, time_schedule=time_schedule, shortcut_mode="16,16,8")
        scheduler.set_timesteps(12)
        scheduler.save_pretrained(tmp_path)
        restored = MagiEulerScheduler.from_pretrained(tmp_path)
        restored.set_timesteps(12)
        torch.testing.assert_close(restored.timestep_schedule, scheduler.timestep_schedule, atol=0, rtol=0)
        sample = torch.ones(1, 2, 4, 4, 4)
        torch.testing.assert_close(
            restored.step(sample, restored.timesteps[0], sample).prev_sample,
            scheduler.step(sample, scheduler.timesteps[0], sample).prev_sample,
            atol=0,
            rtol=0,
        )
