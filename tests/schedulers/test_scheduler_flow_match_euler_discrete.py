# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import time
import unittest

import torch

from diffusers import FlowMatchEulerDiscreteScheduler


def _legacy_index_for_timestep(scheduler, timestep, schedule_timesteps):
    """Reference implementation using per-element nonzero()."""
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=schedule_timesteps.device, dtype=schedule_timesteps.dtype)
    elif timestep.ndim == 0:
        timestep = timestep.reshape(1)

    indices = []
    for t in timestep:
        matches = (schedule_timesteps == t).nonzero()
        pos = 1 if len(matches) > 1 else 0
        indices.append(matches[pos].item())
    return indices[0] if len(indices) == 1 else torch.tensor(indices, device=schedule_timesteps.device)


class FlowMatchEulerDiscreteSchedulerTest(unittest.TestCase):
    scheduler_class = FlowMatchEulerDiscreteScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 1.0,
        }
        config.update(**kwargs)
        return config

    def _assert_index_parity(self, scheduler, schedule_timesteps, timesteps):
        for t in timesteps:
            expected = _legacy_index_for_timestep(scheduler, t, schedule_timesteps)
            actual = scheduler.index_for_timestep(t, schedule_timesteps)
            self.assertEqual(actual, expected)

        batch = torch.stack([timesteps[0], timesteps[-1], timesteps[len(timesteps) // 2]])
        expected_batch = torch.tensor(
            [_legacy_index_for_timestep(scheduler, t, schedule_timesteps) for t in batch],
            device=schedule_timesteps.device,
        )
        actual_batch = scheduler.index_for_timestep(batch, schedule_timesteps)
        torch.testing.assert_close(actual_batch, expected_batch)

    def test_index_for_timestep_even_shift(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift=1.0))
        scheduler.set_timesteps(num_inference_steps=10)
        self._assert_index_parity(scheduler, scheduler.timesteps, scheduler.timesteps)

    def test_index_for_timestep_non_uniform_shift(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift=3.0))
        scheduler.set_timesteps(num_inference_steps=20)
        self._assert_index_parity(scheduler, scheduler.timesteps, scheduler.timesteps)

    def test_scale_noise_batch_matches_legacy(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift=3.0))
        scheduler.set_timesteps(num_inference_steps=16)

        sample = torch.randn(8, 4, 8, 8)
        noise = torch.randn_like(sample)
        timesteps = scheduler.timesteps[:8]

        out = scheduler.scale_noise(sample, timesteps, noise)

        legacy_indices = [_legacy_index_for_timestep(scheduler, t, scheduler.timesteps) for t in timesteps]
        sigmas = scheduler.sigmas.to(sample.device, dtype=sample.dtype)
        sigma = sigmas[torch.tensor(legacy_indices)].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)
        expected = sigma * noise + (1.0 - sigma) * sample

        torch.testing.assert_close(out, expected)

    def test_scale_noise_training_batch_speedup(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift=3.0))
        scheduler.set_timesteps(num_inference_steps=50)

        sample = torch.randn(64, 16, 32, 32)
        noise = torch.randn_like(sample)
        timesteps = scheduler.timesteps.repeat(64 // scheduler.timesteps.shape[0] + 1)[:64]

        warmup = 5
        repeats = 50
        for _ in range(warmup):
            scheduler.scale_noise(sample, timesteps, noise)

        start = time.perf_counter()
        for _ in range(repeats):
            scheduler.scale_noise(sample, timesteps, noise)
        optimized = time.perf_counter() - start

        schedule_timesteps = scheduler.timesteps.to(sample.device)

        def legacy_scale_noise():
            step_indices = [_legacy_index_for_timestep(scheduler, t, schedule_timesteps) for t in timesteps]
            sigmas = scheduler.sigmas.to(sample.device, dtype=sample.dtype)
            sigma = sigmas[torch.tensor(step_indices)].flatten()
            while len(sigma.shape) < len(sample.shape):
                sigma = sigma.unsqueeze(-1)
            return sigma * noise + (1.0 - sigma) * sample

        for _ in range(warmup):
            legacy_scale_noise()

        start = time.perf_counter()
        for _ in range(repeats):
            legacy_scale_noise()
        legacy = time.perf_counter() - start

        self.assertLess(
            optimized, legacy, msg=f"expected speedup, got optimized={optimized:.4f}s legacy={legacy:.4f}s"
        )
