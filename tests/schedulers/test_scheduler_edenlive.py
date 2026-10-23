# Copyright 2026 The VAST-AI Team. All rights reserved.
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

import tempfile
import unittest

import torch

from diffusers import EdenLiveFlowMatchEulerDiscreteScheduler


class EdenLiveFlowMatchEulerDiscreteSchedulerTests(unittest.TestCase):
    def test_shifted_schedule(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(shift=3.0)
        scheduler.set_timesteps(30)
        self.assertEqual(scheduler.sigmas.dtype, torch.float32)
        self.assertEqual(len(scheduler.timesteps), 30)
        torch.testing.assert_close(
            scheduler.sigmas[[0, 1, 15, 29, 30]],
            torch.tensor([0.9999989867210388, 0.9886479377746582, 0.7503749132156372, 0.09629422426223755, 0.0]),
            rtol=0,
            atol=0,
        )

    def test_explicit_timesteps_are_not_shifted(self):
        for points in ([1000, 937.5, 833.3333333333, 625], [1000, 900, 750, 500]):
            scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(shift=9.0, default_timesteps=points)
            scheduler.set_timesteps(4)
            torch.testing.assert_close(scheduler.sigmas[:-1], torch.tensor(points) / 1000, rtol=0, atol=0)
            scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0])
            torch.testing.assert_close(scheduler.sigmas, torch.tensor([1.0, 0.5, 0.0]), rtol=0, atol=0)
            self.assertEqual(len(scheduler.timesteps), 2)

    def test_stochastic_dtype_and_variance_noise(self):
        for dtype in (torch.float32, torch.bfloat16):
            scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
            scheduler.set_timesteps(timesteps=[1000, 900])
            generator = torch.Generator().manual_seed(9)
            sample = torch.randn((2, 3, 4), generator=generator).to(dtype)
            prediction = torch.randn(sample.shape, generator=generator).to(dtype)
            noise_generator = torch.Generator().manual_seed(12)
            noise = torch.randn(sample.shape, generator=noise_generator, dtype=dtype)
            explicit = scheduler.step(prediction, scheduler.timesteps[0], sample, variance_noise=noise).prev_sample
            scheduler.set_timesteps(timesteps=[1000, 900])
            generated = scheduler.step(
                prediction, scheduler.timesteps[0], sample, generator=torch.Generator().manual_seed(12)
            ).prev_sample
            self.assertEqual(explicit.dtype, dtype)
            torch.testing.assert_close(explicit, generated, rtol=0, atol=0)

    def test_final_step_does_not_advance_generator(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
        scheduler.set_timesteps(timesteps=[500])
        generator = torch.Generator().manual_seed(7)
        state = generator.get_state().clone()
        sample = torch.ones((1, 2, 3), dtype=torch.bfloat16)
        result = scheduler.step(sample, scheduler.timesteps[0], sample, generator=generator).prev_sample
        self.assertTrue(torch.equal(state, generator.get_state()))
        self.assertTrue(torch.equal(result, sample * 0.5))

    def test_chunk_reset_and_serialization(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(
            stochastic_sampling=True, default_timesteps=[1000, 900, 750, 500]
        )
        sample = torch.ones((1, 2, 3))
        scheduler.set_timesteps(4)
        result = scheduler.step(sample, scheduler.timesteps[0], sample, variance_noise=sample).prev_sample
        self.assertEqual(scheduler.step_index, 1)
        scheduler.set_timesteps(4)
        self.assertIsNone(scheduler.step_index)
        repeated = scheduler.step(sample, scheduler.timesteps[0], sample, variance_noise=sample).prev_sample
        torch.testing.assert_close(result, repeated, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            scheduler.save_pretrained(directory)
            restored = EdenLiveFlowMatchEulerDiscreteScheduler.from_pretrained(directory)
        restored.set_timesteps(4)
        torch.testing.assert_close(restored.sigmas, scheduler.sigmas, rtol=0, atol=0)
        self.assertIsNone(restored.step_index)
        self.assertTrue(restored.config.stochastic_sampling)

    def test_invalid_schedule(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(default_timesteps=[1000, 500])
        for kwargs in (
            {"timesteps": [1000], "sigmas": [1]},
            {"timesteps": [500, 1000]},
            {"timesteps": [1000, 1000, 500]},
            {"timesteps": [float("nan")]},
            {"sigmas": []},
            {"sigmas": [1.1, 0.5]},
            {"num_inference_steps": 4},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                scheduler.set_timesteps(**kwargs)
