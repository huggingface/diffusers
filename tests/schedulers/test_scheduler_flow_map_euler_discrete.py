# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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

import unittest

import torch

from diffusers import FlowMapEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_map_euler_discrete import FlowMapEulerDiscreteSchedulerOutput


class FlowMapEulerDiscreteSchedulerTest(unittest.TestCase):
    """
    The flow-map scheduler has a non-standard ``step`` signature that takes both ``timestep`` and
    ``r_timestep`` (the target timestep), so it cannot use ``SchedulerCommonTest``. The tests below
    exercise the contract that the scheduler exposes to ``AnyFlowPipeline`` and ``AnyFlowFARPipeline``.
    """

    scheduler_class = FlowMapEulerDiscreteScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 1.0,
        }
        config.update(**kwargs)
        return config

    def test_instantiation_with_defaults(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        self.assertEqual(scheduler.config.num_train_timesteps, 1000)
        self.assertEqual(scheduler.config.shift, 1.0)

    def test_set_timesteps_endpoints(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        for nfe in [1, 2, 4, 8, 16]:
            scheduler.set_timesteps(num_inference_steps=nfe)
            # `timesteps` is N-length (mirrors FlowMatchEulerDiscreteScheduler); the final
            # r-endpoint sigma=0 lives in the internal `sigmas` buffer of length N+1.
            self.assertEqual(scheduler.timesteps.shape, (nfe,))
            self.assertEqual(scheduler.sigmas.shape, (nfe + 1,))
            self.assertAlmostEqual(scheduler.timesteps[0].item(), 1000.0, places=4)
            self.assertAlmostEqual(scheduler.sigmas[-1].item(), 0.0, places=4)

    def test_apply_shift_identity(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift=1.0))
        sigmas = torch.linspace(0.0, 1.0, 10)
        torch.testing.assert_close(scheduler.apply_shift(sigmas), sigmas)

    def test_apply_shift_monotonic(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift=5.0))
        sigmas = torch.linspace(0.01, 0.99, 16)
        shifted = scheduler.apply_shift(sigmas)
        # shift > 1 must monotonically map [0,1] to [0,1] and increase intermediate values
        self.assertTrue(torch.all(shifted >= 0))
        self.assertTrue(torch.all(shifted <= 1))
        self.assertTrue(torch.all(shifted[1:] - shifted[:-1] >= -1e-6))

    def test_step_shape_preserved(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(2, 16, 21, 30, 52)  # B, C, T, H, W (Wan2.1 latent shape)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0:1]
        r_timestep = scheduler.timesteps[1:2]

        output = scheduler.step(model_output, timestep, sample, r_timestep=r_timestep)
        self.assertIsInstance(output, FlowMapEulerDiscreteSchedulerOutput)
        prev_sample = output.prev_sample
        self.assertEqual(prev_sample.shape, sample.shape)
        self.assertEqual(prev_sample.dtype, model_output.dtype)

        # return_dict=False yields a tuple with the same prev_sample.
        (prev_sample_tuple,) = scheduler.step(model_output, timestep, sample, r_timestep=r_timestep, return_dict=False)
        torch.testing.assert_close(prev_sample_tuple, prev_sample)

    def test_step_zero_interval_is_identity(self):
        # When timestep == r_timestep the update collapses to the input sample.
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8, 8)
        model_output = torch.randn_like(sample)
        t = scheduler.timesteps[2:3]

        prev_sample = scheduler.step(model_output, t, sample, r_timestep=t).prev_sample
        torch.testing.assert_close(prev_sample, sample.to(model_output.dtype))

    def test_step_one_shot_sampling(self):
        # Flow-map promise: stepping straight from t=T to r=0 produces a clean sample in a single call.
        scheduler = self.scheduler_class(**self.get_default_config(shift=5.0))
        scheduler.set_timesteps(num_inference_steps=1)
        # `timesteps` is N=1 (just t=T); r=0 comes from the schedule's terminal sigma.
        # Pass r_timestep=None so step() resolves it via self.sigmas[-1] * num_train_timesteps.
        timesteps = scheduler.timesteps

        sample = torch.randn(1, 4, 4, 4)
        model_output = torch.randn_like(sample)

        prev_sample = scheduler.step(
            model_output,
            timesteps[0:1],
            sample,
        ).prev_sample
        self.assertEqual(prev_sample.shape, sample.shape)
        self.assertFalse(torch.allclose(prev_sample, sample))

    def test_scale_noise_endpoints(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        sample = torch.zeros(2, 4, 4, 4)
        noise = torch.ones_like(sample)
        # t=0 -> all sample, t=num_train_timesteps -> all noise.
        zero_t = torch.tensor([0.0])
        torch.testing.assert_close(scheduler.scale_noise(sample, zero_t, noise), sample)
        full_t = torch.tensor([float(scheduler.config.num_train_timesteps)])
        torch.testing.assert_close(scheduler.scale_noise(sample, full_t, noise), noise)
