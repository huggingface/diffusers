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
            "weight_type": "gaussian",
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
            self.assertEqual(scheduler.timesteps.shape, (nfe + 1,))
            self.assertAlmostEqual(scheduler.timesteps[0].item(), 1000.0, places=4)
            self.assertAlmostEqual(scheduler.timesteps[-1].item(), 0.0, places=4)

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

        prev_sample = scheduler.step(model_output, sample, timestep=timestep, r_timestep=r_timestep)
        self.assertEqual(prev_sample.shape, sample.shape)
        self.assertEqual(prev_sample.dtype, model_output.dtype)

    def test_step_zero_interval_is_identity(self):
        # When timestep == r_timestep the update collapses to the input sample.
        scheduler = self.scheduler_class(**self.get_default_config())
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.randn(1, 4, 8, 8, 8)
        model_output = torch.randn_like(sample)
        t = scheduler.timesteps[2:3]

        prev_sample = scheduler.step(model_output, sample, timestep=t, r_timestep=t)
        torch.testing.assert_close(prev_sample, sample.to(model_output.dtype))

    def test_step_one_shot_sampling(self):
        # Flow-map promise: stepping straight from t=T to r=0 produces a clean sample in a single call.
        scheduler = self.scheduler_class(**self.get_default_config(shift=5.0))
        scheduler.set_timesteps(num_inference_steps=1)
        timesteps = scheduler.timesteps

        sample = torch.randn(1, 4, 4, 4)
        model_output = torch.randn_like(sample)

        prev_sample = scheduler.step(
            model_output,
            sample,
            timestep=timesteps[0:1],
            r_timestep=timesteps[1:2],
        )
        self.assertEqual(prev_sample.shape, sample.shape)
        self.assertFalse(torch.allclose(prev_sample, sample))

    def test_train_weight_gaussian_shape(self):
        scheduler = self.scheduler_class(**self.get_default_config(weight_type="gaussian"))
        weights = scheduler.linear_timesteps_weights
        self.assertEqual(weights.shape, (scheduler.config.num_train_timesteps + 1,))
        self.assertTrue(torch.all(weights >= 0))

    def test_train_weight_beta08_shape(self):
        scheduler = self.scheduler_class(**self.get_default_config(weight_type="beta08"))
        weights = scheduler.linear_timesteps_weights
        self.assertEqual(weights.shape, (scheduler.config.num_train_timesteps + 1,))
        self.assertTrue(torch.all(weights >= 0))

    def test_train_weight_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.scheduler_class(**self.get_default_config(weight_type="not-a-real-type"))

    def test_get_train_weight_returns_per_timestep(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        timesteps = torch.tensor([0.0, 250.0, 500.0, 750.0, 1000.0])
        weights = scheduler.get_train_weight(timesteps)
        self.assertEqual(weights.shape, timesteps.shape)
        self.assertTrue(torch.all(weights >= 0))

    def test_scale_noise_endpoints(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        sample = torch.zeros(2, 4, 4, 4)
        noise = torch.ones_like(sample)
        # t=0 -> all sample, t=num_train_timesteps -> all noise.
        zero_t = torch.tensor([0.0])
        torch.testing.assert_close(scheduler.scale_noise(sample, zero_t, noise), sample)
        full_t = torch.tensor([float(scheduler.config.num_train_timesteps)])
        torch.testing.assert_close(scheduler.scale_noise(sample, full_t, noise), noise)
