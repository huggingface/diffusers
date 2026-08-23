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

import unittest

import torch

from diffusers import FlowMatchEulerDiscreteScheduler


class FlowMatchEulerDiscreteSchedulerTest(unittest.TestCase):
    scheduler_class = FlowMatchEulerDiscreteScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 3.0,
        }
        config.update(**kwargs)
        return config

    def test_set_timesteps_multi_step_reaches_shift_terminal(self):
        # Sanity check that the multi-step schedule still stretches to shift_terminal;
        # guards against the single-step fix below accidentally disabling stretching generally.
        scheduler = self.scheduler_class(**self.get_default_config(shift_terminal=0.1))
        scheduler.set_timesteps(num_inference_steps=10)
        self.assertFalse(torch.isnan(scheduler.sigmas).any())
        self.assertAlmostEqual(scheduler.sigmas[-2].item(), 0.1, places=5)

    def test_set_timesteps_single_step_with_shift_terminal_is_finite(self):
        # With num_inference_steps=1 the lone sigma is both the first and last point of the
        # schedule, so stretch_shift_to_terminal's `one_minus_z[-1] / scale_factor` degenerates to
        # a 0/0 division and produced nan timesteps/sigmas. See gh-14411.
        scheduler = self.scheduler_class(**self.get_default_config(shift_terminal=0.1))
        scheduler.set_timesteps(num_inference_steps=1)
        self.assertFalse(torch.isnan(scheduler.sigmas).any())
        self.assertFalse(torch.isnan(scheduler.timesteps).any())

    def test_step_single_step_with_shift_terminal_runs(self):
        scheduler = self.scheduler_class(**self.get_default_config(shift_terminal=0.1))
        scheduler.set_timesteps(num_inference_steps=1)

        sample = torch.randn(1, 4, 4, 4)
        model_output = torch.randn_like(sample)

        prev_sample = scheduler.step(model_output, scheduler.timesteps[0], sample).prev_sample
        self.assertEqual(prev_sample.shape, sample.shape)
        self.assertFalse(torch.isnan(prev_sample).any())
