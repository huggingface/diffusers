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

from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchLCMScheduler, UniPCMultistepScheduler


class ShiftTerminalSingleStepTest(unittest.TestCase):
    """
    Regression test for https://github.com/huggingface/diffusers/issues/14411.

    `stretch_shift_to_terminal()` rescales sigmas by `one_minus_z[-1] / (1 - shift_terminal)`. With
    `num_inference_steps=1` the only sigma is 1.0, so `one_minus_z[-1]` is 0 and the rescale divides by
    zero, producing a NaN sigma. Schedulers that support `shift_terminal` must skip the stretch when
    there is only a single step instead of stretching into NaN.
    """

    def test_flow_match_euler_discrete_single_step_no_nan(self):
        scheduler = FlowMatchEulerDiscreteScheduler(shift_terminal=0.1)
        scheduler.set_timesteps(num_inference_steps=1)
        self.assertFalse(torch.isnan(scheduler.sigmas).any())

    def test_flow_match_lcm_single_step_no_nan(self):
        scheduler = FlowMatchLCMScheduler(shift_terminal=0.1)
        scheduler.set_timesteps(num_inference_steps=1)
        self.assertFalse(torch.isnan(scheduler.sigmas).any())

    def test_unipc_flow_sigmas_single_step_no_nan(self):
        scheduler = UniPCMultistepScheduler(use_flow_sigmas=True, shift_terminal=0.1)
        scheduler.set_timesteps(num_inference_steps=1)
        self.assertFalse(torch.isnan(scheduler.sigmas).any())

    def test_flow_match_euler_discrete_custom_schedule_ending_at_one_no_nan(self):
        # A multi-step custom schedule whose last sigma is already 1.0 hits the same
        # `one_minus_z[-1] == 0` division-by-zero case as the single-step schedule.
        scheduler = FlowMatchEulerDiscreteScheduler(shift_terminal=0.1)
        scheduler.set_timesteps(sigmas=[0.9, 1.0])
        self.assertFalse(torch.isnan(scheduler.sigmas).any())
        self.assertFalse(torch.isinf(scheduler.sigmas).any())
