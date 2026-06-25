# Copyright 2026 The Helios Team and The HuggingFace Team. All rights reserved.
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

from diffusers import HeliosDMDScheduler

from ..testing_utils import torch_device


class HeliosDMDSchedulerTest(unittest.TestCase):
    @unittest.skipUnless(torch_device == "mps", "test requires the 'mps' backend")
    def test_convert_flow_pred_to_x0_no_float64_on_mps(self):
        # mps has no float64; convert_flow_pred_to_x0 must not upcast to double there.
        scheduler = HeliosDMDScheduler()
        device = torch.device("mps")
        batch_size = 2
        flow_pred = torch.randn(batch_size, 3, 1, 4, 4, device=device)
        xt = torch.randn(batch_size, 3, 1, 4, 4, device=device)
        sigmas = torch.linspace(1.0, 0.0, 10, device=device)
        timesteps = torch.linspace(1000.0, 0.0, 10, device=device)
        timestep = timesteps[:batch_size]

        # Before the fix this raised "Cannot convert a MPS Tensor to float64 dtype".
        x0 = scheduler.convert_flow_pred_to_x0(flow_pred, xt, timestep, sigmas, timesteps)

        self.assertEqual(x0.device.type, "mps")
        self.assertEqual(x0.dtype, flow_pred.dtype)
        self.assertTrue(torch.isfinite(x0).all())
