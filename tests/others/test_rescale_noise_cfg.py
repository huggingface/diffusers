# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

from diffusers.guiders.guider_utils import rescale_noise_cfg as guider_rescale_noise_cfg
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg


class RescaleNoiseCfgTests(unittest.TestCase):
    def test_rescale_noise_cfg_zero_std_cfg_is_finite(self):
        noise_cfg = torch.zeros(1, 4, 8, 8)
        noise_pred_text = torch.randn_like(noise_cfg)

        result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0)

        self.assertTrue(torch.isfinite(result).all())

    def test_guider_rescale_noise_cfg_zero_std_cfg_is_finite(self):
        noise_cfg = torch.zeros(1, 4, 8, 8)
        noise_pred_text = torch.randn_like(noise_cfg)

        result = guider_rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0)

        self.assertTrue(torch.isfinite(result).all())
