# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from diffusers import UNet1DModel
from diffusers.utils import slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class UnetModel1DTests(unittest.TestCase):
    @slow
    def test_unet_1d_maestro(self):
        model_id = "harmonai/maestro-150k"
        model = UNet1DModel.from_pretrained(model_id, subfolder="unet", device_map="auto")
        model.to(torch_device)

        sample_size = 65536
        noise = torch.sin(torch.arange(sample_size)[None, None, :].repeat(1, 2, 1)).to(torch_device)
        timestep = torch.tensor([1]).to(torch_device)

        with torch.no_grad():
            output = model(noise, timestep).sample

        output_sum = output.abs().sum()
        output_max = output.abs().max()

        assert (output_sum - 224.0896).abs() < 4e-2
        assert (output_max - 0.0607).abs() < 4e-4
