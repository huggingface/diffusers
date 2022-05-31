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

import random
import tempfile
import unittest

import torch

from diffusers import UNetConfig, UNetModel


global_rng = random.Random()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


class ModelTesterMixin(unittest.TestCase):

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = torch.tensor([10])

        return (noise, time_step)

    def test_from_pretrained_save_pretrained(self):
        config = UNetConfig(dim=8, dim_mults=(1, 2), resnet_block_groups=2)
        model = UNetModel(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = UNetModel.from_pretrained(tmpdirname)

        dummy_input = self.dummy_input

        image = model(*dummy_input)
        new_image = new_model(*dummy_input)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub(self):
        model = UNetModel.from_pretrained("fusing/ddpm_dummy")

        image = model(*self.dummy_input)

        assert image is not None, "Make sure output is not None"
