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

from diffusers import VQModel
from diffusers.testing_utils import floats_tensor, torch_device

from .test_modeling_common import ModelTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class VQModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = VQModel

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 3,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_from_pretrained_hub(self):
        model, loading_info = VQModel.from_pretrained("fusing/vqgan-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = VQModel.from_pretrained("fusing/vqgan-dummy")
        model.to(torch_device).eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        image = image.to(torch_device)
        with torch.no_grad():
            # Warmup pass when using mps (see #372)
            if torch_device == "mps":
                _ = model(image)
            output = model(image).sample

        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0153, -0.4044, -0.1880, -0.5161, -0.2418, -0.4072, -0.1612, -0.0633, -0.0143])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))
