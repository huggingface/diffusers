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

import inspect
import tempfile

import numpy as np
import torch

from diffusers.testing_utils import torch_device
from diffusers.training_utils import EMAModel


class ModelTesterMixin:
    def test_from_pretrained_save_pretrained(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)

        with torch.no_grad():
            image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image["sample"]

            new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image["sample"]

        max_diff = (image - new_image).abs().sum().item()
        self.assertLessEqual(max_diff, 5e-5, "Models give different forward passes")

    def test_determinism(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            first = model(**inputs_dict)
            if isinstance(first, dict):
                first = first["sample"]

            second = model(**inputs_dict)
            if isinstance(second, dict):
                second = second["sample"]

        out_1 = first.cpu().numpy()
        out_2 = second.cpu().numpy()
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-5)

    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output["sample"]

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = ["sample", "timestep"]
        self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model_from_config(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        # test if the model can be loaded from the config
        # and has all the expected shape
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_config(tmpdirname)
            new_model = self.model_class.from_config(tmpdirname)
            new_model.to(torch_device)
            new_model.eval()

        # check if all paramters shape are the same
        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            self.assertEqual(param_1.shape, param_2.shape)

        with torch.no_grad():
            output_1 = model(**inputs_dict)

            if isinstance(output_1, dict):
                output_1 = output_1["sample"]

            output_2 = new_model(**inputs_dict)

            if isinstance(output_2, dict):
                output_2 = output_2["sample"]

        self.assertEqual(output_1.shape, output_2.shape)

    def test_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output["sample"]

        noise = torch.randn((inputs_dict["sample"].shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()

    def test_ema_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        ema_model = EMAModel(model, device=torch_device)

        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output["sample"]

        noise = torch.randn((inputs_dict["sample"].shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
        ema_model.step(model)
