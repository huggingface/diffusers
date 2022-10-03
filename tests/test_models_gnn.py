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
import tempfile

import torch

from diffusers.testing_utils import floats_tensor, slow, torch_device

from .test_modeling_common import ModelTesterMixin
from diffusers.utils import is_torch_geometric_available

if is_torch_geometric_available():
    from diffusers import MoleculeGNN
else:
    from diffusers.utils.dummy_torch_geometric_objects import *


torch.backends.cuda.matmul.allow_tf32 = False

class MoleculeGNNTests(ModelTesterMixin, unittest.TestCase):
    model_class = MoleculeGNN

    @property
    def dummy_input(self):
        batch_size = 2
        time_step = 10

        class GeoDiffData:
            # constants corresponding to a molecule
            num_nodes = 6
            num_edges = 10
            num_graphs = 1

            # sampling
            torch.Generator(device=torch_device)
            torch.manual_seed(3)

            # molecule components / properties
            atom_type = torch.randint(0, 6, (num_nodes * batch_size,)).to(torch_device)
            edge_index = torch.randint(
                0,
                num_edges,
                (
                    2,
                    num_edges * batch_size,
                ),
            ).to(torch_device)
            edge_type = torch.randint(0, 5, (num_edges * batch_size,)).to(torch_device)
            pos = 0.001 * torch.randn(num_nodes * batch_size, 3).to(torch_device)
            batch = torch.tensor([*range(batch_size)]).repeat_interleave(num_nodes)
            nx = batch_size

        torch.manual_seed(0)
        noise = GeoDiffData()

        return {"sample": noise, "timestep": time_step, "sigma": 1.0}

    @property
    def output_shape(self):
        # subset of shapes for dummy input
        class GeoDiffShapes:
            shape_0 = torch.Size([1305, 1])
            shape_1 = torch.Size([92, 1])

        return GeoDiffShapes()

    # training not implemented for this model yet
    def test_training(self):
        pass

    def test_ema_training(self):
        pass

    def test_determinism(self):
        # TODO
        pass

    def test_output(self):
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
            shapes = self.output_shapes()
            self.assertEqual(output[0].shape, shapes.shape_0, "Input and output shapes do not match")
            self.assertEqual(output[1].shape, shapes.shape_1, "Input and output shapes do not match")

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "hidden_dim": 128,
            "num_convs": 6,
            "num_convs_local": 4,
            "cutoff": 10.0,
            "mlp_act": "relu",
            "edge_order": 3,
            "edge_encoder": "mlp",
            "smooth_conv": True,
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = MoleculeGNN.from_pretrained("fusing/gfn-molecule-gen-drugs", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = MoleculeGNN.from_pretrained("fusing/gfn-molecule-gen-drugs")
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        input = self.dummy_input
        sample, time_step, sigma = input["sample"], input["timestep"], input["sigma"]
        with torch.no_grad():
            output = model(sample, time_step, sigma=sigma)["sample"]

        output_slice = output[:3][:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([ -3.7335,  -7.4622, -29.5600,  16.9646, -11.2205, -32.5315,   1.2303,
          4.2985,   8.8828])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-3))

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

        self.assertEqual(output_1[0].shape, output_2[0].shape)
        self.assertEqual(output_1[1].shape, output_2[1].shape)