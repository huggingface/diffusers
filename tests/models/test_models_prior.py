# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import gc
import inspect
import unittest

import torch
from parameterized import parameterized

from diffusers import PriorTransformer
from diffusers.utils import floats_tensor, slow, torch_all_close, torch_device
from diffusers.utils.testing_utils import enable_full_determinism

from .test_modeling_common import ModelTesterMixin


enable_full_determinism()


class PriorTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = PriorTransformer
    main_input_name = "hidden_states"

    @property
    def dummy_input(self):
        batch_size = 4
        embedding_dim = 8
        num_embeddings = 7

        hidden_states = floats_tensor((batch_size, embedding_dim)).to(torch_device)

        proj_embedding = floats_tensor((batch_size, embedding_dim)).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, num_embeddings, embedding_dim)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": 2,
            "proj_embedding": proj_embedding,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def get_dummy_seed_input(self, seed=0):
        torch.manual_seed(seed)
        batch_size = 4
        embedding_dim = 8
        num_embeddings = 7

        hidden_states = torch.randn((batch_size, embedding_dim)).to(torch_device)

        proj_embedding = torch.randn((batch_size, embedding_dim)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, num_embeddings, embedding_dim)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": 2,
            "proj_embedding": proj_embedding,
            "encoder_hidden_states": encoder_hidden_states,
        }

    @property
    def input_shape(self):
        return (4, 8)

    @property
    def output_shape(self):
        return (4, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 2,
            "embedding_dim": 8,
            "num_embeddings": 7,
            "additional_embeddings": 4,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = PriorTransformer.from_pretrained(
            "hf-internal-testing/prior-dummy", output_loading_info=True
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        hidden_states = model(**self.dummy_input)[0]

        assert hidden_states is not None, "Make sure output is not None"

    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = ["hidden_states", "timestep"]
        self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_output_pretrained(self):
        model = PriorTransformer.from_pretrained("hf-internal-testing/prior-dummy")
        model = model.to(torch_device)

        if hasattr(model, "set_default_attn_processor"):
            model.set_default_attn_processor()

        input = self.get_dummy_seed_input()

        with torch.no_grad():
            output = model(**input)[0]

        output_slice = output[0, :5].flatten().cpu()
        print(output_slice)

        # Since the VAE Gaussian prior's generator is seeded on the appropriate device,
        # the expected output slices are not the same for CPU and GPU.
        expected_output_slice = torch.tensor([-1.3436, -0.2870, 0.7538, 0.4368, -0.0239])
        self.assertTrue(torch_all_close(output_slice, expected_output_slice, rtol=1e-2))


@slow
class PriorTransformerIntegrationTests(unittest.TestCase):
    def get_dummy_seed_input(self, batch_size=1, embedding_dim=768, num_embeddings=77, seed=0):
        torch.manual_seed(seed)
        batch_size = batch_size
        embedding_dim = embedding_dim
        num_embeddings = num_embeddings

        hidden_states = torch.randn((batch_size, embedding_dim)).to(torch_device)

        proj_embedding = torch.randn((batch_size, embedding_dim)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, num_embeddings, embedding_dim)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": 2,
            "proj_embedding": proj_embedding,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.5861,  0.1283, -0.0931,  0.0882,  0.4476,  0.1329, -0.0498,  0.0640]],
            [37, [-0.4913,  0.0110, -0.0483,  0.0541,  0.4954, -0.0170,  0.0354,  0.1651]],
            # fmt: on
        ]
    )
    def test_kandinsky_prior(self, seed, expected_slice):
        model = PriorTransformer.from_pretrained("kandinsky-community/kandinsky-2-1-prior", subfolder="prior")
        model.to(torch_device)
        input = self.get_dummy_seed_input(seed=seed)

        with torch.no_grad():
            sample = model(**input)[0]

        assert list(sample.shape) == [1, 768]

        output_slice = sample[0, :8].flatten().cpu()
        print(output_slice)
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)
