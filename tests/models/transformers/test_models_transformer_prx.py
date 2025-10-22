# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class PRXTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = PRXTransformer2DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        return self.prepare_dummy_input()

    @property
    def input_shape(self):
        return (16, 16, 16)

    @property
    def output_shape(self):
        return (16, 16, 16)

    def prepare_dummy_input(self, height=16, width=16):
        batch_size = 1
        num_latent_channels = 16
        sequence_length = 16
        embedding_dim = 1792

        hidden_states = torch.randn((batch_size, num_latent_channels, height, width)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 16,
            "patch_size": 2,
            "context_in_dim": 1792,
            "hidden_size": 1792,
            "mlp_ratio": 3.5,
            "num_heads": 28,
            "depth": 4,  # Smaller depth for testing
            "axes_dim": [32, 32],
            "theta": 10_000,
        }
        inputs_dict = self.prepare_dummy_input()
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"PRXTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


if __name__ == "__main__":
    unittest.main()
