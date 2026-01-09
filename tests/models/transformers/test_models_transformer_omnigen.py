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

from diffusers import OmniGenTransformer2DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class OmniGenTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = OmniGenTransformer2DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True
    model_split_percents = [0.1, 0.1, 0.1]

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        height = 8
        width = 8
        sequence_length = 24

        hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
        timestep = torch.rand(size=(batch_size,), dtype=hidden_states.dtype).to(torch_device)
        input_ids = torch.randint(0, 10, (batch_size, sequence_length)).to(torch_device)
        input_img_latents = [torch.randn((1, num_channels, height, width)).to(torch_device)]
        input_image_sizes = {0: [[0, 0 + height * width // 2 // 2]]}

        attn_seq_length = sequence_length + 1 + height * width // 2 // 2
        attention_mask = torch.ones((batch_size, attn_seq_length, attn_seq_length)).to(torch_device)
        position_ids = torch.LongTensor([list(range(attn_seq_length))] * batch_size).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "input_ids": input_ids,
            "input_img_latents": input_img_latents,
            "input_image_sizes": input_image_sizes,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    @property
    def input_shape(self):
        return (4, 8, 8)

    @property
    def output_shape(self):
        return (4, 8, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 32,
            "num_layers": 20,
            "pad_token_id": 0,
            "vocab_size": 1000,
            "in_channels": 4,
            "time_step_dim": 4,
            "rope_scaling": {"long_factor": list(range(1, 3)), "short_factor": list(range(1, 3))},
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"OmniGenTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
