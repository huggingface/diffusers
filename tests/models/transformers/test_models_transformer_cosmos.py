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

from diffusers import CosmosTransformer3DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class CosmosTransformer3DModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = CosmosTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 4
        num_frames = 1
        height = 16
        width = 16
        text_embed_dim = 16
        sequence_length = 12
        fps = 30

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, text_embed_dim)).to(torch_device)
        attention_mask = torch.ones((batch_size, sequence_length)).to(torch_device)
        padding_mask = torch.zeros(batch_size, 1, height, width).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
            "fps": fps,
            "padding_mask": padding_mask,
        }

    @property
    def input_shape(self):
        return (4, 1, 16, 16)

    @property
    def output_shape(self):
        return (4, 1, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "num_layers": 2,
            "mlp_ratio": 2,
            "text_embed_dim": 16,
            "adaln_lora_dim": 4,
            "max_size": (4, 32, 32),
            "patch_size": (1, 2, 2),
            "rope_scale": (2.0, 1.0, 1.0),
            "concat_padding_mask": True,
            "extra_pos_embed_type": "learnable",
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class CosmosTransformer3DModelVideoToWorldTests(ModelTesterMixin, unittest.TestCase):
    model_class = CosmosTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 4
        num_frames = 1
        height = 16
        width = 16
        text_embed_dim = 16
        sequence_length = 12
        fps = 30

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, text_embed_dim)).to(torch_device)
        attention_mask = torch.ones((batch_size, sequence_length)).to(torch_device)
        condition_mask = torch.ones(batch_size, 1, num_frames, height, width).to(torch_device)
        padding_mask = torch.zeros(batch_size, 1, height, width).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
            "fps": fps,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
        }

    @property
    def input_shape(self):
        return (4, 1, 16, 16)

    @property
    def output_shape(self):
        return (4, 1, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 4 + 1,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "num_layers": 2,
            "mlp_ratio": 2,
            "text_embed_dim": 16,
            "adaln_lora_dim": 4,
            "max_size": (4, 32, 32),
            "patch_size": (1, 2, 2),
            "rope_scale": (2.0, 1.0, 1.0),
            "concat_padding_mask": True,
            "extra_pos_embed_type": "learnable",
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
