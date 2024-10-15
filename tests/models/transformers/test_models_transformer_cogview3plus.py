# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from diffusers import CogView3PlusTransformer2DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class CogView3PlusTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = CogView3PlusTransformer2DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        height = 8
        width = 8
        embedding_dim = 8
        sequence_length = 8

        hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        original_size = torch.tensor([height * 8, width * 8]).unsqueeze(0).repeat(batch_size, 1).to(torch_device)
        target_size = torch.tensor([height * 8, width * 8]).unsqueeze(0).repeat(batch_size, 1).to(torch_device)
        crop_coords = torch.tensor([0, 0]).unsqueeze(0).repeat(batch_size, 1).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "original_size": original_size,
            "target_size": target_size,
            "crop_coords": crop_coords,
            "timestep": timestep,
        }

    @property
    def input_shape(self):
        return (1, 4, 8, 8)

    @property
    def output_shape(self):
        return (1, 4, 8, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "patch_size": 2,
            "in_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 4,
            "num_attention_heads": 2,
            "out_channels": 4,
            "text_embed_dim": 8,
            "time_embed_dim": 8,
            "condition_dim": 2,
            "pos_embed_max_size": 8,
            "sample_size": 8,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict
