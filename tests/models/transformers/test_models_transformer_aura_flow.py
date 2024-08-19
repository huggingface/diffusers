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

from diffusers import AuraFlowTransformer2DModel
from diffusers.utils.testing_utils import enable_full_determinism, torch_device

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class AuraFlowTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = AuraFlowTransformer2DModel
    main_input_name = "hidden_states"
    # We override the items here because the transformer under consideration is small.
    model_split_percents = [0.7, 0.6, 0.6]

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        height = width = embedding_dim = 32
        sequence_length = 256

        hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "num_mmdit_layers": 1,
            "num_single_dit_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "caption_projection_dim": 32,
            "joint_attention_dim": 32,
            "out_channels": 4,
            "pos_embed_max_size": 256,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skip("AuraFlowTransformer2DModel uses its own dedicated attention processor. This test does not apply")
    def test_set_attn_processor_for_determinism(self):
        pass
