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

from diffusers import LatteTransformer3DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class LatteTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = LatteTransformer3DModel
    main_input_name = "hidden_states"

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        num_frames = 1
        height = width = 8
        embedding_dim = 8
        sequence_length = 8

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "enable_temporal_attentions": True,
        }

    @property
    def input_shape(self):
        return (4, 1, 8, 8)

    @property
    def output_shape(self):
        return (8, 1, 8, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 8,
            "num_layers": 1,
            "patch_size": 2,
            "attention_head_dim": 4,
            "num_attention_heads": 2,
            "caption_channels": 8,
            "in_channels": 4,
            "cross_attention_dim": 8,
            "out_channels": 8,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "norm_type": "ada_norm_single",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        super().test_output(
            expected_output_shape=(self.dummy_input[self.main_input_name].shape[0],) + self.output_shape
        )
