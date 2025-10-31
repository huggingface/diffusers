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

from diffusers import BriaFiboTransformer2DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class BriaFiboTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = BriaFiboTransformer2DModel
    main_input_name = "hidden_states"
    # We override the items here because the transformer under consideration is small.
    model_split_percents = [0.8, 0.7, 0.7]

    # Skip setting testing with default: AttnProcessor
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_latent_channels = 48
        num_image_channels = 3
        height = width = 16
        sequence_length = 32
        embedding_dim = 64

        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        text_ids = torch.randn((sequence_length, num_image_channels)).to(torch_device)
        image_ids = torch.randn((height * width, num_image_channels)).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "text_encoder_layers": [encoder_hidden_states[:, :, :32], encoder_hidden_states[:, :, :32]],
        }

    @property
    def input_shape(self):
        return (16, 16)

    @property
    def output_shape(self):
        return (256, 48)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "patch_size": 1,
            "in_channels": 48,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "joint_attention_dim": 64,
            "text_encoder_dim": 32,
            "pooled_projection_dim": None,
            "axes_dims_rope": [0, 4, 4],
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"BriaFiboTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
