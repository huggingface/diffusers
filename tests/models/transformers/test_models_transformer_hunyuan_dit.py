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

from diffusers import HunyuanDiT2DModel

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class HunyuanDiTTests(ModelTesterMixin, unittest.TestCase):
    model_class = HunyuanDiT2DModel
    main_input_name = "hidden_states"

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        height = width = 8
        embedding_dim = 8
        sequence_length = 4
        sequence_length_t5 = 4

        hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        text_embedding_mask = torch.ones(size=(batch_size, sequence_length)).to(torch_device)
        encoder_hidden_states_t5 = torch.randn((batch_size, sequence_length_t5, embedding_dim)).to(torch_device)
        text_embedding_mask_t5 = torch.ones(size=(batch_size, sequence_length_t5)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,), dtype=encoder_hidden_states.dtype).to(torch_device)

        original_size = [1024, 1024]
        target_size = [16, 16]
        crops_coords_top_left = [0, 0]
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids, add_time_ids], dtype=encoder_hidden_states.dtype).to(torch_device)
        style = torch.zeros(size=(batch_size,), dtype=int).to(torch_device)
        image_rotary_emb = [
            torch.ones(size=(1, 8), dtype=encoder_hidden_states.dtype),
            torch.zeros(size=(1, 8), dtype=encoder_hidden_states.dtype),
        ]

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embedding_mask": text_embedding_mask,
            "encoder_hidden_states_t5": encoder_hidden_states_t5,
            "text_embedding_mask_t5": text_embedding_mask_t5,
            "timestep": timestep,
            "image_meta_size": add_time_ids,
            "style": style,
            "image_rotary_emb": image_rotary_emb,
        }

    @property
    def input_shape(self):
        return (4, 8, 8)

    @property
    def output_shape(self):
        return (8, 8, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 8,
            "patch_size": 2,
            "in_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "cross_attention_dim": 8,
            "cross_attention_dim_t5": 8,
            "pooled_projection_dim": 4,
            "hidden_size": 16,
            "text_len": 4,
            "text_len_t5": 4,
            "activation_fn": "gelu-approximate",
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        super().test_output(
            expected_output_shape=(self.dummy_input[self.main_input_name].shape[0],) + self.output_shape
        )

    @unittest.skip("HunyuanDIT use a custom processor HunyuanAttnProcessor2_0")
    def test_set_xformers_attn_processor_for_determinism(self):
        pass

    @unittest.skip("HunyuanDIT use a custom processor HunyuanAttnProcessor2_0")
    def test_set_attn_processor_for_determinism(self):
        pass
