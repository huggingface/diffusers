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

from diffusers import ConsisIDTransformer3DModel

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class ConsisIDTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = ConsisIDTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        num_frames = 1
        height = 8
        width = 8
        embedding_dim = 8
        sequence_length = 8

        hidden_states = torch.randn((batch_size, num_frames, num_channels, height, width)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        id_vit_hidden = [torch.ones([batch_size, 2, 2]).to(torch_device)] * 1
        id_cond = torch.ones(batch_size, 2).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "id_vit_hidden": id_vit_hidden,
            "id_cond": id_cond,
        }

    @property
    def input_shape(self):
        return (1, 4, 8, 8)

    @property
    def output_shape(self):
        return (1, 4, 8, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "time_embed_dim": 2,
            "text_embed_dim": 8,
            "num_layers": 1,
            "sample_width": 8,
            "sample_height": 8,
            "sample_frames": 8,
            "patch_size": 2,
            "temporal_compression_ratio": 4,
            "max_text_seq_length": 8,
            "cross_attn_interval": 1,
            "is_kps": False,
            "is_train_face": True,
            "cross_attn_dim_head": 1,
            "cross_attn_num_heads": 1,
            "LFE_id_dim": 2,
            "LFE_vit_dim": 2,
            "LFE_depth": 5,
            "LFE_dim_head": 8,
            "LFE_num_heads": 2,
            "LFE_num_id_token": 1,
            "LFE_num_querie": 1,
            "LFE_output_dim": 10,
            "LFE_ff_mult": 1,
            "LFE_num_scale": 1,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"ConsisIDTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
