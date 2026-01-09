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

from diffusers import HiDreamImageTransformer2DModel

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class HiDreamTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = HiDreamImageTransformer2DModel
    main_input_name = "hidden_states"
    model_split_percents = [0.8, 0.8, 0.9]

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        height = width = 32
        embedding_dim_t5, embedding_dim_llama, embedding_dim_pooled = 8, 4, 8
        sequence_length = 8

        hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
        encoder_hidden_states_t5 = torch.randn((batch_size, sequence_length, embedding_dim_t5)).to(torch_device)
        encoder_hidden_states_llama3 = torch.randn((batch_size, batch_size, sequence_length, embedding_dim_llama)).to(
            torch_device
        )
        pooled_embeds = torch.randn((batch_size, embedding_dim_pooled)).to(torch_device)
        timesteps = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states_t5": encoder_hidden_states_t5,
            "encoder_hidden_states_llama3": encoder_hidden_states_llama3,
            "pooled_embeds": pooled_embeds,
            "timesteps": timesteps,
        }

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "patch_size": 2,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "caption_channels": [8, 4],
            "text_emb_dim": 8,
            "num_routed_experts": 2,
            "num_activated_experts": 2,
            "axes_dims_rope": (4, 2, 2),
            "max_resolution": (32, 32),
            "llama_layers": (0, 1),
            "force_inference_output": True,  # TODO: as we don't implement MoE loss in training tests.
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skip("HiDreamImageTransformer2DModel uses a dedicated attention processor. This test doesn't apply")
    def test_set_attn_processor_for_determinism(self):
        pass

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HiDreamImageTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
