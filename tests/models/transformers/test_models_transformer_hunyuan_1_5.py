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

from diffusers import HunyuanVideo15Transformer3DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class HunyuanVideo15Transformer3DTests(ModelTesterMixin, unittest.TestCase):
    model_class = HunyuanVideo15Transformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True
    model_split_percents = [0.99, 0.99, 0.99]

    text_embed_dim = 16
    text_embed_2_dim = 8
    image_embed_dim = 12

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 4
        num_frames = 1
        height = 8
        width = 8
        sequence_length = 6
        sequence_length_2 = 4
        image_sequence_length = 3

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, self.text_embed_dim), device=torch_device)
        encoder_hidden_states_2 = torch.randn(
            (batch_size, sequence_length_2, self.text_embed_2_dim), device=torch_device
        )
        encoder_attention_mask = torch.ones((batch_size, sequence_length), device=torch_device)
        encoder_attention_mask_2 = torch.ones((batch_size, sequence_length_2), device=torch_device)
        # All zeros for inducing T2V path in the model.
        image_embeds = torch.zeros((batch_size, image_sequence_length, self.image_embed_dim), device=torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_hidden_states_2": encoder_hidden_states_2,
            "encoder_attention_mask_2": encoder_attention_mask_2,
            "image_embeds": image_embeds,
        }

    @property
    def input_shape(self):
        return (4, 1, 8, 8)

    @property
    def output_shape(self):
        return (4, 1, 8, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 2,
            "num_refiner_layers": 1,
            "mlp_ratio": 2.0,
            "patch_size": 1,
            "patch_size_t": 1,
            "text_embed_dim": self.text_embed_dim,
            "text_embed_2_dim": self.text_embed_2_dim,
            "image_embed_dim": self.image_embed_dim,
            "rope_axes_dim": (2, 2, 4),
            "target_size": 16,
            "task_type": "t2v",
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HunyuanVideo15Transformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
