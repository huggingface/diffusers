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

from diffusers import HunyuanVideoTransformer3DModel
from diffusers.utils.testing_utils import enable_full_determinism, torch_device

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class HunyuanVideoTransformer3DTests(ModelTesterMixin, unittest.TestCase):
    model_class = HunyuanVideoTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 4
        num_frames = 1
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        pooled_projection_dim = 8
        sequence_length = 12

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, text_encoder_embedding_dim)).to(torch_device)
        pooled_projections = torch.randn((batch_size, pooled_projection_dim)).to(torch_device)
        encoder_attention_mask = torch.ones((batch_size, sequence_length)).to(torch_device)
        guidance = torch.randint(0, 1000, size=(batch_size,)).to(torch_device, dtype=torch.float32)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "encoder_attention_mask": encoder_attention_mask,
            "guidance": guidance,
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
            "attention_head_dim": 10,
            "num_layers": 1,
            "num_single_layers": 1,
            "num_refiner_layers": 1,
            "patch_size": 1,
            "patch_size_t": 1,
            "guidance_embeds": True,
            "text_embed_dim": 16,
            "pooled_projection_dim": 8,
            "rope_axes_dim": (2, 4, 4),
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HunyuanVideoTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
