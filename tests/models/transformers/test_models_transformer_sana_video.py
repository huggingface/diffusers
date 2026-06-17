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

from diffusers import SanaVideoTransformer3DModel

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class SanaVideoTransformer3DTests(ModelTesterMixin, unittest.TestCase):
    model_class = SanaVideoTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 16
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, text_encoder_embedding_dim)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }

    @property
    def input_shape(self):
        return (16, 2, 16, 16)

    @property
    def output_shape(self):
        return (16, 2, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 16,
            "out_channels": 16,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "num_layers": 2,
            "num_cross_attention_heads": 2,
            "cross_attention_head_dim": 12,
            "cross_attention_dim": 24,
            "caption_channels": 16,
            "mlp_ratio": 2.5,
            "dropout": 0.0,
            "attention_bias": False,
            "sample_size": 8,
            "patch_size": (1, 2, 2),
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"SanaVideoTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class SanaVideoTransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = SanaVideoTransformer3DModel

    def prepare_init_args_and_inputs_for_common(self):
        return SanaVideoTransformer3DTests().prepare_init_args_and_inputs_for_common()
