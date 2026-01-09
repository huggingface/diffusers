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

from diffusers import LTXVideoTransformer3DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class LTXTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = LTXVideoTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        embedding_dim = 16
        sequence_length = 16

        hidden_states = torch.randn((batch_size, num_frames * height * width, num_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        encoder_attention_mask = torch.ones((batch_size, sequence_length)).bool().to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }

    @property
    def input_shape(self):
        return (512, 4)

    @property
    def output_shape(self):
        return (512, 4)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "num_layers": 1,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": 16,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"LTXVideoTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class LTXTransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = LTXVideoTransformer3DModel

    def prepare_init_args_and_inputs_for_common(self):
        return LTXTransformerTests().prepare_init_args_and_inputs_for_common()
