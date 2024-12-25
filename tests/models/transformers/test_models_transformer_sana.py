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

import pytest
import torch

from diffusers import SanaTransformer2DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class SanaTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = SanaTransformer2DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 4
        height = 32
        width = 32
        embedding_dim = 8
        sequence_length = 8

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
            "patch_size": 1,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 4,
            "num_attention_heads": 2,
            "num_cross_attention_heads": 2,
            "cross_attention_head_dim": 4,
            "cross_attention_dim": 8,
            "caption_channels": 8,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"SanaTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @pytest.mark.xfail(
        condition=torch.device(torch_device).type == "cuda",
        reason="Test currently fails.",
        strict=True,
    )
    def test_cpu_offload(self):
        return super().test_cpu_offload()

    @pytest.mark.xfail(
        condition=torch.device(torch_device).type == "cuda",
        reason="Test currently fails.",
        strict=True,
    )
    def test_disk_offload_with_safetensors(self):
        return super().test_disk_offload_with_safetensors()

    @pytest.mark.xfail(
        condition=torch.device(torch_device).type == "cuda",
        reason="Test currently fails.",
        strict=True,
    )
    def test_disk_offload_without_safetensors(self):
        return super().test_disk_offload_without_safetensors()
