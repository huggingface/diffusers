# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

from diffusers import JiTTransformer2DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class JiTTransformer2DModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = JiTTransformer2DModel
    main_input_name = "hidden_states"

    @property
    def dummy_input(self):
        batch_size = 4
        in_channels = 3
        sample_size = 32
        num_classes = 10
        # JiT expects floats for hidden_states
        hidden_states = floats_tensor((batch_size, in_channels, sample_size, sample_size)).to(torch_device)
        # timestep is LongTensor for embeddings
        timesteps = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        class_labels = torch.randint(0, num_classes, size=(batch_size,)).to(torch_device)

        return {"hidden_states": hidden_states, "timestep": timesteps, "class_labels": class_labels}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 32,
            "patch_size": 4,
            "in_channels": 3,
            "hidden_size": 32,
            "num_layers": 2,
            "num_attention_heads": 4,
            "mlp_ratio": 4.0,
            "num_classes": 10,
            "bottleneck_dim": 16,
            "in_context_len": 4,
            "in_context_start": 1,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        super().test_output(
            expected_output_shape=(self.dummy_input[self.main_input_name].shape[0],) + self.output_shape
        )

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"JiTTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_effective_gradient_checkpointing(self):
        super().test_effective_gradient_checkpointing(loss_tolerance=1e-4)
