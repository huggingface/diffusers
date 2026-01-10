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

import pytest
import torch

from diffusers import QwenImageTransformer2DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class QwenImageTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = QwenImageTransformer2DModel
    main_input_name = "hidden_states"
    # We override the items here because the transformer under consideration is small.
    model_split_percents = [0.7, 0.6, 0.6]

    # Skip setting testing with default: AttnProcessor
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        return self.prepare_dummy_input()

    @property
    def input_shape(self):
        return (16, 16)

    @property
    def output_shape(self):
        return (16, 16)

    def prepare_dummy_input(self, height=4, width=4):
        batch_size = 1
        num_latent_channels = embedding_dim = 16
        sequence_length = 7
        vae_scale_factor = 4

        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        encoder_hidden_states_mask = torch.ones((batch_size, sequence_length)).to(torch_device, torch.long)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        orig_height = height * 2 * vae_scale_factor
        orig_width = width * 2 * vae_scale_factor
        img_shapes = [(1, orig_height // vae_scale_factor // 2, orig_width // vae_scale_factor // 2)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "txt_seq_lens": encoder_hidden_states_mask.sum(dim=1).tolist(),
        }

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 3,
            "joint_attention_dim": 16,
            "guidance_embeds": False,
            "axes_dims_rope": (8, 4, 4),
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"QwenImageTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class QwenImageTransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = QwenImageTransformer2DModel

    def prepare_init_args_and_inputs_for_common(self):
        return QwenImageTransformerTests().prepare_init_args_and_inputs_for_common()

    def prepare_dummy_input(self, height, width):
        return QwenImageTransformerTests().prepare_dummy_input(height=height, width=width)

    @pytest.mark.xfail(condition=True, reason="RoPE needs to be revisited.", strict=True)
    def test_torch_compile_recompilation_and_graph_break(self):
        super().test_torch_compile_recompilation_and_graph_break()
