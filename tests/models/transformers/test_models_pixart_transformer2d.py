# coding=utf-8
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

from diffusers import PixArtTransformer2DModel, Transformer2DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    slow,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class PixArtTransformer2DModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = PixArtTransformer2DModel
    main_input_name = "hidden_states"
    # We override the items here because the transformer under consideration is small.
    model_split_percents = [0.7, 0.6, 0.6]

    @property
    def dummy_input(self):
        batch_size = 4
        in_channels = 4
        sample_size = 8
        scheduler_num_train_steps = 1000
        cross_attention_dim = 8
        seq_len = 8

        hidden_states = floats_tensor((batch_size, in_channels, sample_size, sample_size)).to(torch_device)
        timesteps = torch.randint(0, scheduler_num_train_steps, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, seq_len, cross_attention_dim)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "added_cond_kwargs": {"aspect_ratio": None, "resolution": None},
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
            "num_layers": 1,
            "patch_size": 2,
            "attention_head_dim": 2,
            "num_attention_heads": 2,
            "in_channels": 4,
            "cross_attention_dim": 8,
            "out_channels": 8,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 8,
            "norm_type": "ada_norm_single",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "use_additional_conditions": False,
            "caption_channels": None,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        super().test_output(
            expected_output_shape=(self.dummy_input[self.main_input_name].shape[0],) + self.output_shape
        )

    def test_correct_class_remapping_from_dict_config(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = Transformer2DModel.from_config(init_dict)
        assert isinstance(model, PixArtTransformer2DModel)

    def test_correct_class_remapping_from_pretrained_config(self):
        config = PixArtTransformer2DModel.load_config("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="transformer")
        model = Transformer2DModel.from_config(config)
        assert isinstance(model, PixArtTransformer2DModel)

    @slow
    def test_correct_class_remapping(self):
        model = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="transformer")
        assert isinstance(model, PixArtTransformer2DModel)
