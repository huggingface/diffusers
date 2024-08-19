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

from diffusers import LuminaNextDiT2DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class LuminaNextDiT2DModelTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = LuminaNextDiT2DModel
    main_input_name = "hidden_states"

    @property
    def dummy_input(self):
        """
        Args:
            None
        Returns:
            Dict: Dictionary of dummy input tensors
        """
        batch_size = 2  # N
        num_channels = 4  # C
        height = width = 16  # H, W
        embedding_dim = 32  # D
        sequence_length = 16  # L

        hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        timestep = torch.rand(size=(batch_size,)).to(torch_device)
        encoder_mask = torch.randn(size=(batch_size, sequence_length)).to(torch_device)
        image_rotary_emb = torch.randn((384, 384, 4)).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_mask": encoder_mask,
            "image_rotary_emb": image_rotary_emb,
            "cross_attention_kwargs": {},
        }

    @property
    def input_shape(self):
        """
        Args:
            None
        Returns:
            Tuple: (int, int, int)
        """
        return (4, 16, 16)

    @property
    def output_shape(self):
        """
        Args:
            None
        Returns:
            Tuple: (int, int, int)
        """
        return (4, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        """
        Args:
            None

        Returns:
            Tuple: (Dict, Dict)
        """
        init_dict = {
            "sample_size": 16,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 24,
            "num_layers": 2,
            "num_attention_heads": 3,
            "num_kv_heads": 1,
            "multiple_of": 16,
            "ffn_dim_multiplier": None,
            "norm_eps": 1e-5,
            "learn_sigma": False,
            "qk_norm": True,
            "cross_attention_dim": 32,
            "scaling_factor": 1.0,
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict
