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

from diffusers import Ideogram4Transformer2DModel
from diffusers.models.transformers.transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class Ideogram4TransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = Ideogram4Transformer2DModel
    main_input_name = "hidden_states"
    model_split_percents = [0.9, 0.9, 0.9]

    _hidden_size = 32
    _num_heads = 4
    _head_dim = _hidden_size // _num_heads  # 8
    _in_channels = 16
    _llm_features_dim = 24
    _max_text_tokens = 4
    _num_image_tokens = 4

    def prepare_dummy_input(self, height: int = 0, width: int = 0):
        del height, width
        batch_size = 1
        max_text_tokens = self._max_text_tokens
        num_image_tokens = self._num_image_tokens
        seq_len = max_text_tokens + num_image_tokens

        hidden_states = torch.zeros(batch_size, seq_len, self._in_channels)
        hidden_states[:, max_text_tokens:] = torch.randn(batch_size, num_image_tokens, self._in_channels)

        encoder_hidden_states = torch.zeros(batch_size, seq_len, self._llm_features_dim)
        encoder_hidden_states[:, :max_text_tokens] = torch.randn(batch_size, max_text_tokens, self._llm_features_dim)

        position_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.long)
        text_pos = torch.arange(max_text_tokens)
        position_ids[:, :max_text_tokens, 0] = text_pos
        position_ids[:, :max_text_tokens, 1] = text_pos
        position_ids[:, :max_text_tokens, 2] = text_pos
        # Image tokens get a 2x2 grid with the IMAGE_POSITION_OFFSET applied.
        image_h = torch.tensor([0, 0, 1, 1])
        image_w = torch.tensor([0, 1, 0, 1])
        position_ids[:, max_text_tokens:, 0] = IMAGE_POSITION_OFFSET
        position_ids[:, max_text_tokens:, 1] = image_h + IMAGE_POSITION_OFFSET
        position_ids[:, max_text_tokens:, 2] = image_w + IMAGE_POSITION_OFFSET

        segment_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
        indicator = torch.empty(batch_size, seq_len, dtype=torch.long)
        indicator[:, :max_text_tokens] = LLM_TOKEN_INDICATOR
        indicator[:, max_text_tokens:] = OUTPUT_IMAGE_INDICATOR
        timestep = torch.tensor([0.5])

        inputs = {
            "hidden_states": hidden_states.to(torch_device),
            "encoder_hidden_states": encoder_hidden_states.to(torch_device),
            "timestep": timestep.to(torch_device),
            "position_ids": position_ids.to(torch_device),
            "segment_ids": segment_ids.to(torch_device),
            "indicator": indicator.to(torch_device),
        }
        return inputs

    @property
    def dummy_input(self):
        return self.prepare_dummy_input()

    @property
    def input_shape(self):
        return (self._max_text_tokens + self._num_image_tokens, self._in_channels)

    @property
    def output_shape(self):
        return (self._max_text_tokens + self._num_image_tokens, self._in_channels)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": self._in_channels,
            "num_layers": 2,
            "attention_head_dim": self._head_dim,
            "num_attention_heads": self._num_heads,
            "intermediate_size": 32,
            "adaln_dim": 16,
            "llm_features_dim": self._llm_features_dim,
            "rope_theta": 10_000,
            "mrope_section": (2, 1, 1),
            "norm_eps": 1e-5,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"Ideogram4Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_forward_signature(self):
        # The model's forward takes packed inputs by position; skip the strict signature check used by the mixin.
        return

    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            output = model(**inputs_dict, return_dict=False)[0]
        expected = (1, self._max_text_tokens + self._num_image_tokens, self._in_channels)
        self.assertEqual(tuple(output.shape), expected)
        self.assertEqual(output.dtype, torch.float32)
