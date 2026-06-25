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

import torch

from diffusers import Krea2Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class Krea2TransformerTesterConfig(BaseModelTesterConfig):
    _head_dim = 8
    _num_heads = 4
    _num_kv_heads = 2
    _in_channels = 16
    _text_hidden_dim = 16
    _num_text_layers = 3
    _text_seq_len = 4
    _grid_size = 2  # 2x2 image grid -> 4 image tokens

    @property
    def model_class(self):
        return Krea2Transformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, self._grid_size * self._grid_size, self._in_channels)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, self._grid_size * self._grid_size, self._in_channels)

    @property
    def model_split_percents(self) -> list:
        return [0.9, 0.9, 0.9]

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": self._in_channels,
            "num_layers": 2,
            "attention_head_dim": self._head_dim,
            "num_attention_heads": self._num_heads,
            "num_key_value_heads": self._num_kv_heads,
            "intermediate_size": 32,
            "timestep_embed_dim": 8,
            "text_hidden_dim": self._text_hidden_dim,
            "num_text_layers": self._num_text_layers,
            "text_num_attention_heads": 2,
            "text_num_key_value_heads": 1,
            "text_intermediate_size": 16,
            "num_layerwise_text_blocks": 1,
            "num_refiner_text_blocks": 1,
            "axes_dims_rope": (4, 2, 2),
            "rope_theta": 1000.0,
            "norm_eps": 1e-5,
        }

    def get_dummy_inputs(self, height: int | None = None, width: int | None = None) -> dict[str, torch.Tensor]:
        # height/width are the latent-grid dimensions (number of image tokens per axis).
        height = height if height is not None else self._grid_size
        width = width if width is not None else self._grid_size
        batch_size = 1
        text_seq_len = self._text_seq_len
        num_image_tokens = height * width

        hidden_states = randn_tensor(
            (batch_size, num_image_tokens, self._in_channels),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, text_seq_len, self._num_text_layers, self._text_hidden_dim),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )
        timestep = torch.tensor([0.5], device=torch_device, dtype=self.torch_dtype)

        position_ids = torch.zeros(text_seq_len + num_image_tokens, 3, device=torch_device)
        grid_h = torch.arange(height, device=torch_device).repeat_interleave(width)
        grid_w = torch.arange(width, device=torch_device).repeat(height)
        position_ids[text_seq_len:, 1] = grid_h
        position_ids[text_seq_len:, 2] = grid_w

        # Mark the last text token as padding to exercise the key-padding mask path.
        encoder_attention_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool, device=torch_device)
        encoder_attention_mask[:, -1] = False

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "position_ids": position_ids,
            "encoder_attention_mask": encoder_attention_mask,
        }


class TestKrea2TransformerModel(Krea2TransformerTesterConfig, ModelTesterMixin):
    """Core model tests for the Krea 2 Transformer."""


class TestKrea2TransformerMemory(Krea2TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the Krea 2 Transformer."""


class TestKrea2TransformerTorchCompile(Krea2TransformerTesterConfig, TorchCompileTesterMixin):
    """torch.compile tests for the Krea 2 Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]


class TestKrea2TransformerTraining(Krea2TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for the Krea 2 Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"Krea2Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestKrea2TransformerAttention(Krea2TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the Krea 2 Transformer."""
