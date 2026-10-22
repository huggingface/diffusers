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

from diffusers import MiniMaxMusic3Transformer1DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class MiniMaxMusic3Transformer1DTesterConfig(BaseModelTesterConfig):
    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_class(self):
        return MiniMaxMusic3Transformer1DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (8, 24)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int]:
        return {
            "in_channels": 8,
            "condition_dim": 16,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "ff_inner_dim": 32,
            "rotary_dim": 4,
            "fourier_embedding_dim": 8,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        latent_length = 24
        in_channels = 8
        condition_dim = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, in_channels, latent_length), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, latent_length, condition_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.full((batch_size,), 0.5, device=torch_device),
        }


class TestMiniMaxMusic3Transformer1DModel(MiniMaxMusic3Transformer1DTesterConfig, ModelTesterMixin):
    pass


class TestMiniMaxMusic3Transformer1DMemory(MiniMaxMusic3Transformer1DTesterConfig, MemoryTesterMixin):
    pass


class TestMiniMaxMusic3Transformer1DTorchCompile(MiniMaxMusic3Transformer1DTesterConfig, TorchCompileTesterMixin):
    pass


class TestMiniMaxMusic3Transformer1DAttention(MiniMaxMusic3Transformer1DTesterConfig, AttentionTesterMixin):
    pass
