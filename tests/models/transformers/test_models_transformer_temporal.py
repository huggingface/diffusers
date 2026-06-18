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

import torch

from diffusers.models.transformers import TransformerTemporalModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class TemporalTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return TransformerTemporalModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def output_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "num_attention_heads": 8,
            "attention_head_dim": 4,
            "in_channels": 4,
            "num_layers": 1,
            "norm_num_groups": 1,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = 32

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestTemporalTransformer(TemporalTransformerTesterConfig, ModelTesterMixin):
    pass


class TestTemporalTransformerMemory(TemporalTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestTemporalTransformerAttention(TemporalTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestTemporalTransformerTraining(TemporalTransformerTesterConfig, TrainingTesterMixin):
    pass
