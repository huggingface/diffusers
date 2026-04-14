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

from diffusers import MochiTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class MochiTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return MochiTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.7, 0.6, 0.6]

    @property
    def output_shape(self) -> tuple:
        return (4, 2, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (4, 2, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 2,
            "pooled_projection_dim": 16,
            "in_channels": 4,
            "out_channels": None,
            "qk_norm": "rms_norm",
            "text_embed_dim": 16,
            "time_embed_dim": 4,
            "activation_fn": "swiglu",
            "max_sequence_length": 16,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        embedding_dim = 16
        sequence_length = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.bool).to(torch_device),
        }


class TestMochiTransformer(MochiTransformerTesterConfig, ModelTesterMixin):
    pass


class TestMochiTransformerTraining(MochiTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"MochiTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
