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

from diffusers import CogView4Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class CogView4TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return CogView4Transformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (4, 8, 8)

    @property
    def input_shape(self) -> tuple:
        return (4, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 2,
            "in_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 4,
            "num_attention_heads": 4,
            "out_channels": 4,
            "text_embed_dim": 8,
            "time_embed_dim": 8,
            "condition_dim": 4,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = 8
        width = 8
        embedding_dim = 8
        sequence_length = 8

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "original_size": torch.tensor([height * 8, width * 8]).unsqueeze(0).repeat(batch_size, 1).to(torch_device),
            "target_size": torch.tensor([height * 8, width * 8]).unsqueeze(0).repeat(batch_size, 1).to(torch_device),
            "crop_coords": torch.tensor([0, 0]).unsqueeze(0).repeat(batch_size, 1).to(torch_device),
        }


class TestCogView4Transformer(CogView4TransformerTesterConfig, ModelTesterMixin):
    pass


class TestCogView4TransformerTraining(CogView4TransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CogView4Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
