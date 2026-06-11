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

from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class PRXTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return PRXTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (16, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (16, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 16,
            "patch_size": 2,
            "context_in_dim": 1792,
            "hidden_size": 1792,
            "mlp_ratio": 3.5,
            "num_heads": 28,
            "depth": 4,
            "axes_dim": [32, 32],
            "theta": 10_000,
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_latent_channels = 16
        height = width = 16
        sequence_length = 16
        embedding_dim = 1792

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_latent_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
        }


class TestPRXTransformer(PRXTransformerTesterConfig, ModelTesterMixin):
    pass


class TestPRXTransformerTraining(PRXTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"PRXTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
