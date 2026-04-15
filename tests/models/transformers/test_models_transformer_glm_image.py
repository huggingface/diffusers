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

from diffusers import GlmImageTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class GlmImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return GlmImageTransformer2DModel

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
            "out_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "text_embed_dim": 32,
            "time_embed_dim": 16,
            "condition_dim": 8,
            "prior_vq_quantizer_codebook_size": 64,
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = 8
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, 32), generator=self.generator, device=torch_device
            ),
            "prior_token_id": torch.randint(0, 64, size=(batch_size,), generator=self.generator).to(torch_device),
            "prior_token_drop": torch.zeros(batch_size, dtype=torch.bool, device=torch_device),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "target_size": torch.tensor([[height, width]] * batch_size, dtype=torch.float32).to(torch_device),
            "crop_coords": torch.tensor([[0, 0]] * batch_size, dtype=torch.float32).to(torch_device),
        }


class TestGlmImageTransformer(GlmImageTransformerTesterConfig, ModelTesterMixin):
    pass


class TestGlmImageTransformerTraining(GlmImageTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"GlmImageTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
