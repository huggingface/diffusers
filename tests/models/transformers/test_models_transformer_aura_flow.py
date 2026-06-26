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

from diffusers import AuraFlowTransformer2DModel
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


class AuraFlowTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AuraFlowTransformer2DModel

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
    def model_split_percents(self) -> list:
        # We override the items here because the transformer under consideration is small.
        return [0.7, 0.6, 0.6]

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "num_mmdit_layers": 1,
            "num_single_dit_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "caption_projection_dim": 32,
            "joint_attention_dim": 32,
            "out_channels": 4,
            "pos_embed_max_size": 256,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = embedding_dim = 32
        sequence_length = 256

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestAuraFlowTransformer(AuraFlowTransformerTesterConfig, ModelTesterMixin):
    pass


class TestAuraFlowTransformerMemory(AuraFlowTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestAuraFlowTransformerAttention(AuraFlowTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestAuraFlowTransformerTraining(AuraFlowTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"AuraFlowTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
