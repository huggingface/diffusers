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

from diffusers import HiDreamImageTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class HiDreamTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HiDreamImageTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.8, 0.8, 0.9]

    @property
    def output_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def input_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 2,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "caption_channels": [8, 4],
            "text_emb_dim": 8,
            "num_routed_experts": 2,
            "num_activated_experts": 2,
            "axes_dims_rope": (4, 2, 2),
            "max_resolution": (32, 32),
            "llama_layers": (0, 1),
            "force_inference_output": True,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = 32
        embedding_dim_t5, embedding_dim_llama, embedding_dim_pooled = 8, 4, 8
        sequence_length = 8

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states_t5": randn_tensor(
                (batch_size, sequence_length, embedding_dim_t5), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states_llama3": randn_tensor(
                (batch_size, batch_size, sequence_length, embedding_dim_llama),
                generator=self.generator,
                device=torch_device,
            ),
            "pooled_embeds": randn_tensor(
                (batch_size, embedding_dim_pooled), generator=self.generator, device=torch_device
            ),
            "timesteps": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestHiDreamTransformer(HiDreamTransformerTesterConfig, ModelTesterMixin):
    pass


class TestHiDreamTransformerTraining(HiDreamTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HiDreamImageTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
