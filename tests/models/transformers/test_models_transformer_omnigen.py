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

from diffusers import OmniGenTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class OmniGenTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return OmniGenTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.1, 0.1, 0.1]

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
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 32,
            "num_layers": 20,
            "pad_token_id": 0,
            "vocab_size": 1000,
            "in_channels": 4,
            "time_step_dim": 4,
            "rope_scaling": {"long_factor": list(range(1, 3)), "short_factor": list(range(1, 3))},
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = 8
        width = 8
        sequence_length = 24

        hidden_states = randn_tensor(
            (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
        )
        attn_seq_length = sequence_length + 1 + height * width // 2 // 2

        return {
            "hidden_states": hidden_states,
            "timestep": torch.rand(size=(batch_size,), generator=self.generator).to(torch_device),
            "input_ids": torch.randint(0, 10, (batch_size, sequence_length), generator=self.generator).to(
                torch_device
            ),
            "input_img_latents": [
                randn_tensor((1, num_channels, height, width), generator=self.generator, device=torch_device)
            ],
            "input_image_sizes": {0: [[0, 0 + height * width // 2 // 2]]},
            "attention_mask": torch.ones((batch_size, attn_seq_length, attn_seq_length)).to(torch_device),
            "position_ids": torch.LongTensor([list(range(attn_seq_length))] * batch_size).to(torch_device),
        }


class TestOmniGenTransformer(OmniGenTransformerTesterConfig, ModelTesterMixin):
    pass


class TestOmniGenTransformerTraining(OmniGenTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"OmniGenTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestOmniGenTransformerCompile(OmniGenTransformerTesterConfig, TorchCompileTesterMixin):
    pass
