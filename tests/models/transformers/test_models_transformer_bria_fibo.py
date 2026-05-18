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

from diffusers import BriaFiboTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class BriaFiboTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return BriaFiboTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.8, 0.7, 0.7]

    @property
    def output_shape(self) -> tuple:
        return (256, 48)

    @property
    def input_shape(self) -> tuple:
        return (16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 1,
            "in_channels": 48,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "joint_attention_dim": 64,
            "text_encoder_dim": 32,
            "pooled_projection_dim": None,
            "axes_dims_rope": [0, 4, 4],
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_latent_channels = 48
        num_image_channels = 3
        height = width = 16
        sequence_length = 32
        embedding_dim = 64

        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )
        return {
            "hidden_states": randn_tensor(
                (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": randn_tensor(
                (height * width, num_image_channels), generator=self.generator, device=torch_device
            ),
            "txt_ids": randn_tensor(
                (sequence_length, num_image_channels), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
            "text_encoder_layers": [encoder_hidden_states[:, :, :32], encoder_hidden_states[:, :, :32]],
        }


class TestBriaFiboTransformer(BriaFiboTransformerTesterConfig, ModelTesterMixin):
    pass


class TestBriaFiboTransformerTraining(BriaFiboTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"BriaFiboTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
