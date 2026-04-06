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

from diffusers import HunyuanDiT2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class HunyuanDiTTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HunyuanDiT2DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-hunyuan-dit-pipe"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (8, 8, 8)

    @property
    def input_shape(self) -> tuple:
        return (4, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 8,
            "patch_size": 2,
            "in_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "cross_attention_dim": 8,
            "cross_attention_dim_t5": 8,
            "pooled_projection_dim": 4,
            "hidden_size": 16,
            "text_len": 4,
            "text_len_t5": 4,
            "activation_fn": "gelu-approximate",
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = 8
        embedding_dim = 8
        sequence_length = 4
        sequence_length_t5 = 4

        hidden_states = randn_tensor(
            (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )
        text_embedding_mask = torch.ones(size=(batch_size, sequence_length)).to(torch_device)
        encoder_hidden_states_t5 = randn_tensor(
            (batch_size, sequence_length_t5, embedding_dim), generator=self.generator, device=torch_device
        )
        text_embedding_mask_t5 = torch.ones(size=(batch_size, sequence_length_t5)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,), generator=self.generator).float().to(torch_device)

        original_size = [1024, 1024]
        target_size = [16, 16]
        crops_coords_top_left = [0, 0]
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids] * batch_size, dtype=torch.float32).to(torch_device)
        style = torch.zeros(size=(batch_size,), dtype=int).to(torch_device)
        image_rotary_emb = [
            torch.ones(size=(1, 8), dtype=torch.float32),
            torch.zeros(size=(1, 8), dtype=torch.float32),
        ]

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embedding_mask": text_embedding_mask,
            "encoder_hidden_states_t5": encoder_hidden_states_t5,
            "text_embedding_mask_t5": text_embedding_mask_t5,
            "timestep": timestep,
            "image_meta_size": add_time_ids,
            "style": style,
            "image_rotary_emb": image_rotary_emb,
        }


class TestHunyuanDiT(HunyuanDiTTesterConfig, ModelTesterMixin):
    def test_output(self):
        batch_size = self.get_dummy_inputs()[self.main_input_name].shape[0]
        super().test_output(expected_output_shape=(batch_size,) + self.output_shape)


class TestHunyuanDiTTraining(HunyuanDiTTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HunyuanDiT2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
