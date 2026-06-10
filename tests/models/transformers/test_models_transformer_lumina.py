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

from diffusers import LuminaNextDiT2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class LuminaNextDiTTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return LuminaNextDiT2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (4, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (4, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 16,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 24,
            "num_layers": 2,
            "num_attention_heads": 3,
            "num_kv_heads": 1,
            "multiple_of": 16,
            "ffn_dim_multiplier": None,
            "norm_eps": 1e-5,
            "learn_sigma": False,
            "qk_norm": True,
            "cross_attention_dim": 32,
            "scaling_factor": 1.0,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = 16
        embedding_dim = 32
        sequence_length = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.rand(size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_mask": randn_tensor((batch_size, sequence_length), generator=self.generator, device=torch_device),
            "image_rotary_emb": randn_tensor((384, 384, 4), generator=self.generator, device=torch_device),
            "cross_attention_kwargs": {},
        }


class TestLuminaNextDiT(LuminaNextDiTTesterConfig, ModelTesterMixin):
    pass


class TestLuminaNextDiTTraining(LuminaNextDiTTesterConfig, TrainingTesterMixin):
    pass
