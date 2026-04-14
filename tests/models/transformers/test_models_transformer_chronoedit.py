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

from diffusers import ChronoEditTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class ChronoEditTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return ChronoEditTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (16, 8, 8)

    @property
    def input_shape(self) -> tuple:
        return (16, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 16,
            "out_channels": 16,
            "text_dim": 32,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "eps": 1e-06,
            "image_dim": None,
            "added_kv_proj_dim": None,
            "rope_max_seq_len": 64,
            "pos_embed_seq_len": None,
            "rope_temporal_skip_len": 8,
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 16
        num_frames = 2
        height = 8
        width = 8
        embedding_dim = 32
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states_image": None,
        }


class TestChronoEditTransformer(ChronoEditTransformerTesterConfig, ModelTesterMixin):
    pass


class TestChronoEditTransformerTraining(ChronoEditTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"ChronoEditTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
