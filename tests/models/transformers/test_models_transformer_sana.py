# Copyright 2026 HuggingFace Inc.
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

import pytest
import torch

from diffusers import SanaTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import IS_GITHUB_ACTIONS, enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    SingleFileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class SanaTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return SanaTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        return True

    @property
    def output_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def input_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def model_split_percents(self) -> list:
        return [0.7, 0.7, 0.9]

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 1,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 4,
            "num_attention_heads": 2,
            "num_cross_attention_heads": 2,
            "cross_attention_head_dim": 4,
            "cross_attention_dim": 8,
            "caption_channels": 8,
            "sample_size": 32,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_channels = 4
        height = 32
        width = 32
        embedding_dim = 8
        sequence_length = 8

        hidden_states = randn_tensor(
            (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )
        timestep = torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }


class TestSanaTransformer(SanaTransformerTesterConfig, ModelTesterMixin):
    pass


class TestSanaTransformerMemory(SanaTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestSanaTransformerTraining(SanaTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"SanaTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestSanaTransformerAttention(SanaTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestSanaTransformerLoRA(SanaTransformerTesterConfig, LoraTesterMixin):
    @pytest.mark.skipif(IS_GITHUB_ACTIONS, reason="Skipping test inside GitHub Actions environment")
    def test_lora_layerwise_casting_inference(self):
        super().test_lora_layerwise_casting_inference()


class TestSanaTransformer2DSingleFile(SanaTransformerTesterConfig, SingleFileTesterMixin):
    @property
    def ckpt_path(self):
        return "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/blob/main/checkpoints/Sana_1600M_1024px.pth"

    @property
    def pretrained_model_name_or_path(self):
        return "Efficient-Large-Model/Sana_1600M_1024px_diffusers"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}
