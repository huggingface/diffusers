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

import copy

import torch

from diffusers import UNetSpatioTemporalConditionModel
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


class UNetSpatioTemporalConditionModelTesterConfig(BaseModelTesterConfig):
    addition_time_embed_dim = 32

    @property
    def model_class(self):
        return UNetSpatioTemporalConditionModel

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "block_out_channels": (32, 64),
            "down_block_types": (
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            "up_block_types": (
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            "cross_attention_dim": 32,
            "num_attention_heads": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
            "projection_class_embeddings_input_dim": self.addition_time_embed_dim * 3,
            "addition_time_embed_dim": self.addition_time_embed_dim,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 2
        num_channels = 4
        sizes = (32, 32)
        noise = randn_tensor(
            (batch_size, num_frames, num_channels, *sizes), generator=self.generator, device=torch_device
        )
        timestep = torch.tensor([10], device=torch_device)
        encoder_hidden_states = randn_tensor((batch_size, 1, 32), generator=self.generator, device=torch_device)
        add_time_ids = torch.tensor([[6, 127, 0.02]], device=torch_device)
        add_time_ids = torch.cat([add_time_ids, add_time_ids])
        return {
            "sample": noise,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "added_time_ids": add_time_ids,
        }


class TestUNetSpatioTemporalConditionModel(UNetSpatioTemporalConditionModelTesterConfig, ModelTesterMixin):
    def test_model_with_num_attention_heads_tuple(self):
        init_dict = self.get_init_dict()
        init_dict["num_attention_heads"] = (8, 16)
        model = self.model_class(**init_dict).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs()).sample

        assert output.shape == self.get_dummy_inputs()["sample"].shape, "Input and output shapes do not match"

    def test_model_with_cross_attention_dim_tuple(self):
        init_dict = self.get_init_dict()
        init_dict["cross_attention_dim"] = (32, 32)
        model = self.model_class(**init_dict).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs()).sample

        assert output.shape == self.get_dummy_inputs()["sample"].shape, "Input and output shapes do not match"

    def test_pickle(self):
        init_dict = self.get_init_dict()
        init_dict["num_attention_heads"] = (8, 16)
        model = self.model_class(**init_dict).to(torch_device)

        with torch.no_grad():
            sample = model(**self.get_dummy_inputs()).sample

        sample_copy = copy.copy(sample)
        assert (sample - sample_copy).abs().max() < 1e-4


class TestUNetSpatioTemporalConditionModelTraining(UNetSpatioTemporalConditionModelTesterConfig, TrainingTesterMixin):
    """Training tests for UNetSpatioTemporalConditionModel."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "TransformerSpatioTemporalModel",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "UNetMidBlockSpatioTemporal",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestUNetSpatioTemporalConditionModelMemory(UNetSpatioTemporalConditionModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for UNetSpatioTemporalConditionModel."""


class TestUNetSpatioTemporalConditionModelAttention(
    UNetSpatioTemporalConditionModelTesterConfig, AttentionTesterMixin
):
    """Attention processor tests for UNetSpatioTemporalConditionModel."""
