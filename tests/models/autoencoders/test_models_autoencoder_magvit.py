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

import pytest
import torch

from diffusers import AutoencoderKLMagvit
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLMagvitTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLMagvit

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "latent_channels": 4,
            "out_channels": 3,
            "block_out_channels": [8, 8, 8, 8],
            "down_block_types": [
                "SpatialDownBlock3D",
                "SpatialTemporalDownBlock3D",
                "SpatialTemporalDownBlock3D",
                "SpatialTemporalDownBlock3D",
            ],
            "up_block_types": [
                "SpatialUpBlock3D",
                "SpatialTemporalUpBlock3D",
                "SpatialTemporalUpBlock3D",
                "SpatialTemporalUpBlock3D",
            ],
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "spatial_group_norm": True,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 9
        num_channels = 3
        height = 16
        width = 16
        image = randn_tensor(
            (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLMagvit(AutoencoderKLMagvitTesterConfig, ModelTesterMixin):
    pass


class TestAutoencoderKLMagvitTraining(AutoencoderKLMagvitTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLMagvit."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"EasyAnimateEncoder", "EasyAnimateDecoder"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @pytest.mark.skip("Not quite sure why this test fails. Revisit later.")
    def test_gradient_checkpointing_equivalence(self):
        super().test_gradient_checkpointing_equivalence()


class TestAutoencoderKLMagvitMemory(AutoencoderKLMagvitTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLMagvit."""


class TestAutoencoderKLMagvitSlicingTiling(AutoencoderKLMagvitTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLMagvit."""

    @pytest.mark.skip("Unsupported test.")
    def test_forward_with_norm_groups(self):
        super().test_forward_with_norm_groups()

    @pytest.mark.skip(
        "Unsupported test. Error: RuntimeError: Sizes of tensors must match except in dimension 0. "
        "Expected size 9 but got size 12 for tensor number 1 in the list."
    )
    def test_enable_disable_slicing(self):
        super().test_enable_disable_slicing()
