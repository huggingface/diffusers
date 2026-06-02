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

from diffusers import AutoencoderKLMochi
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLMochiTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLMochi

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 7, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 15,
            "out_channels": 3,
            "latent_channels": 4,
            "encoder_block_out_channels": (32, 32, 32, 32),
            "decoder_block_out_channels": (32, 32, 32, 32),
            "layers_per_block": (1, 1, 1, 1, 1),
            "act_fn": "silu",
            "scaling_factor": 1,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 7
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLMochi(AutoencoderKLMochiTesterConfig, ModelTesterMixin):
    @pytest.mark.skip("Unsupported test.")
    def test_model_parallelism(self):
        super().test_model_parallelism()


class TestAutoencoderKLMochiTraining(AutoencoderKLMochiTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLMochi."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "MochiDecoder3D",
            "MochiDownBlock3D",
            "MochiEncoder3D",
            "MochiMidBlock3D",
            "MochiUpBlock3D",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAutoencoderKLMochiMemory(AutoencoderKLMochiTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLMochi."""


class TestAutoencoderKLMochiSlicingTiling(AutoencoderKLMochiTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLMochi."""
