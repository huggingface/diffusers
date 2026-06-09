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

from diffusers import AutoencoderKLKVAE
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLKVAETesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLKVAE

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "channels": 32,
            "num_enc_blocks": 1,
            "num_dec_blocks": 1,
            "z_channels": 4,
            "double_z": True,
            "ch_mult": (1, 2),
            "sample_size": 32,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_channels = 3
        sizes = (32, 32)
        image = randn_tensor((batch_size, num_channels, *sizes), generator=self.generator, device=torch_device)
        return {"sample": image}


class TestAutoencoderKLKVAE(AutoencoderKLKVAETesterConfig, ModelTesterMixin):
    pass


class TestAutoencoderKLKVAETraining(AutoencoderKLKVAETesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLKVAE."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"KVAEEncoder2D", "KVAEDecoder2D"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAutoencoderKLKVAEMemory(AutoencoderKLKVAETesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLKVAE."""


class TestAutoencoderKLKVAESlicingTiling(AutoencoderKLKVAETesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLKVAE."""
