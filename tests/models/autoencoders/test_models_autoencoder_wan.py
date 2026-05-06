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

from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLWanTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLWan

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "base_dim": 3,
            "z_dim": 16,
            "dim_mult": [1, 1, 1, 1],
            "num_res_blocks": 1,
            "temperal_downsample": [False, True, True],
        }

    def get_dummy_inputs(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLWan(AutoencoderKLWanTesterConfig, ModelTesterMixin):
    base_precision = 1e-2


class TestAutoencoderKLWanTraining(AutoencoderKLWanTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLWan."""

    @pytest.mark.skip(reason="Gradient checkpointing has not been implemented yet")
    def test_gradient_checkpointing_is_applied(self):
        pass


class TestAutoencoderKLWanMemory(AutoencoderKLWanTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLWan."""

    @pytest.mark.skip(reason="RuntimeError: fill_out not implemented for 'Float8_e4m3fn'")
    def test_layerwise_casting_memory(self):
        pass

    @pytest.mark.skip(reason="RuntimeError: fill_out not implemented for 'Float8_e4m3fn'")
    def test_layerwise_casting_training(self):
        pass


class TestAutoencoderKLWanSlicingTiling(AutoencoderKLWanTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLWan."""
