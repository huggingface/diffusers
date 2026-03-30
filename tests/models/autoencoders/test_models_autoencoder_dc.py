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

from diffusers import AutoencoderDC

from ...testing_utils import IS_GITHUB_ACTIONS, enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderDCTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderDC

    @property
    def output_shape(self):
        return (3, 32, 32)

    def get_init_dict(self):
        return {
            "in_channels": 3,
            "latent_channels": 4,
            "attention_head_dim": 2,
            "encoder_block_types": (
                "ResBlock",
                "EfficientViTBlock",
            ),
            "decoder_block_types": (
                "ResBlock",
                "EfficientViTBlock",
            ),
            "encoder_block_out_channels": (8, 8),
            "decoder_block_out_channels": (8, 8),
            "encoder_qkv_multiscales": ((), (5,)),
            "decoder_qkv_multiscales": ((), (5,)),
            "encoder_layers_per_block": (1, 1),
            "decoder_layers_per_block": [1, 1],
            "downsample_block_type": "conv",
            "upsample_block_type": "interpolate",
            "decoder_norm_types": "rms_norm",
            "decoder_act_fns": "silu",
            "scaling_factor": 0.41407,
        }

    def get_dummy_inputs(self, seed=0):
        torch.manual_seed(seed)
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)
        image = torch.randn(batch_size, num_channels, *sizes).to(torch_device)
        return {"sample": image}

    # Bridge for AutoencoderTesterMixin which still uses the old interface
    def prepare_init_args_and_inputs_for_common(self):
        return self.get_init_dict(), self.get_dummy_inputs()


class TestAutoencoderDC(AutoencoderDCTesterConfig, ModelTesterMixin):
    base_precision = 1e-2


class TestAutoencoderDCTraining(AutoencoderDCTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderDC."""


class TestAutoencoderDCMemory(AutoencoderDCTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderDC."""

    @pytest.mark.skipif(IS_GITHUB_ACTIONS, reason="Skipping test inside GitHub Actions environment")
    def test_layerwise_casting_memory(self):
        super().test_layerwise_casting_memory()


class TestAutoencoderDCSlicingTiling(AutoencoderDCTesterConfig, AutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderDC."""
