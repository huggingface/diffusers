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

from diffusers import AutoencoderKLLTX2Video
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLLTX2VideoTesterConfig(BaseModelTesterConfig):
    @property
    def main_input_name(self):
        return "sample"

    @property
    def model_class(self):
        return AutoencoderKLLTX2Video

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "decoder_block_out_channels": (16, 32, 64),
            "layers_per_block": (1, 1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, True, True),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (False, False, False, False),
            "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": False,
            "patch_size": 1,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
            "encoder_spatial_padding_mode": "zeros",
            # Full model uses `reflect` but this does not have deterministic backward implementation, so use `zeros`
            "decoder_spatial_padding_mode": "zeros",
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


class TestAutoencoderKLLTX2Video(AutoencoderKLLTX2VideoTesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    def test_outputs_equivalence(self):
        pytest.skip("Unsupported test.")


class TestAutoencoderKLLTX2VideoTraining(AutoencoderKLLTX2VideoTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLLTX2Video."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "LTX2VideoEncoder3d",
            "LTX2VideoDecoder3d",
            "LTX2VideoDownBlock3D",
            "LTX2VideoMidBlock3d",
            "LTX2VideoUpBlock3d",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAutoencoderKLLTX2VideoMemory(AutoencoderKLLTX2VideoTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLLTX2Video."""


class TestAutoencoderKLLTX2VideoSlicingTiling(AutoencoderKLLTX2VideoTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLLTX2Video."""
