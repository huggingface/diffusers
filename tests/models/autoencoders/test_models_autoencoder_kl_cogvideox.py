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

from diffusers import AutoencoderKLCogVideoX
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLCogVideoXTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLCogVideoX

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 8, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": (
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ),
            "up_block_types": (
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
            ),
            "block_out_channels": (8, 8, 8, 8),
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 2,
            "temporal_compression_ratio": 4,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 4
        num_frames = 8
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLCogVideoX(AutoencoderKLCogVideoXTesterConfig, ModelTesterMixin):
    pass


class TestAutoencoderKLCogVideoXTraining(AutoencoderKLCogVideoXTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLCogVideoX."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "CogVideoXDownBlock3D",
            "CogVideoXDecoder3D",
            "CogVideoXEncoder3D",
            "CogVideoXUpBlock3D",
            "CogVideoXMidBlock3D",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAutoencoderKLCogVideoXMemory(AutoencoderKLCogVideoXTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLCogVideoX."""


class TestAutoencoderKLCogVideoXSlicingTiling(AutoencoderKLCogVideoXTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLCogVideoX."""

    # Overwritten because the base test's block_out_channels doesn't account for the length of down_block_types.
    def test_forward_with_norm_groups(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 32, 32, 32)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        assert output is not None
        expected_shape = inputs_dict["sample"].shape
        assert output.shape == expected_shape, "Input and output shapes do not match"
