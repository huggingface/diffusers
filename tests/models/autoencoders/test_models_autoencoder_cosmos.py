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

from diffusers import AutoencoderKLCosmos
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLCosmosTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLCosmos

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 9, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "encoder_block_out_channels": (8, 8, 8, 8),
            "decode_block_out_channels": (8, 8, 8, 8),
            "attention_resolutions": (8,),
            "resolution": 64,
            "num_layers": 2,
            "patch_size": 4,
            "patch_type": "haar",
            "scaling_factor": 1.0,
            "spatial_compression_ratio": 4,
            "temporal_compression_ratio": 4,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 9
        num_channels = 3
        height = 32
        width = 32
        image = randn_tensor(
            (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLCosmos(AutoencoderKLCosmosTesterConfig, ModelTesterMixin):
    base_precision = 1e-2


class TestAutoencoderKLCosmosTraining(AutoencoderKLCosmosTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosEncoder3d", "CosmosDecoder3d"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @pytest.mark.skip("Not sure why this test fails. Investigate later.")
    def test_gradient_checkpointing_equivalence(self):
        super().test_gradient_checkpointing_equivalence()


class TestAutoencoderKLCosmosMemory(AutoencoderKLCosmosTesterConfig, MemoryTesterMixin):
    pass


class TestAutoencoderKLCosmosSlicingTiling(AutoencoderKLCosmosTesterConfig, NewAutoencoderTesterMixin):
    pass
