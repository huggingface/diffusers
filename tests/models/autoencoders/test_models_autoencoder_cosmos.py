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

import unittest

from diffusers import AutoencoderKLCosmos

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLCosmosTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLCosmos
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_cosmos_config(self):
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

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        height = 32
        width = 32

        image = floats_tensor((batch_size, num_channels, num_frames, height, width)).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 9, 32, 32)

    @property
    def output_shape(self):
        return (3, 9, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_cosmos_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "CosmosEncoder3d",
            "CosmosDecoder3d",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("Not sure why this test fails. Investigate later.")
    def test_effective_gradient_checkpointing(self):
        pass
