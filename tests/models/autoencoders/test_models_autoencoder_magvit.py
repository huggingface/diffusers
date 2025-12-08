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

import unittest

from diffusers import AutoencoderKLMagvit

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLMagvitTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLMagvit
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_magvit_config(self):
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

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        height = 16
        width = 16

        image = floats_tensor((batch_size, num_channels, num_frames, height, width)).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 9, 16, 16)

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_magvit_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"EasyAnimateEncoder", "EasyAnimateDecoder"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("Not quite sure why this test fails. Revisit later.")
    def test_effective_gradient_checkpointing(self):
        pass

    @unittest.skip("Unsupported test.")
    def test_forward_with_norm_groups(self):
        pass

    @unittest.skip(
        "Unsupported test. Error: RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 9 but got size 12 for tensor number 1 in the list."
    )
    def test_enable_disable_slicing(self):
        pass
