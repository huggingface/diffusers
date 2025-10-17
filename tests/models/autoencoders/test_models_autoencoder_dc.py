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

from diffusers import AutoencoderDC

from ...testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderDCTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderDC
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_dc_config(self):
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

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_dc_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict
