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

from diffusers import AutoencoderKLWan

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLWanTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLWan
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_wan_config(self):
        return {
            "base_dim": 3,
            "z_dim": 16,
            "dim_mult": [1, 1, 1, 1],
            "num_res_blocks": 1,
            "temperal_downsample": [False, True, True],
        }

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        image = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)
        return {"sample": image}

    @property
    def dummy_input_tiling(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (128, 128)
        image = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)
        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 9, 16, 16)

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_wan_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def prepare_init_args_and_inputs_for_tiling(self):
        init_dict = self.get_autoencoder_kl_wan_config()
        inputs_dict = self.dummy_input_tiling
        return init_dict, inputs_dict

    @unittest.skip("Gradient checkpointing has not been implemented yet")
    def test_gradient_checkpointing_is_applied(self):
        pass

    @unittest.skip("Test not supported")
    def test_forward_with_norm_groups(self):
        pass

    @unittest.skip("RuntimeError: fill_out not implemented for 'Float8_e4m3fn'")
    def test_layerwise_casting_inference(self):
        pass

    @unittest.skip("RuntimeError: fill_out not implemented for 'Float8_e4m3fn'")
    def test_layerwise_casting_training(self):
        pass
