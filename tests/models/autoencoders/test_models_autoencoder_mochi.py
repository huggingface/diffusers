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

from diffusers import AutoencoderKLMochi

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLMochiTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLMochi
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_mochi_config(self):
        return {
            "in_channels": 15,
            "out_channels": 3,
            "latent_channels": 4,
            "encoder_block_out_channels": (32, 32, 32, 32),
            "decoder_block_out_channels": (32, 32, 32, 32),
            "layers_per_block": (1, 1, 1, 1, 1),
            "act_fn": "silu",
            "scaling_factor": 1,
        }

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 7
        num_channels = 3
        sizes = (16, 16)

        image = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 7, 16, 16)

    @property
    def output_shape(self):
        return (3, 7, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_mochi_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "MochiDecoder3D",
            "MochiDownBlock3D",
            "MochiEncoder3D",
            "MochiMidBlock3D",
            "MochiUpBlock3D",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("Unsupported test.")
    def test_model_parallelism(self):
        """
        tests/models/autoencoders/test_models_autoencoder_mochi.py::AutoencoderKLMochiTests::test_outputs_equivalence -
        RuntimeError: values expected sparse tensor layout but got Strided
        """
        pass

    @unittest.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        """
        tests/models/autoencoders/test_models_autoencoder_mochi.py::AutoencoderKLMochiTests::test_outputs_equivalence -
        RuntimeError: values expected sparse tensor layout but got Strided
        """
        pass

    @unittest.skip("Unsupported test.")
    def test_sharded_checkpoints_device_map(self):
        """
        tests/models/autoencoders/test_models_autoencoder_mochi.py::AutoencoderKLMochiTests::test_sharded_checkpoints_device_map -
        RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:5!
        """
