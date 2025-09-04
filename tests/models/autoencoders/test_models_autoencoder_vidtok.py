# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import torch

from diffusers import AutoencoderVidTok
from diffusers.utils.testing_utils import (
    floats_tensor,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


class AutoencoderVidTokTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AutoencoderVidTok
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_vidtok_config(self):
        return {
            "is_causal": False,
            "in_channels": 3,
            "out_channels": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4, 4],
            "z_channels": 6,
            "double_z": False,
            "num_res_blocks": 2,
            "regularizer": "fsq",
            "codebook_size": 262144,
        }

    @property
    def dummy_input(self):
        batch_size = 4
        num_frames = 16
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 16, 32, 32)

    @property
    def output_shape(self):
        return (3, 16, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_vidtok_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_enable_disable_tiling(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)

        torch.manual_seed(0)
        output_without_tiling = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        torch.manual_seed(0)
        model.enable_tiling()
        output_with_tiling = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        self.assertLess(
            (output_without_tiling.detach().cpu().numpy() - output_with_tiling.detach().cpu().numpy()).max(),
            0.5,
            "VAE tiling should not affect the inference results",
        )

        torch.manual_seed(0)
        model.disable_tiling()
        output_without_tiling_2 = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        self.assertEqual(
            output_without_tiling.detach().cpu().numpy().all(),
            output_without_tiling_2.detach().cpu().numpy().all(),
            "Without tiling outputs should match with the outputs when tiling is manually disabled.",
        )

    def test_enable_disable_slicing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)

        inputs_dict.update({"return_dict": False})

        torch.manual_seed(0)
        output_without_slicing = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        torch.manual_seed(0)
        model.enable_slicing()
        output_with_slicing = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        self.assertLess(
            (output_without_slicing.detach().cpu().numpy() - output_with_slicing.detach().cpu().numpy()).max(),
            0.5,
            "VAE slicing should not affect the inference results",
        )

        torch.manual_seed(0)
        model.disable_slicing()
        output_without_slicing_2 = model(**inputs_dict, generator=torch.manual_seed(0))[0]

        self.assertEqual(
            output_without_slicing.detach().cpu().numpy().all(),
            output_without_slicing_2.detach().cpu().numpy().all(),
            "Without slicing outputs should match with the outputs when slicing is manually disabled.",
        )

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "VidTokEncoder3D",
            "VidTokDecoder3D",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_forward_with_norm_groups(self):
        r"""VidTok uses layernorm instead of groupnorm."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    @unittest.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        pass
