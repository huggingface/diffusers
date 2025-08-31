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

import torch

from diffusers import AutoencoderKLLTXVideo

from ...testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


enable_full_determinism()


class AutoencoderKLLTXVideo090Tests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLLTXVideo
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_ltx_video_config(self):
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "decoder_block_out_channels": (8, 8, 8, 8),
            "layers_per_block": (1, 1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, False, False),
            "decoder_spatio_temporal_scaling": (True, True, False, False),
            "decoder_inject_noise": (False, False, False, False, False),
            "upsample_residual": (False, False, False, False),
            "upsample_factor": (1, 1, 1, 1),
            "timestep_conditioning": False,
            "patch_size": 1,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
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
    def input_shape(self):
        return (3, 9, 16, 16)

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_ltx_video_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "LTXVideoEncoder3d",
            "LTXVideoDecoder3d",
            "LTXVideoDownBlock3D",
            "LTXVideoMidBlock3d",
            "LTXVideoUpBlock3d",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        pass

    @unittest.skip("AutoencoderKLLTXVideo does not support `norm_num_groups` because it does not use GroupNorm.")
    def test_forward_with_norm_groups(self):
        pass


class AutoencoderKLLTXVideo091Tests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLLTXVideo
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_ltx_video_config(self):
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "decoder_block_out_channels": (16, 32, 64),
            "layers_per_block": (1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, True, False),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (True, True, True, False),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": True,
            "patch_size": 1,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
        }

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)

        image = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)
        timestep = torch.tensor([0.05] * batch_size, device=torch_device)

        return {"sample": image, "temb": timestep}

    @property
    def input_shape(self):
        return (3, 9, 16, 16)

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_ltx_video_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "LTXVideoEncoder3d",
            "LTXVideoDecoder3d",
            "LTXVideoDownBlock3D",
            "LTXVideoMidBlock3d",
            "LTXVideoUpBlock3d",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        pass

    @unittest.skip("AutoencoderKLLTXVideo does not support `norm_num_groups` because it does not use GroupNorm.")
    def test_forward_with_norm_groups(self):
        pass

    def test_enable_disable_tiling(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)

        inputs_dict.update({"return_dict": False})

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
