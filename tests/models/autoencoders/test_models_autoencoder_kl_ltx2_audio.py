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

from diffusers import AutoencoderKLLTX2Audio

from ...testing_utils import (
    floats_tensor,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


class AutoencoderKLLTX2AudioTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLLTX2Audio
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_ltx_video_config(self):
        return {
            "in_channels": 2,  # stereo,
            "output_channels": 2,
            "latent_channels": 4,
            "base_channels": 16,
            "ch_mult": (1, 2, 4),
            "resolution": 16,
            "attn_resolutions": None,
            "num_res_blocks": 2,
            "norm_type": "pixel",
            "causality_axis": "height",
            "mid_block_add_attention": False,
            "sample_rate": 16000,
            "mel_hop_length": 160,
            "mel_bins": 16,
            "is_causal": True,
            "double_z": True,
        }

    @property
    def dummy_input(self):
        batch_size = 2
        num_channels = 2
        num_frames = 8
        num_mel_bins = 16

        spectrogram = floats_tensor((batch_size, num_channels, num_frames, num_mel_bins)).to(torch_device)

        input_dict = {"sample": spectrogram}
        return input_dict

    @property
    def input_shape(self):
        return (2, 5, 16)

    @property
    def output_shape(self):
        return (2, 5, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_autoencoder_kl_ltx_video_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    # Overriding as output shape is not the same as input shape for LTX 2.0 audio VAE
    def test_output(self):
        super().test_output(expected_output_shape=(2, 2, 5, 16))

    @unittest.skip("Unsupported test.")
    def test_outputs_equivalence(self):
        pass

    @unittest.skip("AutoencoderKLLTX2Audio does not support `norm_num_groups` because it does not use GroupNorm.")
    def test_forward_with_norm_groups(self):
        pass
