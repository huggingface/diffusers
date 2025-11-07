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

from diffusers import AutoencoderKLHunyuanVideo
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import prepare_causal_attention_mask

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLHunyuanVideoTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLHunyuanVideo
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_kl_hunyuan_video_config(self):
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "down_block_types": (
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ),
            "up_block_types": (
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ),
            "block_out_channels": (8, 8, 8, 8),
            "layers_per_block": 1,
            "act_fn": "silu",
            "norm_num_groups": 4,
            "scaling_factor": 0.476986,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "mid_block_add_attention": True,
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
        init_dict = self.get_autoencoder_kl_hunyuan_video_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "HunyuanVideoDecoder3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoEncoder3D",
            "HunyuanVideoMidBlock3D",
            "HunyuanVideoUpBlock3D",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    # We need to overwrite this test because the base test does not account length of down_block_types
    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 16, 16, 16)

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

    def test_prepare_causal_attention_mask(self):
        def prepare_causal_attention_mask_orig(
            num_frames: int, height_width: int, dtype: torch.dtype, device: torch.device, batch_size: int = None
        ) -> torch.Tensor:
            seq_len = num_frames * height_width
            mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
            for i in range(seq_len):
                i_frame = i // height_width
                mask[i, : (i_frame + 1) * height_width] = 0
            if batch_size is not None:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            return mask

        # test with some odd shapes
        original_mask = prepare_causal_attention_mask_orig(
            num_frames=31, height_width=111, dtype=torch.float32, device=torch_device
        )
        new_mask = prepare_causal_attention_mask(
            num_frames=31, height_width=111, dtype=torch.float32, device=torch_device
        )
        self.assertTrue(
            torch.allclose(original_mask, new_mask),
            "Causal attention mask should be the same",
        )
