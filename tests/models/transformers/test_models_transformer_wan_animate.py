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

from diffusers import WanAnimateTransformer3DModel

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class WanAnimateTransformer3DTests(ModelTesterMixin, unittest.TestCase):
    model_class = WanAnimateTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 4
        num_frames = 20  # To make the shapes work out; for complicated reasons we want 21 to divide num_frames + 1
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        clip_seq_len = 12
        clip_dim = 16

        inference_segment_length = 77  # The inference segment length in the full Wan2.2-Animate-14B model
        face_height = 16  # Should be square and match `motion_encoder_size` below
        face_width = 16

        hidden_states = torch.randn((batch_size, 2 * num_channels + 4, num_frames + 1, height, width)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, text_encoder_embedding_dim)).to(torch_device)
        clip_ref_features = torch.randn((batch_size, clip_seq_len, clip_dim)).to(torch_device)
        pose_latents = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        face_pixel_values = torch.randn((batch_size, 3, inference_segment_length, face_height, face_width)).to(
            torch_device
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_image": clip_ref_features,
            "pose_hidden_states": pose_latents,
            "face_pixel_values": face_pixel_values,
        }

    @property
    def input_shape(self):
        return (12, 1, 16, 16)

    @property
    def output_shape(self):
        return (4, 1, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        # Use custom channel sizes since the default Wan Animate channel sizes will cause the motion encoder to
        # contain the vast majority of the parameters in the test model
        channel_sizes = {"4": 16, "8": 16, "16": 16}

        init_dict = {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 12,  # 2 * C + 4 = 2 * 4 + 4 = 12
            "latent_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "image_dim": 16,
            "rope_max_seq_len": 32,
            "motion_encoder_channel_sizes": channel_sizes,  # Start of Wan Animate-specific config
            "motion_encoder_size": 16,  # Ensures that there will be 2 motion encoder resblocks
            "motion_style_dim": 8,
            "motion_dim": 4,
            "motion_encoder_dim": 16,
            "face_encoder_hidden_dim": 16,
            "face_encoder_num_heads": 2,
            "inject_face_latents_blocks": 2,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"WanAnimateTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    # Override test_output because the transformer output is expected to have less channels than the main transformer
    # input.
    def test_output(self):
        expected_output_shape = (1, 4, 21, 16, 16)
        super().test_output(expected_output_shape=expected_output_shape)


class WanAnimateTransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = WanAnimateTransformer3DModel

    def prepare_init_args_and_inputs_for_common(self):
        return WanAnimateTransformer3DTests().prepare_init_args_and_inputs_for_common()
