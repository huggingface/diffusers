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

from diffusers import StableCascadeUNet

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class StableCascadeUNetTests(ModelTesterMixin, unittest.TestCase):
    model_class = StableCascadeUNet
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 2
        sample = floats_tensor((batch_size, 4, 8, 8)).to(torch_device)
        timestep_ratio = torch.ones(batch_size, device=torch_device)
        clip_text_pooled = floats_tensor((batch_size, 1, 8)).to(torch_device)

        return {
            "sample": sample,
            "timestep_ratio": timestep_ratio,
            "clip_text_pooled": clip_text_pooled,
        }

    @property
    def input_shape(self):
        return (4, 8, 8)

    @property
    def output_shape(self):
        return (4, 8, 8)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 4,
            "out_channels": 4,
            "conditioning_dim": 8,
            "block_out_channels": (8,),
            "num_attention_heads": (1,),
            "down_num_layers_per_block": (1,),
            "up_num_layers_per_block": (1,),
            "down_blocks_repeat_mappers": (1,),
            "up_blocks_repeat_mappers": (1,),
            "block_types_per_layer": (("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),),
            "clip_text_in_channels": 8,
            "clip_text_pooled_in_channels": 8,
            "clip_image_in_channels": 8,
            "dropout": 0.0,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_get_clip_embeddings_accepts_2d_pooled_text(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        clip_text_pooled = floats_tensor((2, 8)).to(torch_device)
        clip = model.get_clip_embeddings(clip_text_pooled)

        self.assertEqual(clip.shape, (2, model.config.clip_seq, model.config.conditioning_dim))

    def test_get_clip_embeddings_accepts_optional_clip_modalities(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        clip_text_pooled = floats_tensor((2, 1, 8)).to(torch_device)
        clip_text = floats_tensor((2, 3, 8)).to(torch_device)
        clip_img = floats_tensor((2, 8)).to(torch_device)

        text_only_clip = model.get_clip_embeddings(clip_text_pooled, clip_txt=clip_text)
        image_only_clip = model.get_clip_embeddings(clip_text_pooled, clip_img=clip_img)

        self.assertEqual(text_only_clip.shape, (2, 3 + model.config.clip_seq, model.config.conditioning_dim))
        self.assertEqual(image_only_clip.shape, (2, 2 * model.config.clip_seq, model.config.conditioning_dim))

    def test_forward_accepts_batched_sca_crp(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        inputs_dict["sca"] = torch.tensor([0.1, 0.2], device=torch_device)
        inputs_dict["crp"] = torch.tensor([0.3, 0.4], device=torch_device)

        with torch.no_grad():
            output = model(**inputs_dict).sample

        self.assertEqual(output.shape, (2,) + self.output_shape)

    def test_forward_uses_configured_pixel_mapper_channels(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["pixel_mapper_in_channels"] = 5
        model = self.model_class(**init_dict).to(torch_device)

        with torch.no_grad():
            output = model(**inputs_dict).sample

        self.assertEqual(output.shape, (2,) + self.output_shape)

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"StableCascadeUNet"})
