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

from diffusers import FluxTransformer2DModel
from diffusers.utils.testing_utils import enable_full_determinism, torch_device

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class FluxTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = FluxTransformer2DModel
    main_input_name = "hidden_states"
    # We override the items here because the transformer under consideration is small.
    model_split_percents = [0.7, 0.6, 0.6]

    # Skip setting testing with default: AttnProcessor
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        pooled_prompt_embeds = torch.randn((batch_size, embedding_dim)).to(torch_device)
        text_ids = torch.randn((sequence_length, num_image_channels)).to(torch_device)
        image_ids = torch.randn((height * width, num_image_channels)).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep,
        }

    @property
    def input_shape(self):
        return (16, 4)

    @property
    def output_shape(self):
        return (16, 4)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 32,
            "axes_dims_rope": [4, 4, 8],
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_deprecated_inputs_img_txt_ids_3d(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output_1 = model(**inputs_dict).to_tuple()[0]

        # update inputs_dict with txt_ids and img_ids as 3d tensors (deprecated)
        text_ids_3d = inputs_dict["txt_ids"].unsqueeze(0)
        image_ids_3d = inputs_dict["img_ids"].unsqueeze(0)

        assert text_ids_3d.ndim == 3, "text_ids_3d should be a 3d tensor"
        assert image_ids_3d.ndim == 3, "img_ids_3d should be a 3d tensor"

        inputs_dict["txt_ids"] = text_ids_3d
        inputs_dict["img_ids"] = image_ids_3d

        with torch.no_grad():
            output_2 = model(**inputs_dict).to_tuple()[0]

        self.assertEqual(output_1.shape, output_2.shape)
        self.assertTrue(
            torch.allclose(output_1, output_2, atol=1e-5),
            msg="output with deprecated inputs (img_ids and txt_ids as 3d torch tensors) are not equal as them as 2d inputs",
        )
