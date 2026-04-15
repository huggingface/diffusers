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

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lora_to_diffusers
from diffusers.modular_pipelines.flux2.encoders import Flux2KleinTextEncoderStep


class Flux2LoraConversionTests(unittest.TestCase):
    def test_convert_non_diffusers_flux2_lora_maps_guidance_embedder(self):
        state_dict = {
            "diffusion_model.img_in.lora_A.weight": torch.randn(2, 2),
            "diffusion_model.img_in.lora_B.weight": torch.randn(2, 2),
            "diffusion_model.txt_in.lora_A.weight": torch.randn(2, 2),
            "diffusion_model.txt_in.lora_B.weight": torch.randn(2, 2),
            "diffusion_model.time_in.in_layer.lora_A.weight": torch.randn(2, 2),
            "diffusion_model.time_in.in_layer.lora_B.weight": torch.randn(2, 2),
            "diffusion_model.time_in.out_layer.lora_A.weight": torch.randn(2, 2),
            "diffusion_model.time_in.out_layer.lora_B.weight": torch.randn(2, 2),
            "diffusion_model.guidance_in.in_layer.lora_A.weight": torch.randn(2, 2),
            "diffusion_model.guidance_in.in_layer.lora_B.weight": torch.randn(2, 2),
            "diffusion_model.guidance_in.out_layer.lora_A.weight": torch.randn(2, 2),
            "diffusion_model.guidance_in.out_layer.lora_B.weight": torch.randn(2, 2),
        }

        converted_state_dict = _convert_non_diffusers_flux2_lora_to_diffusers(state_dict)

        expected_keys = {
            "transformer.x_embedder.lora_A.weight",
            "transformer.x_embedder.lora_B.weight",
            "transformer.context_embedder.lora_A.weight",
            "transformer.context_embedder.lora_B.weight",
            "transformer.time_guidance_embed.timestep_embedder.linear_1.lora_A.weight",
            "transformer.time_guidance_embed.timestep_embedder.linear_1.lora_B.weight",
            "transformer.time_guidance_embed.timestep_embedder.linear_2.lora_A.weight",
            "transformer.time_guidance_embed.timestep_embedder.linear_2.lora_B.weight",
            "transformer.time_guidance_embed.guidance_embedder.linear_1.lora_A.weight",
            "transformer.time_guidance_embed.guidance_embedder.linear_1.lora_B.weight",
            "transformer.time_guidance_embed.guidance_embedder.linear_2.lora_A.weight",
            "transformer.time_guidance_embed.guidance_embedder.linear_2.lora_B.weight",
        }

        self.assertEqual(set(converted_state_dict.keys()), expected_keys)

    def test_flux2_text_subpipeline_rejects_transformer_lora_loading(self):
        text_pipe = Flux2KleinTextEncoderStep().init_pipeline()

        with self.assertRaisesRegex(ValueError, "defines a `transformer` component"):
            text_pipe.load_lora_weights({})
