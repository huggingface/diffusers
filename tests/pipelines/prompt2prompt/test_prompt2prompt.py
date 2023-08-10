# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
from parameterized import parameterized

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import Prompt2PromptPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device


enable_full_determinism()

replace_steps = {
    "cross_replace_steps": 0.4,
    "self_replace_steps": 0.4
}


class Prompt2PrompteFastTests(unittest.TestCase):

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components


    test_matrix = [
        # fmt: off
        (
            ["A turtle playing with a ball", "A monkey playing with a ball"],
            "replace",
            {**replace_steps},
            [0.582, 0.610, 0.503, 0.507, 0.542, 0.471, 0.498, 0.490, 0.487]
        ), 
        (
            ["A turtle playing with a ball", "A monkey playing with a ball"],
            "replace",
            {**replace_steps, "local_blend_words": ["turtle", "monkey"]},
            [0.582, 0.610, 0.503, 0.507, 0.542, 0.471, 0.498, 0.490, 0.487]
        ), 
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps},
            [0.571, 0.605, 0.499, 0.502, 0.541, 0.468, 0.500, 0.484, 0.483]
        ),
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps, "local_blend_words": ["in", "a" , "forest"]},
            [0.571, 0.605, 0.499, 0.502, 0.541, 0.468, 0.500, 0.484, 0.483]
        ), 
        (
            ["A smiling turtle"] * 2,
            "reweight",
            {**replace_steps, "equalizer_words": ["smiling"], "equalizer_strengths": [5]},
            [0.573, 0.607, 0.502, 0.504, 0.540, 0.469, 0.500, 0.486, 0.483]
        ), 
        # todo: include save edit?
        # fmt: on
    ]

    @parameterized.expand(test_matrix)
    def test_fast_inference(self, prompts, edit_type, edit_kwargs, expected_slice):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = Prompt2PromptPipeline(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        image = pipe(prompts, height=64, width=64, num_inference_steps=2, generator=generator, edit_type=edit_type, edit_kwargs=edit_kwargs, output_type="numpy").images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = pipe(prompts, height=64, width=64, num_inference_steps=2, generator=generator, edit_type=edit_type, edit_kwargs=edit_kwargs, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (2, 64, 64, 3)
        expected_slice = np.array(expected_slice)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
@require_torch_gpu
class Prompt2PromptIntegrationTests(unittest.TestCase):
    test_matrix = [
        # fmt: off
        (
            ["A turtle playing with a ball", "A monkey playing with a ball"],
            "replace",
            {**replace_steps},
            [0.243, 0.233, 0.227, 0.253, 0.242, 0.237, 0.293, 0.292, 0.283]
        ), 
        (
            ["A turtle playing with a ball", "A monkey playing with a ball"],
            "replace",
            {**replace_steps, "local_blend_words": ["turtle", "monkey"]},
            [0.243, 0.233, 0.227, 0.253, 0.242, 0.237, 0.293, 0.292, 0.283]
        ), 
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps},
            [0.256, 0.232, 0.209, 0.259, 0.254, 0.229, 0.285, 0.307, 0.295]
        ),
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps, "local_blend_words": ["in", "a" , "forest"]},
            [0.256, 0.232, 0.209, 0.259, 0.254, 0.229, 0.285, 0.307, 0.295]
        ), 
        (
            ["A smiling turtle"] * 2,
            "reweight",
            {**replace_steps, "equalizer_words": ["smiling"], "equalizer_strengths": [5]},
            [0.006, 0.010, 0.009, 0.003, 0.011, 0.008, 0.014, 0.009, 0.000]
        ), 
        # todo: include save edit?
        # fmt: on
    ]

    @parameterized.expand(test_matrix)
    def test_inference(self, prompts, edit_type, edit_kwargs, expected_slice):
        model_id = "CompVis/stable-diffusion-v1-4"  # TODO: Q: Use smaller model?

        pipe = Prompt2PromptPipeline.from_pretrained(model_id)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator().manual_seed(0)
        image = pipe(prompts, height=512, width=512, num_inference_steps=50, generator=generator, edit_type=edit_type, edit_kwargs=edit_kwargs, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (2, 512, 512, 3)
        expected_slice = np.array([0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
