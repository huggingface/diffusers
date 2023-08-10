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

from diffusers import Prompt2PromptPipeline, DDIMScheduler, UNet2DModel
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device


enable_full_determinism()

replace_steps = {
    "cross_replace_steps": 0.4,
    "self_replace_steps": 0.4
}


class Prompt2PrompteFastTests(unittest.TestCase):
    @property
    def dummy_uncond_unet(self):
        # TODO: Use conditional model
        torch.manual_seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    test_matrix = [
        # fmt: off
        (
            ["A turtle playing with a ball", "A monkey playing with a ball"],
            "replace",
            {**replace_steps},
            [9.956e-01, 5.785e-01, 4.675e-01, 9.930e-01, 0.0, 1.000, 1.199e-03, 2.648e-04, 5.101e-04] # todo: adapt
        ), 
        (
            ["A turtle playing with a ball", "A monkey playing with a ball"],
            "replace",
            {**replace_steps, "local_blend_words": ["turtle", "monkey"]},
            [9.956e-01, 5.785e-01, 4.675e-01, 9.930e-01, 0.0, 1.000, 1.199e-03, 2.648e-04, 5.101e-04] # todo: adapt
        ), 
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps},
            [9.956e-01, 5.785e-01, 4.675e-01, 9.930e-01, 0.0, 1.000, 1.199e-03, 2.648e-04, 5.101e-04] # todo: adapt
        ),
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps, "local_blend_words": ["in", "a" , "forest"]},
            [9.956e-01, 5.785e-01, 4.675e-01, 9.930e-01, 0.0, 1.000, 1.199e-03, 2.648e-04, 5.101e-04] # todo: adapt
        ), 
        (
            ["A smiling turtle"] * 2,
            "reweight",
            {**replace_steps, "equalizer_words": ["smiling"], "equalizer_strengths": [5]},
            [9.956e-01, 5.785e-01, 4.675e-01, 9.930e-01, 0.0, 1.000, 1.199e-03, 2.648e-04, 5.101e-04] # todo: adapt
        ), 
        # todo: include save edit
        # fmt: on
    ]

    @parameterized.expand(test_matrix)
    def test_fast_inference(self, prompts, edit_type, edit_kwargs, expected_slice):
        device = "cpu"
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()

        pipe = Prompt2PromptPipeline(unet=unet, scheduler=scheduler)
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
            [0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385] # todo: adapt
        ), 
        (
            ["A turtle playing with a ball", "A monkey playing with a ball"],
            "replace",
            {**replace_steps, "local_blend_words": ["turtle", "monkey"]},
            [0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385] # todo: adapt
        ), 
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps},
            [0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385] # todo: adapt
        ),
        (
            ["A turtle", "A turtle in a forest"],
            "refine",
            {**replace_steps, "local_blend_words": ["in", "a" , "forest"]},
            [0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385] # todo: adapt
        ), 
        (
            ["A smiling turtle"] * 2,
            "reweight",
            {**replace_steps, "equalizer_words": ["smiling"], "equalizer_strengths": [5]},
            [0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385] # todo: adapt
        ), 
        # todo: include save edit
        # fmt: on
    ]

    @parameterized.expand(test_matrix)
    def test_inference(self, prompts, edit_type, edit_kwargs, expected_slice):
        model_id = "CompVis/stable-diffusion-v1-4"  # TODO: Q: Use smaller model?

        pipe = Prompt2PromptPipeline.from_pretrained(model_id)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = pipe(prompts, height=512, width=512, num_inference_steps=2, generator=generator, edit_type=edit_type, edit_kwargs=edit_kwargs, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (2, 512, 512, 3)
        expected_slice = np.array([0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
