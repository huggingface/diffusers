# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import numpy as np
import torch

from diffusers import RePaintPipeline, RePaintScheduler, UNet2DModel
from diffusers.utils.testing_utils import require_torch, slow


torch.backends.cuda.matmul.allow_tf32 = False


@slow
@require_torch
class RepaintPipelineIntegrationTests(unittest.TestCase):
    def test_celebahq(self):
        from datasets import load_dataset

        dataset = load_dataset("huggan/CelebA-HQ", split="train", streaming=True)
        original_image = next(iter(dataset))["image"].resize((256, 256))
        original_image = torch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0)
        original_image = (original_image / 255.0) * 2 - 1
        mask = torch.zeros_like(original_image)
        mask[:, :, :128, :] = 1  # mask the top half of the image

        model_id = "google/ddpm-ema-celebahq-256"
        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = RePaintScheduler.from_config(model_id)

        repaint = RePaintPipeline(unet=unet, scheduler=scheduler).to("cuda")

        generator = torch.manual_seed(0)
        image = repaint(
            original_image,
            mask,
            num_inference_steps=250,
            eta=0.0,
            jump_length=10,
            jump_n_sample=10,
            generator=generator,
            output_type="numpy",
        ).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [0.14537135, 0.10728511, 0.08822048, 0.15828621, 0.11806837, 0.11007798, 0.15231332, 0.1214554, 0.15475643]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
