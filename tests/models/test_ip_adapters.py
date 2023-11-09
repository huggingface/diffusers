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

import gc
import unittest

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    nightly,
    require_torch_gpu,
    torch_device,
)


enable_full_determinism()


@nightly
@require_torch_gpu
class IPAdapterSDIntegrationTests(unittest.TestCase):
    dtype = torch.float16

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_image_encoder(self, repo_id, subfolder):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            repo_id, subfolder=subfolder, torch_dtype=self.dtype
        ).to(torch_device)
        return image_encoder

    def get_image_processor(self, repo_id):
        image_processor = CLIPImageProcessor.from_pretrained(repo_id)
        return image_processor

    def get_dummy_inputs(self):
        image = load_image(
            "https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png"
        )
        input_kwargs = {
            "prompt": "best quality, high quality",
            "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
            "num_inference_steps": 5,
            "generator": torch.Generator(device="cpu").manual_seed(33),
            "ip_adapter_image": image,
            "output_type": "np",
        }
        return input_kwargs

    def test_text_to_image(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        slice = image_slice.tolist()
        print(", ".join([str(round(x, 4)) for x in slice]))

        expected_slice = np.array([list(range(9))]).astype("float32")

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)
