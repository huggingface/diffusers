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

import gc
import random
import unittest

import numpy as np
import torch

from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionRepaintPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from diffusers.utils import floats_tensor, load_image, slow, torch_device
from diffusers.utils.testing_utils import load_numpy, require_torch_gpu
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


torch.backends.cuda.matmul.allow_tf32 = False


@slow
@require_torch_gpu
class StableDiffusionRepaintPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_repaint_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint"
            "/red_cat_sitting_on_a_park_bench_repaint.npy"
        )

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionRepaintPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.scheduler = RePaintScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A red cat sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            jump_length=3,
            jump_n_sample=3,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-3
