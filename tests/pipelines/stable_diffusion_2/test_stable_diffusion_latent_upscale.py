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
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import floats_tensor, load_image, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class StableDiffusionLatentUpscalePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionLatentUpscalePipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "height",
        "width",
        "cross_attention_kwargs",
        "negative_prompt_embeds",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {"num_images_per_prompt"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    test_cpu_offload = True

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 4
        sizes = (16, 16)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0)).to(torch_device)
        return image

    def get_dummy_components(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            act_fn="gelu",
            attention_head_dim=8,
            norm_num_groups=None,
            block_out_channels=[32, 32, 64, 64],
            time_cond_proj_dim=160,
            conv_in_kernel=1,
            conv_out_kernel=1,
            cross_attention_dim=32,
            down_block_types=(
                "KDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
            ),
            in_channels=8,
            mid_block_type=None,
            only_cross_attention=False,
            out_channels=5,
            resnet_time_scale_shift="scale_shift",
            time_embedding_type="fourier",
            timestep_post_act="gelu",
            up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
        )
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 64, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        scheduler = EulerDiscreteScheduler(prediction_type="sample")
        text_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="quick_gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": model.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": self.dummy_image.cpu(),
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "numpy",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 256, 256, 3))
        expected_slice = np.array(
            [0.47222412, 0.41921633, 0.44717434, 0.46874192, 0.42588258, 0.46150726, 0.4677534, 0.45583832, 0.48579055]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(relax_max_difference=False)


@require_torch_gpu
@slow
class StableDiffusionLatentUpscalePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_latent_upscaler_fp16(self):
        generator = torch.manual_seed(33)

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.to("cuda")

        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        )
        upscaler.to("cuda")

        prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"

        low_res_latents = pipe(prompt, generator=generator, output_type="latent").images

        image = upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/astronaut_1024.npy"
        )
        assert np.abs((expected_image - image).mean()) < 5e-2

    def test_latent_upscaler_fp16_image(self):
        generator = torch.manual_seed(33)

        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        )
        upscaler.to("cuda")

        prompt = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas"

        low_res_img = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_512.png"
        )

        image = upscaler(
            prompt=prompt,
            image=low_res_img,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_1024.npy"
        )
        assert np.abs((expected_image - image).max()) < 5e-2
