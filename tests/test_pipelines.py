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

import tempfile
import unittest

import numpy as np
import torch

import PIL
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    KarrasVePipeline,
    KarrasVeScheduler,
    LDMPipeline,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMPipeline,
    PNDMScheduler,
    ScoreSdeVePipeline,
    ScoreSdeVeScheduler,
    StableDiffusionPipeline,
    UNet2DModel,
)
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.testing_utils import slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class PipelineTesterMixin(unittest.TestCase):
    def test_from_pretrained_save_pretrained(self):
        # 1. Load models
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        schedular = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline(model, schedular)
        ddpm.to(torch_device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)
            new_ddpm.to(torch_device)

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator, output_type="numpy")["sample"]
        generator = generator.manual_seed(0)
        new_image = new_ddpm(generator=generator, output_type="numpy")["sample"]

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub.to(torch_device)

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator, output_type="numpy")["sample"]
        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy")["sample"]

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub_pass_model(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        # pass unet into DiffusionPipeline
        unet = UNet2DModel.from_pretrained(model_path)
        ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(model_path, unet=unet, scheduler=scheduler)
        ddpm_from_hub_custom_model.to(torch_device)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub.to(torch_device)

        generator = torch.manual_seed(0)

        image = ddpm_from_hub_custom_model(generator=generator, output_type="numpy")["sample"]
        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy")["sample"]

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"

        pipe = DDIMPipeline.from_pretrained(model_path)
        pipe.to(torch_device)

        generator = torch.manual_seed(0)
        images = pipe(generator=generator, output_type="numpy")["sample"]
        assert images.shape == (1, 32, 32, 3)
        assert isinstance(images, np.ndarray)

        images = pipe(generator=generator, output_type="pil")["sample"]
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)

        # use PIL by default
        images = pipe(generator=generator)["sample"]
        assert isinstance(images, list)
        assert isinstance(images[0], PIL.Image.Image)

    @slow
    def test_ddpm_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_config(model_id)
        scheduler = scheduler.set_format("pt")

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.41995, 0.35885, 0.19385, 0.38475, 0.3382, 0.2647, 0.41545, 0.3582, 0.33845])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddim_lsun(self):
        model_id = "google/ddpm-ema-bedroom-256"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler.from_config(model_id)

        ddpm = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.00605, 0.0201, 0.0344, 0.00235, 0.00185, 0.00025, 0.00215, 0.0, 0.00685])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddim_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler(tensor_format="pt")

        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddim.to(torch_device)

        generator = torch.manual_seed(0)
        image = ddim(generator=generator, eta=0.0, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.17235, 0.16175, 0.16005, 0.16255, 0.1497, 0.1513, 0.15045, 0.1442, 0.1453])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_pndm_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = PNDMScheduler(tensor_format="pt")

        pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
        pndm.to(torch_device)
        generator = torch.manual_seed(0)
        image = pndm(generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.1564, 0.14645, 0.1406, 0.14715, 0.12425, 0.14045, 0.13115, 0.12175, 0.125])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_text2img(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.to(torch_device)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="numpy")[
            "sample"
        ]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.9256, 0.9340, 0.8933, 0.9361, 0.9113, 0.8727, 0.9122, 0.8745, 0.8099])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_text2img_fast(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.to(torch_device)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm(prompt, generator=generator, num_inference_steps=1, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.3163, 0.8670, 0.6465, 0.1865, 0.6291, 0.5139, 0.2824, 0.3723, 0.4344])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion(self):
        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1").to(torch_device)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast("cuda"):
            output = sd_pipe(
                [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="np"
            )

        image = output["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.8887, 0.915, 0.91, 0.894, 0.909, 0.912, 0.919, 0.925, 0.883])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_fast_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1").to(torch_device)

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        sd_pipe.scheduler = scheduler

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)

        with torch.autocast("cuda"):
            output = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="numpy")
        image = output["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.8354, 0.83, 0.866, 0.838, 0.8315, 0.867, 0.836, 0.8584, 0.869])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @slow
    def test_score_sde_ve_pipeline(self):
        model_id = "google/ncsnpp-church-256"
        model = UNet2DModel.from_pretrained(model_id)

        scheduler = ScoreSdeVeScheduler.from_config(model_id)

        sde_ve = ScoreSdeVePipeline(unet=model, scheduler=scheduler)
        sde_ve.to(torch_device)

        torch.manual_seed(0)
        image = sde_ve(num_inference_steps=300, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)

        expected_slice = np.array([0.64363, 0.5868, 0.3031, 0.2284, 0.7409, 0.3216, 0.25643, 0.6557, 0.2633])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_uncond(self):
        ldm = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        ldm.to(torch_device)

        generator = torch.manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=5, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.4399, 0.44975, 0.46825, 0.474, 0.4359, 0.4581, 0.45095, 0.4341, 0.4447])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddpm_ddim_equality(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler(tensor_format="pt")
        ddim_scheduler = DDIMScheduler(tensor_format="pt")

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddpm.to(torch_device)
        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
        ddim.to(torch_device)

        generator = torch.manual_seed(0)
        ddpm_image = ddpm(generator=generator, output_type="numpy")["sample"]

        generator = torch.manual_seed(0)
        ddim_image = ddim(generator=generator, num_inference_steps=1000, eta=1.0, output_type="numpy")["sample"]

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_image - ddim_image).max() < 1e-1

    @unittest.skip("(Anton) The test is failing for large batch sizes, needs investigation")
    def test_ddpm_ddim_equality_batched(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler(tensor_format="pt")
        ddim_scheduler = DDIMScheduler(tensor_format="pt")

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddpm.to(torch_device)

        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
        ddim.to(torch_device)

        generator = torch.manual_seed(0)
        ddpm_images = ddpm(batch_size=4, generator=generator, output_type="numpy")["sample"]

        generator = torch.manual_seed(0)
        ddim_images = ddim(batch_size=4, generator=generator, num_inference_steps=1000, eta=1.0, output_type="numpy")[
            "sample"
        ]

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_images - ddim_images).max() < 1e-1

    @slow
    def test_karras_ve_pipeline(self):
        model_id = "google/ncsnpp-celebahq-256"
        model = UNet2DModel.from_pretrained(model_id)
        scheduler = KarrasVeScheduler(tensor_format="pt")

        pipe = KarrasVePipeline(unet=model, scheduler=scheduler)
        pipe.to(torch_device)

        generator = torch.manual_seed(0)
        image = pipe(num_inference_steps=20, generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.26815, 0.1581, 0.2658, 0.23248, 0.1550, 0.2539, 0.1131, 0.1024, 0.0837])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_lms_stable_diffusion_pipeline(self):
        model_id = "CompVis/stable-diffusion-v1-1"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(torch_device)
        scheduler = LMSDiscreteScheduler.from_config(model_id, subfolder="scheduler", use_auth_token=True)
        pipe.scheduler = scheduler

        prompt = "a photograph of an astronaut riding a horse"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = pipe([prompt], generator=generator, guidance_scale=7.5, num_inference_steps=10, output_type="numpy")[
            "sample"
        ]

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9077, 0.9254, 0.9181, 0.9227, 0.9213, 0.9367, 0.9399, 0.9406, 0.9024])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
