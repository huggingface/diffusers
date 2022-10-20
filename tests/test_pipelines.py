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
import os
import random
import tempfile
import tracemalloc
import unittest

import numpy as np
import torch

import accelerate
import PIL
import transformers
from diffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    KarrasVePipeline,
    KarrasVeScheduler,
    LDMPipeline,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionPipeline,
    PNDMPipeline,
    PNDMScheduler,
    ScoreSdeVePipeline,
    ScoreSdeVeScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, WEIGHTS_NAME, floats_tensor, load_image, slow, torch_device
from diffusers.utils.testing_utils import get_tests_dir
from packaging import version
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTextConfig, CLIPTextModel, CLIPTokenizer


torch.backends.cuda.matmul.allow_tf32 = False


def test_progress_bar(capsys):
    model = UNet2DModel(
        block_out_channels=(32, 64),
        layers_per_block=2,
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=10)

    ddpm = DDPMPipeline(model, scheduler).to(torch_device)
    ddpm(output_type="numpy").images
    captured = capsys.readouterr()
    assert "10/10" in captured.err, "Progress bar has to be displayed"

    ddpm.set_progress_bar_config(disable=True)
    ddpm(output_type="numpy").images
    captured = capsys.readouterr()
    assert captured.err == "", "Progress bar should be disabled"


class CustomPipelineTests(unittest.TestCase):
    def test_load_custom_pipeline(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline"
        )
        # NOTE that `"CustomPipeline"` is not a class that is defined in this library, but solely on the Hub
        # under https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py#L24
        assert pipeline.__class__.__name__ == "CustomPipeline"

    def test_run_custom_pipeline(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline"
        )
        images, output_str = pipeline(num_inference_steps=2, output_type="np")

        assert images[0].shape == (1, 32, 32, 3)
        # compare output to https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py#L102
        assert output_str == "This is a test"

    def test_local_custom_pipeline(self):
        local_custom_pipeline_path = get_tests_dir("fixtures/custom_pipeline")
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline=local_custom_pipeline_path
        )
        images, output_str = pipeline(num_inference_steps=2, output_type="np")

        assert pipeline.__class__.__name__ == "CustomLocalPipeline"
        assert images[0].shape == (1, 32, 32, 3)
        # compare to https://github.com/huggingface/diffusers/blob/main/tests/fixtures/custom_pipeline/pipeline.py#L102
        assert output_str == "This is a local test"

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_load_pipeline_from_git(self):
        clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

        feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
        clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            custom_pipeline="clip_guided_stable_diffusion",
            clip_model=clip_model,
            feature_extractor=feature_extractor,
            torch_dtype=torch.float16,
            revision="fp16",
        )
        pipeline.enable_attention_slicing()
        pipeline = pipeline.to(torch_device)

        # NOTE that `"CLIPGuidedStableDiffusion"` is not a class that is defined in the pypi package of th e library, but solely on the community examples folder of GitHub under:
        # https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py
        assert pipeline.__class__.__name__ == "CLIPGuidedStableDiffusion"

        image = pipeline("a prompt", num_inference_steps=2, output_type="np").images[0]
        assert image.shape == (512, 512, 3)


class PipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0)).to(torch_device)
        return image

    @property
    def dummy_uncond_unet(self):
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

    @property
    def dummy_cond_unet(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_cond_unet_inpaint(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_vq_model(self):
        torch.manual_seed(0)
        model = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
        )
        return model

    @property
    def dummy_vae(self):
        torch.manual_seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        return model

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModel(config)

    @property
    def dummy_safety_checker(self):
        def check(images, *args, **kwargs):
            return images, [False] * len(images)

        return check

    @property
    def dummy_extractor(self):
        def extract(*args, **kwargs):
            class Out:
                def __init__(self):
                    self.pixel_values = torch.ones([0])

                def to(self, device):
                    self.pixel_values.to(device)
                    return self

            return Out()

        return extract

    def test_ddim(self):
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()

        ddpm = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = ddpm(num_inference_steps=1)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = torch.manual_seed(0)
        image_from_tuple = ddpm(generator=generator, num_inference_steps=2, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [1.000e00, 5.717e-01, 4.717e-01, 1.000e00, 0.000e00, 1.000e00, 3.000e-04, 0.000e00, 9.000e-04]
        )
        tolerance = 1e-2 if torch_device != "mps" else 3e-2
        assert np.abs(image_slice.flatten() - expected_slice).max() < tolerance
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < tolerance

    def test_pndm_cifar10(self):
        unet = self.dummy_uncond_unet
        scheduler = PNDMScheduler()

        pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
        pndm.to(torch_device)
        pndm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = pndm(generator=generator, num_inference_steps=20, output_type="numpy").images

        generator = torch.manual_seed(0)
        image_from_tuple = pndm(generator=generator, num_inference_steps=20, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_ldm_text2img(self):
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        ldm = LDMTextToImagePipeline(vqvae=vae, bert=bert, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            generator = torch.manual_seed(0)
            _ = ldm(
                [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=1, output_type="numpy"
            ).images

        generator = torch.manual_seed(0)
        image = ldm(
            [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="numpy"
        ).images

        generator = torch.manual_seed(0)
        image_from_tuple = ldm(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="numpy",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5074, 0.5026, 0.4998, 0.4056, 0.3523, 0.4649, 0.5289, 0.5299, 0.4897])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")
        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.5112, 0.4692, 0.4715, 0.5206, 0.4894, 0.5114, 0.5096, 0.4932, 0.4755])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_ddim_factor_8(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            height=536,
            width=536,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 134, 134, 3)
        expected_slice = np.array([0.7834, 0.5488, 0.5781, 0.46, 0.3609, 0.5369, 0.542, 0.4855, 0.5557])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_pndm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.4937, 0.4649, 0.4716, 0.5145, 0.4889, 0.513, 0.513, 0.4905, 0.4738])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_from_pretrained_error_message_uninstalled_packages(self):
        # TODO(Patrick, Pedro) - need better test here for the future
        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-lms-pipe")
        assert isinstance(pipe, StableDiffusionPipeline)
        assert isinstance(pipe.scheduler, LMSDiscreteScheduler)

    def test_stable_diffusion_no_safety_checker(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-lms-pipe", safety_checker=None
        )
        assert isinstance(pipe, StableDiffusionPipeline)
        assert isinstance(pipe.scheduler, LMSDiscreteScheduler)
        assert pipe.safety_checker is None

        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

    def test_stable_diffusion_k_lms(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.5067, 0.4689, 0.4614, 0.5233, 0.4903, 0.5112, 0.524, 0.5069, 0.4785])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_attention_chunk(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output_1 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        # make sure chunking the attention yields the same result
        sd_pipe.enable_attention_slicing(slice_size=1)
        generator = torch.Generator(device=device).manual_seed(0)
        output_2 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 1e-4

    def test_stable_diffusion_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        negative_prompt = "french fries"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.4851, 0.4617, 0.4765, 0.5127, 0.4845, 0.5153, 0.5141, 0.4886, 0.4719])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_score_sde_ve_pipeline(self):
        unet = self.dummy_uncond_unet
        scheduler = ScoreSdeVeScheduler()

        sde_ve = ScoreSdeVePipeline(unet=unet, scheduler=scheduler)
        sde_ve.to(torch_device)
        sde_ve.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = sde_ve(num_inference_steps=2, output_type="numpy", generator=generator).images

        generator = torch.manual_seed(0)
        image_from_tuple = sde_ve(num_inference_steps=2, output_type="numpy", generator=generator, return_dict=False)[
            0
        ]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_ldm_uncond(self):
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()
        vae = self.dummy_vq_model

        ldm = LDMPipeline(unet=unet, vqvae=vae, scheduler=scheduler)
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            generator = torch.manual_seed(0)
            _ = ldm(generator=generator, num_inference_steps=1, output_type="numpy").images

        generator = torch.manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = torch.manual_seed(0)
        image_from_tuple = ldm(generator=generator, num_inference_steps=2, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.8512, 0.818, 0.6411, 0.6808, 0.4465, 0.5618, 0.46, 0.6231, 0.5172])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_karras_ve_pipeline(self):
        unet = self.dummy_uncond_unet
        scheduler = KarrasVeScheduler()

        pipe = KarrasVePipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = pipe(num_inference_steps=2, generator=generator, output_type="numpy").images

        generator = torch.manual_seed(0)
        image_from_tuple = pipe(num_inference_steps=2, generator=generator, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_img2img(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image.to(device)

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        )

        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4492, 0.3865, 0.4222, 0.5854, 0.5139, 0.4379, 0.4193, 0.48, 0.4218])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_img2img_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image.to(device)

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        negative_prompt = "french fries"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        )
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4065, 0.3783, 0.4050, 0.5266, 0.4781, 0.4252, 0.4203, 0.4692, 0.4365])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_img2img_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image.to(device).repeat(2, 1, 1, 1)

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = 2 * ["A painting of a squirrel eating a burger"]
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            prompt,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        )

        image = output.images

        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array([0.5144, 0.4447, 0.4735, 0.6676, 0.5526, 0.5454, 0.645, 0.5149, 0.4689])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_img2img_k_lms(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image.to(device)

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        )
        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            return_dict=False,
        )
        image_from_tuple = output[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4367, 0.4986, 0.4372, 0.6706, 0.5665, 0.444, 0.5864, 0.6019, 0.5203])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((128, 128))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
        )

        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4731, 0.5346, 0.4531, 0.6251, 0.5446, 0.4057, 0.5527, 0.5896, 0.5153])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet_inpaint
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((128, 128))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((128, 128))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        )

        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.5075, 0.4485, 0.4558, 0.5369, 0.5369, 0.5236, 0.5127, 0.4983, 0.4776])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((128, 128))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        negative_prompt = "french fries"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4765, 0.5339, 0.4541, 0.6240, 0.5439, 0.4055, 0.5503, 0.5891, 0.5150])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_num_images_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # test num_images_per_prompt=1 (default)
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images

        assert images.shape == (1, 128, 128, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe([prompt] * batch_size, num_inference_steps=2, output_type="np").images

        assert images.shape == (batch_size, 128, 128, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images

        assert images.shape == (num_images_per_prompt, 128, 128, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 128, 128, 3)

    def test_stable_diffusion_img2img_num_images_per_prompt(self):
        device = "cpu"
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image.to(device)

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # test num_images_per_prompt=1 (default)
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        ).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        ).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (num_images_per_prompt, 32, 32, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)

    def test_stable_diffusion_inpaint_legacy_num_images_per_prompt(self):
        device = "cpu"
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((128, 128))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # test num_images_per_prompt=1 (default)
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (num_images_per_prompt, 32, 32, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)

    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")
    def test_stable_diffusion_fp16(self):
        """Test that stable diffusion works with fp16"""
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # put models in fp16
        unet = unet.half()
        vae = vae.half()
        bert = bert.half()

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="np").images

        assert image.shape == (1, 128, 128, 3)

    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")
    def test_stable_diffusion_img2img_fp16(self):
        """Test that stable diffusion img2img works with fp16"""
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image.to(torch_device)

        # put models in fp16
        unet = unet.half()
        vae = vae.half()
        bert = bert.half()

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = sd_pipe(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        ).images

        assert image.shape == (1, 32, 32, 3)

    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")
    def test_stable_diffusion_inpaint_fp16(self):
        """Test that stable diffusion inpaint_legacy works with fp16"""
        unet = self.dummy_cond_unet_inpaint
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((128, 128))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((128, 128))

        # put models in fp16
        unet = unet.half()
        vae = vae.half()
        bert = bert.half()

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = sd_pipe(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images

        assert image.shape == (1, 128, 128, 3)


class PipelineTesterMixin(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_smart_download(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = DiffusionPipeline.from_pretrained(model_id, cache_dir=tmpdirname, force_download=True)
            local_repo_name = "--".join(["models"] + model_id.split("/"))
            snapshot_dir = os.path.join(tmpdirname, local_repo_name, "snapshots")
            snapshot_dir = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])

            # inspect all downloaded files to make sure that everything is included
            assert os.path.isfile(os.path.join(snapshot_dir, DiffusionPipeline.config_name))
            assert os.path.isfile(os.path.join(snapshot_dir, CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "scheduler", SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            # let's make sure the super large numpy file:
            # https://huggingface.co/hf-internal-testing/unet-pipeline-dummy/blob/main/big_array.npy
            # is not downloaded, but all the expected ones
            assert not os.path.isfile(os.path.join(snapshot_dir, "big_array.npy"))

    @property
    def dummy_safety_checker(self):
        def check(images, *args, **kwargs):
            return images, [False] * len(images)

        return check

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
        ddpm.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)
            new_ddpm.to(torch_device)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        generator = generator.manual_seed(0)
        new_image = new_ddpm(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub.to(torch_device)
        ddpm_from_hub.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub_pass_model(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        # pass unet into DiffusionPipeline
        unet = UNet2DModel.from_pretrained(model_path)
        ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(model_path, unet=unet, scheduler=scheduler)
        ddpm_from_hub_custom_model.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm_from_hub_custom_model(generator=generator, output_type="numpy").images

        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"

        pipe = DDIMPipeline.from_pretrained(model_path)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        images = pipe(generator=generator, output_type="numpy").images
        assert images.shape == (1, 32, 32, 3)
        assert isinstance(images, np.ndarray)

        images = pipe(generator=generator, output_type="pil").images
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)

        # use PIL by default
        images = pipe(generator=generator).images
        assert isinstance(images, list)
        assert isinstance(images[0], PIL.Image.Image)

    @slow
    def test_ddpm_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_config(model_id)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

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
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.00605, 0.0201, 0.0344, 0.00235, 0.00185, 0.00025, 0.00215, 0.0, 0.00685])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddim_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler()

        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddim(generator=generator, eta=0.0, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.17235, 0.16175, 0.16005, 0.16255, 0.1497, 0.1513, 0.15045, 0.1442, 0.1453])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_pndm_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = PNDMScheduler()

        pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
        pndm.to(torch_device)
        pndm.set_progress_bar_config(disable=None)
        generator = torch.manual_seed(0)
        image = pndm(generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.1564, 0.14645, 0.1406, 0.14715, 0.12425, 0.14045, 0.13115, 0.12175, 0.125])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_text2img(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm(
            [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="numpy"
        ).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.9256, 0.9340, 0.8933, 0.9361, 0.9113, 0.8727, 0.9122, 0.8745, 0.8099])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_text2img_fast(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm(prompt, generator=generator, num_inference_steps=1, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.3163, 0.8670, 0.6465, 0.1865, 0.6291, 0.5139, 0.2824, 0.3723, 0.4344])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion(self):
        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast("cuda"):
            output = sd_pipe(
                [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="np"
            )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.8887, 0.915, 0.91, 0.894, 0.909, 0.912, 0.919, 0.925, 0.883])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_fast_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

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
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9326, 0.923, 0.951, 0.9365, 0.9214, 0.951, 0.9365, 0.9414, 0.918])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_score_sde_ve_pipeline(self):
        model_id = "google/ncsnpp-church-256"
        model = UNet2DModel.from_pretrained(model_id)

        scheduler = ScoreSdeVeScheduler.from_config(model_id)

        sde_ve = ScoreSdeVePipeline(unet=model, scheduler=scheduler)
        sde_ve.to(torch_device)
        sde_ve.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = sde_ve(num_inference_steps=10, output_type="numpy", generator=generator).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)

        expected_slice = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_uncond(self):
        ldm = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=5, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.4399, 0.44975, 0.46825, 0.474, 0.4359, 0.4581, 0.45095, 0.4341, 0.4447])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddpm_ddim_equality(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler()
        ddim_scheduler = DDIMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)
        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        ddpm_image = ddpm(generator=generator, output_type="numpy").images

        generator = torch.manual_seed(0)
        ddim_image = ddim(generator=generator, num_inference_steps=1000, eta=1.0, output_type="numpy").images

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_image - ddim_image).max() < 1e-1

    @unittest.skip("(Anton) The test is failing for large batch sizes, needs investigation")
    def test_ddpm_ddim_equality_batched(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler()
        ddim_scheduler = DDIMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        ddpm_images = ddpm(batch_size=4, generator=generator, output_type="numpy").images

        generator = torch.manual_seed(0)
        ddim_images = ddim(
            batch_size=4, generator=generator, num_inference_steps=1000, eta=1.0, output_type="numpy"
        ).images

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_images - ddim_images).max() < 1e-1

    @slow
    def test_karras_ve_pipeline(self):
        model_id = "google/ncsnpp-celebahq-256"
        model = UNet2DModel.from_pretrained(model_id)
        scheduler = KarrasVeScheduler()

        pipe = KarrasVePipeline(unet=model, scheduler=scheduler)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = pipe(num_inference_steps=20, generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.578, 0.5811, 0.5924, 0.5809, 0.587, 0.5886, 0.5861, 0.5802, 0.586])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_lms_stable_diffusion_pipeline(self):
        model_id = "CompVis/stable-diffusion-v1-1"
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        scheduler = LMSDiscreteScheduler.from_config(model_id, subfolder="scheduler")
        pipe.scheduler = scheduler

        prompt = "a photograph of an astronaut riding a horse"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = pipe(
            [prompt], generator=generator, guidance_scale=7.5, num_inference_steps=10, output_type="numpy"
        ).images

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9077, 0.9254, 0.9181, 0.9227, 0.9213, 0.9367, 0.9399, 0.9406, 0.9024])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_memory_chunking(self):
        torch.cuda.reset_peak_memory_stats()
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16).to(
            torch_device
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "a photograph of an astronaut riding a horse"

        # make attention efficient
        pipe.enable_attention_slicing()
        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast(torch_device):
            output_chunked = pipe(
                [prompt], generator=generator, guidance_scale=7.5, num_inference_steps=10, output_type="numpy"
            )
            image_chunked = output_chunked.images

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 3.75 GB is allocated
        assert mem_bytes < 3.75 * 10**9

        # disable chunking
        pipe.disable_attention_slicing()
        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast(torch_device):
            output = pipe(
                [prompt], generator=generator, guidance_scale=7.5, num_inference_steps=10, output_type="numpy"
            )
            image = output.images

        # make sure that more than 3.75 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 3.75 * 10**9
        assert np.abs(image_chunked.flatten() - image.flatten()).max() < 1e-3

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_text2img_pipeline_fp16(self):
        torch.cuda.reset_peak_memory_stats()
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16).to(
            torch_device
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "a photograph of an astronaut riding a horse"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output_chunked = pipe(
            [prompt], generator=generator, guidance_scale=7.5, num_inference_steps=10, output_type="numpy"
        )
        image_chunked = output_chunked.images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast(torch_device):
            output = pipe(
                [prompt], generator=generator, guidance_scale=7.5, num_inference_steps=10, output_type="numpy"
            )
            image = output.images

        # Make sure results are close enough
        diff = np.abs(image_chunked.flatten() - image.flatten())
        # They ARE different since ops are not run always at the same precision
        # however, they should be extremely close.
        assert diff.mean() < 2e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_text2img_pipeline(self):
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/text2img/astronaut_riding_a_horse.png"
        )
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=self.dummy_safety_checker,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "astronaut riding a horse"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(prompt=prompt, strength=0.75, guidance_scale=7.5, generator=generator, output_type="np")
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_img2img_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/fantasy_landscape.png"
        )
        init_image = init_image.resize((768, 512))
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=self.dummy_safety_checker,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 768, 3)
        # img2img is flaky across GPUs even in fp32, so using MAE here
        assert np.abs(expected_image - image).mean() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_img2img_pipeline_k_lms(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/fantasy_landscape_k_lms.png"
        )
        init_image = init_image.resize((768, 512))
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            scheduler=lms,
            safety_checker=self.dummy_safety_checker,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 768, 3)
        # img2img is flaky across GPUs even in fp32, so using MAE here
        assert np.abs(expected_image - image).mean() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_inpaint_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/yellow_cat_sitting_on_a_park_bench.png"
        )
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            safety_checker=self.dummy_safety_checker,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_inpaint_pipeline_fp16(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/yellow_cat_sitting_on_a_park_bench_fp16.png"
        )
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_inpaint_legacy_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/red_cat_sitting_on_a_park_bench.png"
        )
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            safety_checker=self.dummy_safety_checker,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A red cat sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_inpaint_pipeline_pndm(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/yellow_cat_sitting_on_a_park_bench_pndm.png"
        )
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        pndm = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True)
        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, safety_checker=self.dummy_safety_checker, scheduler=pndm
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_inpaint_legacy_pipeline_k_lms(self):
        # TODO(Anton, Patrick) - I think we can remove this test soon
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/red_cat_sitting_on_a_park_bench_k_lms.png"
        )
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            scheduler=lms,
            safety_checker=self.dummy_safety_checker,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A red cat sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-2

    @slow
    def test_stable_diffusion_onnx(self):
        sd_pipe = OnnxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="onnx", provider="CPUExecutionProvider"
        )

        prompt = "A painting of a squirrel eating a burger"
        np.random.seed(0)
        output = sd_pipe([prompt], guidance_scale=6.0, num_inference_steps=5, output_type="np")
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3602, 0.3688, 0.3652, 0.3895, 0.3782, 0.3747, 0.3927, 0.4241, 0.4327])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @slow
    def test_stable_diffusion_img2img_onnx(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="onnx", provider="CPUExecutionProvider"
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        np.random.seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=8,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.4830, 0.5242, 0.5603, 0.5016, 0.5131, 0.5111, 0.4928, 0.5025, 0.5055])
        # TODO: lower the tolerance after finding the cause of onnxruntime reproducibility issues
        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2

    @slow
    def test_stable_diffusion_inpaint_onnx(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )

        pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision="onnx", provider="CPUExecutionProvider"
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A red cat sitting on a park bench"

        np.random.seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            guidance_scale=7.5,
            num_inference_steps=8,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 255:258, -1]

        assert images.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.2951, 0.2955, 0.2922, 0.2036, 0.1977, 0.2279, 0.1716, 0.1641, 0.1799])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_text2img_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [1.8285, 1.2857, -0.1024, 1.2406, -2.3068, 1.0747, -0.0818, -0.6520, -2.9506]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 50:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [1.1078, 1.5803, 0.2773, -0.0589, -1.7928, -0.3665, -0.4695, -1.0727, -1.1601]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-2

        test_callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Andromeda galaxy in a bottle"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast(torch_device):
            pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                callback=test_callback_fn,
                callback_steps=1,
            )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 51

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_img2img_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.9052, -0.0184, 0.4810, 0.2898, 0.5851, 1.4920, 0.5362, 1.9838, 0.0530])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 37:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.7071, 0.7831, 0.8300, 1.8140, 1.7840, 1.9402, 1.3651, 1.6590, 1.2828])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-2

        test_callback_fn.has_been_called = False

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast(torch_device):
            pipe(
                prompt=prompt,
                init_image=init_image,
                strength=0.75,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                callback=test_callback_fn,
                callback_steps=1,
            )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 38

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_inpaint_legacy_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.5472, 1.1218, -0.5505, -0.9390, -1.0794, 0.4063, 0.5158, 0.6429, -1.5246]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 37:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.4781, 1.1572, 0.6258, 0.2291, 0.2554, -0.1443, 0.7085, -0.1598, -0.5659])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        test_callback_fn.has_been_called = False

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A red cat sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast(torch_device):
            pipe(
                prompt=prompt,
                init_image=init_image,
                mask_image=mask_image,
                strength=0.75,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                callback=test_callback_fn,
                callback_steps=1,
            )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 38

    @slow
    def test_stable_diffusion_onnx_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: np.ndarray) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.5950, -0.3039, -1.1672, 0.1594, -1.1572, 0.6719, -1.9712, -0.0403, 0.9592]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 5:
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.4776, -0.0119, -0.8519, -0.0275, -0.9764, 0.9820, -0.3843, 0.3788, 1.2264]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        test_callback_fn.has_been_called = False

        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="onnx", provider="CPUExecutionProvider"
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "Andromeda galaxy in a bottle"

        np.random.seed(0)
        pipe(prompt=prompt, num_inference_steps=5, guidance_scale=7.5, callback=test_callback_fn, callback_steps=1)
        assert test_callback_fn.has_been_called
        assert number_of_steps == 6

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_accelerate_load_works(self):
        if version.parse(version.parse(transformers.__version__).base_version) < version.parse("4.23"):
            return

        if version.parse(version.parse(accelerate.__version__).base_version) < version.parse("0.14"):
            return

        model_id = "CompVis/stable-diffusion-v1-4"
        _ = StableDiffusionPipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=True, device_map="auto"
        ).to(torch_device)

    @slow
    @unittest.skipIf(torch_device == "cpu", "This test is supposed to run on GPU")
    def test_stable_diffusion_accelerate_load_reduces_memory_footprint(self):
        if version.parse(version.parse(transformers.__version__).base_version) < version.parse("4.23"):
            return

        if version.parse(version.parse(accelerate.__version__).base_version) < version.parse("0.14"):
            return

        pipeline_id = "CompVis/stable-diffusion-v1-4"

        torch.cuda.empty_cache()
        gc.collect()

        tracemalloc.start()
        pipeline_normal_load = StableDiffusionPipeline.from_pretrained(
            pipeline_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=True
        )
        pipeline_normal_load.to(torch_device)
        _, peak_normal = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        del pipeline_normal_load
        torch.cuda.empty_cache()
        gc.collect()

        tracemalloc.start()
        _ = StableDiffusionPipeline.from_pretrained(
            pipeline_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=True, device_map="auto"
        )
        _, peak_accelerate = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        assert peak_accelerate < peak_normal
