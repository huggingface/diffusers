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
import unittest

import numpy as np
import torch

import PIL
from diffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
    logging,
)
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, WEIGHTS_NAME, floats_tensor, slow, torch_device
from diffusers.utils.testing_utils import CaptureLogger, get_tests_dir
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

        feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id, device_map="auto")
        clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16, device_map="auto")

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            custom_pipeline="clip_guided_stable_diffusion",
            clip_model=clip_model,
            feature_extractor=feature_extractor,
            torch_dtype=torch.float16,
            revision="fp16",
            device_map="auto",
        )
        pipeline.enable_attention_slicing()
        pipeline = pipeline.to(torch_device)

        # NOTE that `"CLIPGuidedStableDiffusion"` is not a class that is defined in the pypi package of th e library, but solely on the community examples folder of GitHub under:
        # https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py
        assert pipeline.__class__.__name__ == "CLIPGuidedStableDiffusion"

        image = pipeline("a prompt", num_inference_steps=2, output_type="np").images[0]
        assert image.shape == (512, 512, 3)


class PipelineFastTests(unittest.TestCase):
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

    def test_components(self):
        """Test that components property works correctly"""
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((128, 128))

        # make sure here that pndm scheduler skips prk
        inpaint = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        ).to(torch_device)
        img2img = StableDiffusionImg2ImgPipeline(**inpaint.components).to(torch_device)
        text2img = StableDiffusionPipeline(**inpaint.components).to(torch_device)

        prompt = "A painting of a squirrel eating a burger"

        # Device type MPS is not supported for torch.Generator() api.
        if torch_device == "mps":
            generator = torch.manual_seed(0)
        else:
            generator = torch.Generator(device=torch_device).manual_seed(0)

        image_inpaint = inpaint(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
        ).images
        image_img2img = img2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        ).images
        image_text2img = text2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images

        assert image_inpaint.shape == (1, 32, 32, 3)
        assert image_img2img.shape == (1, 32, 32, 3)
        assert image_text2img.shape == (1, 128, 128, 3)


@slow
class PipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_smart_download(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = DiffusionPipeline.from_pretrained(
                model_id, cache_dir=tmpdirname, force_download=True, device_map="auto"
            )
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

    def test_warning_unused_kwargs(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        logger = logging.get_logger("diffusers.pipeline_utils")
        with tempfile.TemporaryDirectory() as tmpdirname:
            with CaptureLogger(logger) as cap_logger:
                DiffusionPipeline.from_pretrained(
                    model_id, not_used=True, cache_dir=tmpdirname, force_download=True, device_map="auto"
                )

        assert cap_logger.out == "Keyword arguments {'not_used': True} not recognized.\n"

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
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname, device_map="auto")
            new_ddpm.to(torch_device)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        generator = generator.manual_seed(0)
        new_image = new_ddpm(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline.from_pretrained(model_path, scheduler=scheduler, device_map="auto")
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, device_map="auto")
        ddpm_from_hub.to(torch_device)
        ddpm_from_hub.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub_pass_model(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        # pass unet into DiffusionPipeline
        unet = UNet2DModel.from_pretrained(model_path, device_map="auto")
        ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(
            model_path, unet=unet, scheduler=scheduler, device_map="auto"
        )
        ddpm_from_hub_custom_model.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, device_map="auto")
        ddpm_from_hub.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm_from_hub_custom_model(generator=generator, output_type="numpy").images

        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"

        pipe = DDIMPipeline.from_pretrained(model_path, device_map="auto")
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

    def test_ddpm_ddim_equality(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id, device_map="auto")
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

        unet = UNet2DModel.from_pretrained(model_id, device_map="auto")
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
