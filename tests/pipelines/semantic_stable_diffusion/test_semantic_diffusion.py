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
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.semantic_stable_diffusion import SemanticStableDiffusionPipeline as StableDiffusionPipeline
from diffusers.utils import floats_tensor, nightly, torch_device
from diffusers.utils.testing_utils import require_torch_gpu


torch.backends.cuda.matmul.allow_tf32 = False


class SafeDiffusionPipelineFastTests(unittest.TestCase):
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

    def test_semantic_diffusion_ddim(self):
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
            safety_checker=None,
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

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5644, 0.6018, 0.4799, 0.5267, 0.5585, 0.4641, 0.516, 0.4964, 0.4792])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_semantic_diffusion_pndm(self):
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
            safety_checker=None,
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

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5095, 0.5674, 0.4668, 0.5126, 0.5697, 0.4675, 0.5278, 0.4964, 0.4945])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_semantic_diffusion_no_safety_checker(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-lms-pipe", safety_checker=None
        )
        assert isinstance(pipe, StableDiffusionPipeline)
        assert isinstance(pipe.scheduler, LMSDiscreteScheduler)
        assert pipe.safety_checker is None

        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

        # check that there's no error when saving a pipeline with one of the models being None
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = StableDiffusionPipeline.from_pretrained(tmpdirname)

        # sanity check that the pipeline still works
        assert pipe.safety_checker is None
        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")
    def test_semantic_diffusion_fp16(self):
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
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        image = sd_pipe([prompt], num_inference_steps=2, output_type="np").images

        assert image.shape == (1, 64, 64, 3)


@nightly
@require_torch_gpu
class SemanticDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_positive_guidance(self):
        torch_device = "cuda"
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "a photo of a cat"
        edit = {
            "editing_prompt": ["sunglasses"],
            "reverse_editing_direction": [False],
            "edit_warmup_steps": 10,
            "edit_guidance_scale": 6,
            "edit_threshold": 0.95,
            "edit_momentum_scale": 0.5,
            "edit_mom_beta": 0.6,
        }

        seed = 3
        guidance_scale = 7

        # no sega enabled
        generator = torch.Generator(torch_device)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.34673113,
            0.38492733,
            0.37597352,
            0.34086335,
            0.35650748,
            0.35579205,
            0.3384763,
            0.34340236,
            0.3573271,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

        # with sega enabled
        # generator = torch.manual_seed(seed)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
            **edit,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.41887826,
            0.37728766,
            0.30138272,
            0.41416335,
            0.41664985,
            0.36283392,
            0.36191246,
            0.43364465,
            0.43001732,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_negative_guidance(self):
        torch_device = "cuda"
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "an image of a crowded boulevard, realistic, 4k"
        edit = {
            "editing_prompt": "crowd, crowded, people",
            "reverse_editing_direction": True,
            "edit_warmup_steps": 10,
            "edit_guidance_scale": 8.3,
            "edit_threshold": 0.9,
            "edit_momentum_scale": 0.5,
            "edit_mom_beta": 0.6,
        }

        seed = 9
        guidance_scale = 7

        # no sega enabled
        generator = torch.Generator(torch_device)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.43497998,
            0.91814065,
            0.7540739,
            0.55580205,
            0.8467265,
            0.5389691,
            0.62574506,
            0.58897763,
            0.50926757,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

        # with sega enabled
        # generator = torch.manual_seed(seed)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
            **edit,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.3089719,
            0.30500144,
            0.29016042,
            0.30630964,
            0.325687,
            0.29419225,
            0.2908091,
            0.28723598,
            0.27696294,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_multi_cond_guidance(self):
        torch_device = "cuda"
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "a castle next to a river"
        edit = {
            "editing_prompt": ["boat on a river, boat", "monet, impression, sunrise"],
            "reverse_editing_direction": False,
            "edit_warmup_steps": [15, 18],
            "edit_guidance_scale": 6,
            "edit_threshold": [0.9, 0.8],
            "edit_momentum_scale": 0.5,
            "edit_mom_beta": 0.6,
        }

        seed = 48
        guidance_scale = 7

        # no sega enabled
        generator = torch.Generator(torch_device)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.75163555,
            0.76037145,
            0.61785,
            0.9189673,
            0.8627701,
            0.85189694,
            0.8512813,
            0.87012076,
            0.8312857,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

        # with sega enabled
        # generator = torch.manual_seed(seed)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
            **edit,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.73553365,
            0.7537271,
            0.74341905,
            0.66480356,
            0.6472925,
            0.63039416,
            0.64812905,
            0.6749717,
            0.6517102,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_guidance_fp16(self):
        torch_device = "cuda"
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "a photo of a cat"
        edit = {
            "editing_prompt": ["sunglasses"],
            "reverse_editing_direction": [False],
            "edit_warmup_steps": 10,
            "edit_guidance_scale": 6,
            "edit_threshold": 0.95,
            "edit_momentum_scale": 0.5,
            "edit_mom_beta": 0.6,
        }

        seed = 3
        guidance_scale = 7

        # no sega enabled
        generator = torch.Generator(torch_device)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.34887695,
            0.3876953,
            0.375,
            0.34423828,
            0.3581543,
            0.35717773,
            0.3383789,
            0.34570312,
            0.359375,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

        # with sega enabled
        # generator = torch.manual_seed(seed)
        generator.manual_seed(seed)
        output = pipe(
            [prompt],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            output_type="np",
            width=512,
            height=512,
            **edit,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = [
            0.42285156,
            0.36914062,
            0.29077148,
            0.42041016,
            0.41918945,
            0.35498047,
            0.3618164,
            0.4423828,
            0.43115234,
        ]

        assert image.shape == (1, 512, 512, 3)

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
