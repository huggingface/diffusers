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
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipelineLegacy,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from diffusers.utils import floats_tensor, load_image, nightly, slow, torch_device
from diffusers.utils.testing_utils import enable_full_determinism, load_numpy, preprocess_image, require_torch_gpu


enable_full_determinism()


class StableDiffusionInpaintLegacyPipelineFastTests(unittest.TestCase):
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

    def test_stable_diffusion_inpaint_legacy(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
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

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4941, 0.5396, 0.4689, 0.6338, 0.5392, 0.4094, 0.5477, 0.5904, 0.5165])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_batched(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        init_images_tens = preprocess_image(init_image, batch_size=2)
        init_masks_tens = init_images_tens + 4

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
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
        images = sd_pipe(
            [prompt] * 2,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_images_tens,
            mask_image=init_masks_tens,
        ).images

        assert images.shape == (2, 32, 32, 3)

        image_slice_0 = images[0, -3:, -3:, -1].flatten()
        image_slice_1 = images[1, -3:, -3:, -1].flatten()

        expected_slice_0 = np.array([0.4697, 0.3770, 0.4096, 0.4653, 0.4497, 0.4183, 0.3950, 0.4668, 0.4672])
        expected_slice_1 = np.array([0.4105, 0.4987, 0.5771, 0.4921, 0.4237, 0.5684, 0.5496, 0.4645, 0.5272])

        assert np.abs(expected_slice_0 - image_slice_0).max() < 1e-2
        assert np.abs(expected_slice_1 - image_slice_1).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
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
        negative_prompt = "french fries"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4941, 0.5396, 0.4689, 0.6338, 0.5392, 0.4094, 0.5477, 0.5904, 0.5165])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_num_images_per_prompt(self):
        device = "cpu"
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
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

        # test num_images_per_prompt=1 (default)
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
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
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
@require_torch_gpu
class StableDiffusionInpaintLegacyPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, generator_device="cpu", seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "A red cat sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint_legacy_pndm(self):
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5665, 0.6117, 0.6430, 0.4057, 0.4594, 0.5658, 0.1596, 0.3106, 0.4305])

        assert np.abs(expected_slice - image_slice).max() < 3e-3

    def test_stable_diffusion_inpaint_legacy_batched(self):
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = preprocess_image(inputs["image"], batch_size=2)

        mask = inputs["mask_image"].convert("L")
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(1 - mask)
        masks = torch.vstack([mask[None][None]] * 2)
        inputs["mask_image"] = masks

        image = pipe(**inputs).images
        assert image.shape == (2, 512, 512, 3)

        image_slice_0 = image[0, 253:256, 253:256, -1].flatten()
        image_slice_1 = image[1, 253:256, 253:256, -1].flatten()

        expected_slice_0 = np.array(
            [0.52093095, 0.4176447, 0.32752383, 0.6175223, 0.50563973, 0.36470804, 0.65460044, 0.5775188, 0.44332123]
        )
        expected_slice_1 = np.array(
            [0.3592432, 0.4233033, 0.3914635, 0.31014425, 0.3702293, 0.39412856, 0.17526966, 0.2642669, 0.37480092]
        )

        assert np.abs(expected_slice_0 - image_slice_0).max() < 3e-3
        assert np.abs(expected_slice_1 - image_slice_1).max() < 3e-3

    def test_stable_diffusion_inpaint_legacy_k_lms(self):
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.4534, 0.4467, 0.4329, 0.4329, 0.4339, 0.4220, 0.4244, 0.4332, 0.4426])

        assert np.abs(expected_slice - image_slice).max() < 3e-3

    def test_stable_diffusion_inpaint_legacy_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.5977, 1.5449, 1.0586, -0.3250, 0.7383, -0.0862, 0.4631, -0.2571, -1.1289])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.5190, 1.1621, 0.6885, 0.2424, 0.3337, -0.1617, 0.6914, -0.1957, -0.5474])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        callback_fn.has_been_called = False

        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2


@nightly
@require_torch_gpu
class StableDiffusionInpaintLegacyPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "A red cat sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 50,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_inpaint_pndm(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_ddim(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_lms(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_dpm(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
