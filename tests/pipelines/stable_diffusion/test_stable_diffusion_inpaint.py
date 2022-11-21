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
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from diffusers.utils import floats_tensor, load_image, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class StableDiffusionInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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

    def test_stable_diffusion_inpaint_with_num_images_per_prompt(self):
        device = "cpu"
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
        images = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=2,
        ).images

        # check if the output is a list of 2 images
        assert len(images) == 2

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


@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_inpaint_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint"
            "/yellow_cat_sitting_on_a_park_bench.npy"
        )

        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None)
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
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_inpaint_pipeline_fp16(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint"
            "/yellow_cat_sitting_on_a_park_bench_fp16.npy"
        )

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
        assert np.abs(expected_image - image).max() < 5e-1

    def test_stable_diffusion_inpaint_pipeline_pndm(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint"
            "/yellow_cat_sitting_on_a_park_bench_pndm.npy"
        )

        model_id = "runwayml/stable-diffusion-inpainting"
        pndm = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None, scheduler=pndm)
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

    def test_stable_diffusion_inpaint_pipeline_k_lms(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint"
            "/yellow_cat_sitting_on_a_park_bench_k_lms.npy"
        )

        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.to(torch_device)

        # switch to LMS
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

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

    @unittest.skipIf(torch_device == "cpu", "This test is supposed to run on GPU")
    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )

        model_id = "runwayml/stable-diffusion-inpainting"
        pndm = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            scheduler=pndm,
            device_map="auto",
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        _ = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator,
            num_inference_steps=5,
            output_type="np",
        )

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.2 GB is allocated
        assert mem_bytes < 2.2 * 10**9


class StableDiffusionInpaintingPrepareMaskAndMaskedImageTests(unittest.TestCase):
    def test_pil_inputs(self):
        im = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        im = Image.fromarray(im)
        mask = np.random.randint(0, 255, (32, 32), dtype=np.uint8) > 127.5
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        t_mask, t_masked = prepare_mask_and_masked_image(im, mask)

        self.assertTrue(isinstance(t_mask, torch.Tensor))
        self.assertTrue(isinstance(t_masked, torch.Tensor))

        self.assertEqual(t_mask.ndim, 4)
        self.assertEqual(t_masked.ndim, 4)

        self.assertEqual(t_mask.shape, (1, 1, 32, 32))
        self.assertEqual(t_masked.shape, (1, 3, 32, 32))

        self.assertTrue(t_mask.dtype == torch.float32)
        self.assertTrue(t_masked.dtype == torch.float32)

        self.assertTrue(t_mask.min() >= 0.0)
        self.assertTrue(t_mask.max() <= 1.0)
        self.assertTrue(t_masked.min() >= -1.0)
        self.assertTrue(t_masked.min() <= 1.0)

        self.assertTrue(t_mask.sum() > 0.0)

    def test_np_inputs(self):
        im_np = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        im_pil = Image.fromarray(im_np)
        mask_np = np.random.randint(0, 255, (32, 32), dtype=np.uint8) > 127.5
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)
        t_mask_pil, t_masked_pil = prepare_mask_and_masked_image(im_pil, mask_pil)

        self.assertTrue((t_mask_np == t_mask_pil).all())
        self.assertTrue((t_masked_np == t_masked_pil).all())

    def test_torch_3D_2D_inputs(self):
        im_tensor = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        mask_tensor = torch.randint(0, 255, (32, 32), dtype=torch.uint8) > 127.5
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor)
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_torch_3D_3D_inputs(self):
        im_tensor = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        mask_tensor = torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8) > 127.5
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor)
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_torch_4D_2D_inputs(self):
        im_tensor = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
        mask_tensor = torch.randint(0, 255, (32, 32), dtype=torch.uint8) > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor)
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_torch_4D_3D_inputs(self):
        im_tensor = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
        mask_tensor = torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8) > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor)
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_torch_4D_4D_inputs(self):
        im_tensor = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
        mask_tensor = torch.randint(0, 255, (1, 1, 32, 32), dtype=torch.uint8) > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0][0]

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor)
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_torch_batch_4D_3D(self):
        im_tensor = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
        mask_tensor = torch.randint(0, 255, (2, 32, 32), dtype=torch.uint8) > 127.5

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy() for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor)
        nps = [prepare_mask_and_masked_image(i, m) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = torch.cat([n[0] for n in nps])
        t_masked_np = torch.cat([n[1] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_torch_batch_4D_4D(self):
        im_tensor = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
        mask_tensor = torch.randint(0, 255, (2, 1, 32, 32), dtype=torch.uint8) > 127.5

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy()[0] for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor)
        nps = [prepare_mask_and_masked_image(i, m) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = torch.cat([n[0] for n in nps])
        t_masked_np = torch.cat([n[1] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_shape_mismatch(self):
        # test height and width
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(torch.randn(3, 32, 32), torch.randn(64, 64))
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(torch.randn(2, 3, 32, 32), torch.randn(4, 64, 64))
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(torch.randn(2, 3, 32, 32), torch.randn(4, 1, 64, 64))

    def test_type_mismatch(self):
        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(torch.rand(3, 32, 32), torch.rand(3, 32, 32).numpy())
        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(torch.rand(3, 32, 32).numpy(), torch.rand(3, 32, 32))

    def test_channels_first(self):
        # test channels first for 3D tensors
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(torch.rand(32, 32, 3), torch.rand(3, 32, 32))

    def test_tensor_range(self):
        # test im <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(torch.ones(3, 32, 32) * 2, torch.rand(32, 32))
        # test im >= -1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(torch.ones(3, 32, 32) * (-2), torch.rand(32, 32))
        # test mask <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(torch.rand(3, 32, 32), torch.ones(32, 32) * 2)
        # test mask >= 0
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(torch.rand(3, 32, 32), torch.ones(32, 32) * -1)
