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

from diffusers import AutoencoderKL, PaintByExamplePipeline, PNDMScheduler, UNet2DConditionModel, UNet2DModel
from diffusers.pipelines.paint_by_example import PaintByExampleImageEncoder
from diffusers.utils import floats_tensor, load_image, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu
from PIL import Image
from transformers import CLIPVisionConfig

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class PaintByExamplePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
    def dummy_image_encoder(self):
        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            image_size=32,
            patch_size=4,
        )
        return PaintByExampleImageEncoder(config, proj_size=32)

    def convert_to_pt(self, image):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        return image

    def test_paint_by_example_inpaint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet_inpaint
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))

        example_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        example_image = self.convert_to_pt(example_image)

        image_encoder = self.dummy_image_encoder

        # make sure here that pndm scheduler skips prk
        pipe = PaintByExamplePipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            image_encoder=image_encoder,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(
            example_image=example_image,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        )

        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = pipe(
            example_image=example_image,
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

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4397, 0.5553, 0.3802, 0.5222, 0.5811, 0.4342, 0.494, 0.4577, 0.4428])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_paint_by_example_image_tensor(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet_inpaint
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        image_encoder = self.dummy_image_encoder

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        example_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        example_image = self.convert_to_pt(example_image)

        image = self.dummy_image.repeat(1, 1, 2, 2)
        mask_image = image / 2

        # make sure here that pndm scheduler skips prk
        pipe = PaintByExamplePipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            image_encoder=image_encoder,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(
            example_image=example_image,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=image,
            mask_image=mask_image[:, 0],
        )
        out_1 = output.images

        image = image.cpu().permute(0, 2, 3, 1)[0]
        mask_image = mask_image.cpu().permute(0, 2, 3, 1)[0]

        image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(mask_image)).convert("RGB")

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(
            example_image=example_image,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=image,
            mask_image=mask_image,
        )
        out_2 = output.images

        assert out_1.shape == (1, 64, 64, 3)
        assert np.abs(out_1.flatten() - out_2.flatten()).max() < 5e-2

    def test_paint_by_example_inpaint_with_num_images_per_prompt(self):
        device = "cpu"
        unet = self.dummy_cond_unet_inpaint
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))

        example_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        example_image = self.convert_to_pt(example_image)

        image_encoder = self.dummy_image_encoder

        # make sure here that pndm scheduler skips prk
        pipe = PaintByExamplePipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            image_encoder=image_encoder,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        images = pipe(
            example_image=example_image,
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
    def test_paint_by_example_inpaint_fp16(self):
        """Test that stable diffusion inpaint_legacy works with fp16"""
        unet = self.dummy_cond_unet_inpaint
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))

        example_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        example_image = self.convert_to_pt(example_image).to(torch.float16)

        image_encoder = self.dummy_image_encoder

        # put models in fp16
        image_encoder = image_encoder.half()
        unet = unet.half()
        vae = vae.half()

        # make sure here that pndm scheduler skips prk
        pipe = PaintByExamplePipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            image_encoder=image_encoder,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = pipe(
            example_image=example_image,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images

        assert image.shape == (1, 64, 64, 3)


@slow
@require_torch_gpu
class PaintByExamplePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_paint_by_example(self):
        # make sure here that pndm scheduler skips prk
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/paint_by_example/dog_in_bucket.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/paint_by_example/mask.png"
        )
        example_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/paint_by_example/panda.jpg"
        )

        pipe = PaintByExamplePipeline.from_pretrained("patrickvonplaten/new_inpaint_test")
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(321)
        output = pipe(
            image=init_image,
            mask_image=mask_image,
            example_image=example_image,
            generator=generator,
            guidance_scale=5.0,
            num_inference_steps=50,
            output_type="np",
        )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.47455794, 0.47086594, 0.47683704, 0.51024145, 0.5064255, 0.5123164, 0.532502, 0.5328063, 0.5428694]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
