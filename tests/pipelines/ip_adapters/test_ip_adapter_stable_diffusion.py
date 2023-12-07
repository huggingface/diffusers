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
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
    torch_device,
)


enable_full_determinism()


class IPAdapterNightlyTestsMixin(unittest.TestCase):
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

    def get_dummy_inputs(self, for_image_to_image=False, for_inpainting=False, for_sdxl=False):
        image = load_image(
            "https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png"
        )
        if for_sdxl:
            image = image.resize((1024, 1024))

        input_kwargs = {
            "prompt": "best quality, high quality",
            "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
            "num_inference_steps": 5,
            "generator": torch.Generator(device="cpu").manual_seed(33),
            "ip_adapter_image": image,
            "output_type": "np",
        }
        if for_image_to_image:
            image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/vermeer.jpg")
            ip_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/river.png")

            if for_sdxl:
                image = image.resize((1024, 1024))
                ip_image = ip_image.resize((1024, 1024))

            input_kwargs.update({"image": image, "ip_adapter_image": ip_image})

        elif for_inpainting:
            image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/inpaint_image.png")
            mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/mask.png")
            ip_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/girl.png")

            if for_sdxl:
                image = image.resize((1024, 1024))
                mask = mask.resize((1024, 1024))
                ip_image = ip_image.resize((1024, 1024))

            input_kwargs.update({"image": image, "mask_image": mask, "ip_adapter_image": ip_image})

        return input_kwargs


@slow
@require_torch_gpu
class IPAdapterSDIntegrationTests(IPAdapterNightlyTestsMixin):
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

        expected_slice = np.array([0.8110, 0.8843, 0.9326, 0.9224, 0.9878, 1.0, 0.9736, 1.0, 1.0])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.3013, 0.2615, 0.2202, 0.2722, 0.2510, 0.2023, 0.2498, 0.2415, 0.2139])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

    def test_image_to_image(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        inputs = self.get_dummy_inputs(for_image_to_image=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.2253, 0.2251, 0.2219, 0.2312, 0.2236, 0.2434, 0.2275, 0.2575, 0.2805])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs(for_image_to_image=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.3550, 0.2600, 0.2520, 0.2412, 0.1870, 0.3831, 0.1453, 0.1880, 0.5371])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

    def test_inpainting(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.2700, 0.2388, 0.2202, 0.2304, 0.2095, 0.2097, 0.2173, 0.2058, 0.1987])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.2744, 0.2410, 0.2202, 0.2334, 0.2090, 0.2053, 0.2175, 0.2033, 0.1934])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

    def test_text_to_image_full_face(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
        pipeline.set_ip_adapter_scale(0.7)

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [0.1706543, 0.1303711, 0.12573242, 0.21777344, 0.14550781, 0.14038086, 0.40820312, 0.41455078, 0.42529297]
        )

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)


@slow
@require_torch_gpu
class IPAdapterSDXLIntegrationTests(IPAdapterNightlyTestsMixin):
    def test_text_to_image_sdxl(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="sdxl_models/image_encoder")
        feature_extractor = self.get_image_processor("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.0965, 0.0956, 0.0849, 0.0908, 0.0944, 0.0927, 0.0888, 0.0929, 0.0920])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.bin",
        )

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.0592, 0.0573, 0.0459, 0.0542, 0.0559, 0.0523, 0.0500, 0.0540, 0.0501])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

    def test_image_to_image_sdxl(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="sdxl_models/image_encoder")
        feature_extractor = self.get_image_processor("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs(for_image_to_image=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.0652, 0.0698, 0.0723, 0.0744, 0.0699, 0.0636, 0.0784, 0.0803, 0.0742])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        feature_extractor = self.get_image_processor("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.bin",
        )

        inputs = self.get_dummy_inputs(for_image_to_image=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.0708, 0.0701, 0.0735, 0.0760, 0.0739, 0.0679, 0.0756, 0.0824, 0.0837])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

    def test_inpainting_sdxl(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="sdxl_models/image_encoder")
        feature_extractor = self.get_image_processor("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        image_slice.tolist()

        expected_slice = np.array([0.1420, 0.1495, 0.1430, 0.1462, 0.1493, 0.1502, 0.1474, 0.1502, 0.1517])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        feature_extractor = self.get_image_processor("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.bin",
        )

        inputs = self.get_dummy_inputs(for_inpainting=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        image_slice.tolist()

        expected_slice = np.array([0.1398, 0.1476, 0.1407, 0.1442, 0.1470, 0.1480, 0.1449, 0.1481, 0.1494])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)
