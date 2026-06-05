# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
from unittest import mock

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
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import load_image

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    enable_full_determinism,
    is_flaky,
    load_pt,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)


enable_full_determinism()


class IPAdapterNightlyTestsMixin(unittest.TestCase):
    dtype = torch.float16

    _SD_PIPELINE_RANDN_TENSOR_TARGETS = {
        StableDiffusionPipeline: "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.randn_tensor",
        StableDiffusionImg2ImgPipeline: "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.randn_tensor",
        StableDiffusionInpaintPipeline: "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.randn_tensor",
        StableDiffusionXLPipeline: "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.randn_tensor",
        StableDiffusionXLImg2ImgPipeline: "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.randn_tensor",
        StableDiffusionXLInpaintPipeline: "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.randn_tensor",
    }

    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_fixed_noise(self, shape=(1, 4, 64, 64), seed=33):
        return torch.from_numpy(np.random.RandomState(seed).standard_normal(shape)).to(torch.float32)

    def get_fixed_randn_tensor_patch(self, pipeline, shape=(1, 4, 64, 64), seed=33):
        fixed_noise = self.get_fixed_noise(shape=shape, seed=seed)

        def fake_randn_tensor(requested_shape, generator=None, device=None, dtype=None, layout=None):
            self.assertEqual(tuple(requested_shape), tuple(fixed_noise.shape))
            return fixed_noise.to(device=device, dtype=dtype)

        for pipeline_cls, target in self._SD_PIPELINE_RANDN_TENSOR_TARGETS.items():
            if isinstance(pipeline, pipeline_cls):
                return mock.patch(target, side_effect=fake_randn_tensor)

        self.fail(f"No fixed randn_tensor patch target configured for pipeline type {type(pipeline)}")

    def get_image_encoder(self, repo_id, subfolder):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            repo_id, subfolder=subfolder, torch_dtype=self.dtype
        ).to(torch_device)
        return image_encoder

    def get_image_processor(self, repo_id):
        image_processor = CLIPImageProcessor.from_pretrained(repo_id)
        return image_processor

    def get_dummy_inputs(
        self, for_image_to_image=False, for_inpainting=False, for_sdxl=False, for_masks=False, for_instant_style=False
    ):
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

        elif for_masks:
            face_image1 = load_image(
                "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_girl1.png"
            )
            face_image2 = load_image(
                "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_girl2.png"
            )
            mask1 = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_mask1.png")
            mask2 = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_mask2.png")
            input_kwargs.update(
                {
                    "ip_adapter_image": [[face_image1], [face_image2]],
                    "cross_attention_kwargs": {"ip_adapter_masks": [mask1, mask2]},
                }
            )

        elif for_instant_style:
            composition_mask = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/1024_whole_mask.png"
            )
            female_mask = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter_None_20240321125641_mask.png"
            )
            male_mask = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter_None_20240321125344_mask.png"
            )
            background_mask = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter_6_20240321130722_mask.png"
            )
            ip_composition_image = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321125152.png"
            )
            ip_female_style = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321125625.png"
            )
            ip_male_style = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321125329.png"
            )
            ip_background = load_image(
                "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321130643.png"
            )
            input_kwargs.update(
                {
                    "ip_adapter_image": [ip_composition_image, [ip_female_style, ip_male_style, ip_background]],
                    "cross_attention_kwargs": {
                        "ip_adapter_masks": [[composition_mask], [female_mask, male_mask, background_mask]]
                    },
                }
            )

        return input_kwargs


@slow
@require_torch_accelerator
class IPAdapterSDIntegrationTests(IPAdapterNightlyTestsMixin):
    def test_text_to_image(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_encoder=image_encoder,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        inputs = self.get_dummy_inputs()
        with self.get_fixed_randn_tensor_patch(pipeline):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.3291, 0.2964, 0.2742, 0.3010, 0.2698, 0.2507, 0.2917, 0.2671, 0.2478])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs()
        with self.get_fixed_randn_tensor_patch(pipeline):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = Expectations(
            {
                ("cuda", None): np.array([0.1238, 0.0579, 0.0312, 0.0493, 0.0010, 0.0, 0.0188, 0.0, 0.0]),
                ("xpu", None): np.array(
                    [0.11938477, 0.05249023, 0.02490234, 0.04370117, 0.0, 0.0, 0.01342773, 0.0, 0.0]
                ),
                (None, None): np.array([0.1238, 0.0579, 0.0312, 0.0493, 0.0010, 0.0, 0.0188, 0.0, 0.0]),
            }
        ).get_expectation()
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_image_to_image(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_encoder=image_encoder,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        with self.get_fixed_randn_tensor_patch(pipeline):
            inputs = self.get_dummy_inputs(for_image_to_image=True)
            images = pipeline(**inputs).images
            image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.1492, 0.1294, 0.1123, 0.1504, 0.1328, 0.0923, 0.1428, 0.1479, 0.1370])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        with self.get_fixed_randn_tensor_patch(pipeline):
            inputs = self.get_dummy_inputs(for_image_to_image=True)
            images = pipeline(**inputs).images
            image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = Expectations(
            {
                ("cuda", None): np.array([0.0493, 0.0059, 0.0, 0.0166, 0.0056, 0.0027, 0.0139, 0.0090, 0.0129]),
                ("xpu", None): np.array([0.0513, 0.0083, 0.0, 0.0183, 0.0073, 0.0039, 0.0159, 0.0100, 0.0142]),
                (None, None): np.array([0.0493, 0.0059, 0.0, 0.0166, 0.0056, 0.0027, 0.0139, 0.0090, 0.0129]),
            }
        ).get_expectation()

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_inpainting(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_encoder=image_encoder,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        with self.get_fixed_randn_tensor_patch(pipeline):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.2766, 0.2437, 0.2246, 0.2354, 0.2126, 0.2119, 0.2207, 0.2075, 0.1992])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        with self.get_fixed_randn_tensor_patch(pipeline):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.3042, 0.2739, 0.2532, 0.2666, 0.2434, 0.2351, 0.2507, 0.2358, 0.2217])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_text_to_image_model_cpu_offload(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_encoder=image_encoder,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipeline.to(torch_device)

        inputs = self.get_dummy_inputs()
        with self.get_fixed_randn_tensor_patch(pipeline):
            output_without_offload = pipeline(**inputs).images

        pipeline.enable_model_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs()
        with self.get_fixed_randn_tensor_patch(pipeline):
            output_with_offload = pipeline(**inputs).images
        max_diff = np.abs(output_with_offload - output_without_offload).max()
        self.assertLess(max_diff, 1e-3, "CPU offloading should not affect the inference results")

        offloaded_modules = [
            v
            for k, v in pipeline.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipeline._exclude_from_cpu_offload
        ]
        (
            self.assertTrue(all(v.device.type == "cpu" for v in offloaded_modules)),
            f"Not offloaded: {[v for v in offloaded_modules if v.device.type != 'cpu']}",
        )

    def test_text_to_image_full_face(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_encoder=image_encoder,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
        pipeline.set_ip_adapter_scale(0.7)

        inputs = self.get_dummy_inputs()
        with self.get_fixed_randn_tensor_patch(pipeline):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array([0.4033, 0.3989, 0.3992, 0.4006, 0.3879, 0.4355, 0.4192, 0.4333, 0.4753])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_unload(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_encoder=image_encoder,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        before_processors = [attn_proc.__class__ for attn_proc in pipeline.unet.attn_processors.values()]
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipeline.set_ip_adapter_scale(0.7)

        pipeline.unload_ip_adapter()

        assert getattr(pipeline, "image_encoder") is None
        assert getattr(pipeline, "feature_extractor") is not None
        after_processors = [attn_proc.__class__ for attn_proc in pipeline.unet.attn_processors.values()]

        assert before_processors == after_processors

    @is_flaky
    def test_multi(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_encoder=image_encoder,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name=["ip-adapter_sd15.bin", "ip-adapter-plus_sd15.bin"]
        )
        pipeline.set_ip_adapter_scale([0.7, 0.3])

        inputs = self.get_dummy_inputs()
        ip_adapter_image = inputs["ip_adapter_image"]
        inputs["ip_adapter_image"] = [ip_adapter_image, [ip_adapter_image] * 2]
        with self.get_fixed_randn_tensor_patch(pipeline):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array([0.2783, 0.2302, 0.1921, 0.2354, 0.1934, 0.1528, 0.2207, 0.1902, 0.1526])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_text_to_image_face_id(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid_sd15.bin",
            image_encoder_folder=None,
        )
        pipeline.set_ip_adapter_scale(0.7)

        inputs = self.get_dummy_inputs()
        id_embeds = load_pt(
            "https://huggingface.co/datasets/fabiorigano/testing-images/resolve/main/ai_face2.ipadpt",
            map_location=torch_device,
        )[0]
        id_embeds = id_embeds.reshape((2, 1, 1, 512))
        inputs["ip_adapter_image_embeds"] = [id_embeds]
        inputs["ip_adapter_image"] = None
        with self.get_fixed_randn_tensor_patch(pipeline):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array([0.4780, 0.5117, 0.5103, 0.5044, 0.4922, 0.4932, 0.5029, 0.4954, 0.4802])
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4


@slow
@require_torch_accelerator
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
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs()
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [
                0.15138859,
                0.15170279,
                0.14246401,
                0.15483627,
                0.15317351,
                0.15564519,
                0.14952978,
                0.15584505,
                0.14940351,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

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
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [
                0.09022659,
                0.08629113,
                0.07586601,
                0.09006533,
                0.08684656,
                0.08665657,
                0.08367643,
                0.08839294,
                0.08377907,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_image_to_image_sdxl(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="sdxl_models/image_encoder")
        feature_extractor = self.get_image_processor("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs(for_image_to_image=True)
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 64, 64)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [0.05107406, 0.05074775, 0.00099546, 0.05845362, 0.05587912, 0.0, 0.06056768, 0.05724522, 0.0648115]
        )
        assert np.allclose(image_slice, expected_slice, atol=1e-3)

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
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 64, 64)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [0.05652112, 0.05557555, 0.00392720, 0.06261870, 0.06117940, 0.0, 0.05906063, 0.06035855, 0.06263199]
        )

        assert np.allclose(image_slice, expected_slice, atol=1e-3)

    def test_inpainting_sdxl(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="sdxl_models/image_encoder")
        feature_extractor = self.get_image_processor("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=self.dtype,
        )
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [
                0.14227295,
                0.14525282,
                0.14307272,
                0.15040666,
                0.14928216,
                0.14794737,
                0.14742243,
                0.15273672,
                0.15166444,
            ]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

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
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [
                0.14031684,
                0.14346808,
                0.14132470,
                0.14918229,
                0.14789128,
                0.14650577,
                0.14599693,
                0.15143514,
                0.15061957,
            ]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_ip_adapter_mask(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
        )
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
        )
        pipeline.set_ip_adapter_scale(0.7)

        inputs = self.get_dummy_inputs(for_masks=True)
        mask = inputs["cross_attention_kwargs"]["ip_adapter_masks"][0]
        processor = IPAdapterMaskProcessor()
        mask = processor.preprocess(mask)
        inputs["cross_attention_kwargs"]["ip_adapter_masks"] = mask
        inputs["ip_adapter_image"] = inputs["ip_adapter_image"][0]
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [
                0.47833657,
                0.50273246,
                0.49865803,
                0.46196738,
                0.51376355,
                0.49931064,
                0.45902768,
                0.55391037,
                0.50260746,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_ip_adapter_multiple_masks(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
        )
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"] * 2
        )
        pipeline.set_ip_adapter_scale([0.7] * 2)

        inputs = self.get_dummy_inputs(for_masks=True)
        masks = inputs["cross_attention_kwargs"]["ip_adapter_masks"]
        processor = IPAdapterMaskProcessor()
        masks = processor.preprocess(masks)
        inputs["cross_attention_kwargs"]["ip_adapter_masks"] = masks
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [0.3578991, 0.39458388, 0.43545875, 0.35710996, 0.3885604, 0.43619853, 0.37826842, 0.39264038, 0.45008034]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_instant_style_multiple_masks(self):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, image_encoder=image_encoder, variant="fp16"
        )
        pipeline.enable_model_cpu_offload(device=torch_device)

        pipeline.load_ip_adapter(
            ["ostris/ip-composition-adapter", "h94/IP-Adapter"],
            subfolder=["", "sdxl_models"],
            weight_name=[
                "ip_plus_composition_sdxl.safetensors",
                "ip-adapter_sdxl_vit-h.safetensors",
            ],
            image_encoder_folder=None,
        )
        scale_1 = {
            "down": [[0.0, 0.0, 1.0]],
            "mid": [[0.0, 0.0, 1.0]],
            "up": {"block_0": [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]], "block_1": [[0.0, 0.0, 1.0]]},
        }
        pipeline.set_ip_adapter_scale([1.0, scale_1])

        inputs = self.get_dummy_inputs(for_instant_style=True)
        processor = IPAdapterMaskProcessor()
        masks1 = inputs["cross_attention_kwargs"]["ip_adapter_masks"][0]
        masks2 = inputs["cross_attention_kwargs"]["ip_adapter_masks"][1]
        masks1 = processor.preprocess(masks1, height=1024, width=1024)
        masks2 = processor.preprocess(masks2, height=1024, width=1024)
        masks2 = masks2.reshape(1, masks2.shape[0], masks2.shape[2], masks2.shape[3])
        inputs["cross_attention_kwargs"]["ip_adapter_masks"] = [masks1, masks2]
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [
                0.0271,
                0.0004,
                0.0000,
                0.0011,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0037,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_ip_adapter_multiple_masks_one_adapter(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
        )
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
        )
        pipeline.set_ip_adapter_scale([[0.7, 0.7]])

        inputs = self.get_dummy_inputs(for_masks=True)
        masks = inputs["cross_attention_kwargs"]["ip_adapter_masks"]
        processor = IPAdapterMaskProcessor()
        masks = processor.preprocess(masks)
        masks = masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])
        inputs["cross_attention_kwargs"]["ip_adapter_masks"] = [masks]
        ip_images = inputs["ip_adapter_image"]
        inputs["ip_adapter_image"] = [[image[0] for image in ip_images]]
        with self.get_fixed_randn_tensor_patch(pipeline, shape=(1, 4, 128, 128)):
            images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [0.35761628, 0.39357206, 0.43524706, 0.3571607, 0.38741112, 0.43580052, 0.37814528, 0.3915079, 0.44959208]
        )
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4
