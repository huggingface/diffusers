# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    is_flaky,
    load_pt,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)


enable_full_determinism()


class IPAdapterNightlyTestsMixin(unittest.TestCase):
    dtype = torch.float16

    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
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

        expected_slice = np.array([0.80810547, 0.88183594, 0.9296875, 0.9189453, 0.9848633, 1.0, 0.97021484, 1.0, 1.0])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [0.30444336, 0.26513672, 0.22436523, 0.2758789, 0.25585938, 0.20751953, 0.25390625, 0.24633789, 0.21923828]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

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

        expected_slice = np.array(
            [0.22167969, 0.21875, 0.21728516, 0.22607422, 0.21948242, 0.23925781, 0.22387695, 0.25268555, 0.2722168]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs(for_image_to_image=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [0.35913086, 0.265625, 0.26367188, 0.24658203, 0.19750977, 0.39990234, 0.15258789, 0.20336914, 0.5517578]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

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

        expected_slice = np.array(
            [0.27148438, 0.24047852, 0.22167969, 0.23217773, 0.21118164, 0.21142578, 0.21875, 0.20751953, 0.20019531]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_text_to_image_model_cpu_offload(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipeline.to(torch_device)

        inputs = self.get_dummy_inputs()
        output_without_offload = pipeline(**inputs).images

        pipeline.enable_model_cpu_offload()
        inputs = self.get_dummy_inputs()
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
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
        pipeline.set_ip_adapter_scale(0.7)

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array([0.1704, 0.1296, 0.1272, 0.2212, 0.1514, 0.1479, 0.4172, 0.4263, 0.4360])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_unload(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
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
            "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        )
        pipeline.to(torch_device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name=["ip-adapter_sd15.bin", "ip-adapter-plus_sd15.bin"]
        )
        pipeline.set_ip_adapter_scale([0.7, 0.3])

        inputs = self.get_dummy_inputs()
        ip_adapter_image = inputs["ip_adapter_image"]
        inputs["ip_adapter_image"] = [ip_adapter_image, [ip_adapter_image] * 2]
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array([0.5234, 0.5352, 0.5625, 0.5713, 0.5947, 0.6206, 0.5786, 0.6187, 0.6494])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_text_to_image_face_id(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=self.dtype
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
        id_embeds = load_pt("https://huggingface.co/datasets/fabiorigano/testing-images/resolve/main/ai_face2.ipadpt")[
            0
        ]
        id_embeds = id_embeds.reshape((2, 1, 1, 512))
        inputs["ip_adapter_image_embeds"] = [id_embeds]
        inputs["ip_adapter_image"] = None
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.3237, 0.3186, 0.3406, 0.3154, 0.2942, 0.3220, 0.3188, 0.3528, 0.3242])
        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4


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
        pipeline.enable_model_cpu_offload()
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs()
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [
                0.09630299,
                0.09551358,
                0.08480701,
                0.09070173,
                0.09437338,
                0.09264627,
                0.08883232,
                0.09287417,
                0.09197289,
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
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.0596, 0.0539, 0.0459, 0.0580, 0.0560, 0.0548, 0.0501, 0.0563, 0.0500])

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
        pipeline.enable_model_cpu_offload()
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs(for_image_to_image=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [
                0.06513795,
                0.07009393,
                0.07234055,
                0.07426041,
                0.07002589,
                0.06415862,
                0.07827643,
                0.07962808,
                0.07411247,
            ]
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
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array(
            [
                0.07126552,
                0.07025367,
                0.07348302,
                0.07580167,
                0.07467338,
                0.06918576,
                0.07480252,
                0.08279955,
                0.08547315,
            ]
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
        pipeline.enable_model_cpu_offload()
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

        inputs = self.get_dummy_inputs(for_inpainting=True)
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        image_slice.tolist()

        expected_slice = np.array(
            [0.14181179, 0.1493012, 0.14283323, 0.14602411, 0.14915377, 0.15015268, 0.14725655, 0.15009224, 0.15164584]
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
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        image_slice.tolist()

        expected_slice = np.array([0.1398, 0.1476, 0.1407, 0.1442, 0.1470, 0.1480, 0.1449, 0.1481, 0.1494])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_ip_adapter_single_mask(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
        )
        pipeline.enable_model_cpu_offload()
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
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [0.7307304, 0.73450166, 0.73731124, 0.7377061, 0.7318013, 0.73720926, 0.74746597, 0.7409929, 0.74074936]
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
        pipeline.enable_model_cpu_offload()
        pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"] * 2
        )
        pipeline.set_ip_adapter_scale([0.7] * 2)

        inputs = self.get_dummy_inputs(for_masks=True)
        masks = inputs["cross_attention_kwargs"]["ip_adapter_masks"]
        processor = IPAdapterMaskProcessor()
        masks = processor.preprocess(masks)
        inputs["cross_attention_kwargs"]["ip_adapter_masks"] = masks
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [0.79474676, 0.7977683, 0.8013954, 0.7988008, 0.7970615, 0.8029355, 0.80614823, 0.8050743, 0.80627424]
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
        pipeline.enable_model_cpu_offload()

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
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()

        expected_slice = np.array([0.2323, 0.1026, 0.1338, 0.0638, 0.0662, 0.0000, 0.0000, 0.0000, 0.0199])

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4

    def test_ip_adapter_multiple_masks_one_adapter(self):
        image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
        )
        pipeline.enable_model_cpu_offload()
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
        images = pipeline(**inputs).images
        image_slice = images[0, :3, :3, -1].flatten()
        expected_slice = np.array(
            [0.79474676, 0.7977683, 0.8013954, 0.7988008, 0.7970615, 0.8029355, 0.80614823, 0.8050743, 0.80627424]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
        assert max_diff < 5e-4
