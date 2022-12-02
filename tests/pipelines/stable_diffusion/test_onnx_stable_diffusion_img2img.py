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

import unittest

import numpy as np

from diffusers import LMSDiscreteScheduler, OnnxStableDiffusionImg2ImgPipeline
from diffusers.utils.testing_utils import is_onnx_available, load_image, require_onnxruntime, require_torch_gpu, slow

from ...test_pipelines_onnx_common import OnnxPipelineTesterMixin


if is_onnx_available():
    import onnxruntime as ort


class OnnxStableDiffusionPipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    # FIXME: add fast tests
    pass


@slow
@require_onnxruntime
@require_torch_gpu
class OnnxStableDiffusionImg2ImgPipelineIntegrationTests(unittest.TestCase):
    @property
    def gpu_provider(self):
        return (
            "CUDAExecutionProvider",
            {
                "gpu_mem_limit": "15000000000",  # 15GB
                "arena_extend_strategy": "kSameAsRequested",
            },
        )

    @property
    def gpu_options(self):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        return options

    def test_inference_default_pndm(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        # using the PNDM scheduler by default
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="onnx",
            provider=self.gpu_provider,
            sess_options=self.gpu_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        generator = np.random.RandomState(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=10,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.4909, 0.5059, 0.5372, 0.4623, 0.4876, 0.5049, 0.4820, 0.4956, 0.5019])
        # TODO: lower the tolerance after finding the cause of onnxruntime reproducibility issues
        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2

    def test_inference_k_lms(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        lms_scheduler = LMSDiscreteScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler", revision="onnx"
        )
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="onnx",
            scheduler=lms_scheduler,
            provider=self.gpu_provider,
            sess_options=self.gpu_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        generator = np.random.RandomState(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=10,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.7950, 0.7923, 0.7903, 0.5516, 0.5501, 0.5476, 0.4965, 0.4933, 0.4910])
        # TODO: lower the tolerance after finding the cause of onnxruntime reproducibility issues
        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2
